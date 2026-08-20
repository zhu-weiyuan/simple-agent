# -*- coding: utf-8 -*-
"""
Tree Reasoning（树状推理）Retriever 原型
演示 Tier 2 Multi-Fact（多事实）查询的差异化执行：
1. Query Decomposition（查询分解）→ 拆成子问题
2. Parallel Retrieval（并行检索）→ 多索引并发搜索
3. Result Aggregation（结果聚合）→ 去重 + 评分融合 + 结构化输出
"""
from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

# 复用现有的索引类型和文档结构
from .multi_index_retrieval import (
    MultiIndexRetrieval, Document, RetrievalResult, IndexType
)


class DecompositionStrategy(Enum):
    """查询分解策略"""
    RULE_BASED = "rule_based"      # 规则分解（快、省 token）
    LLM_BASED = "llm_based"        # LLM 分解（准、灵活）
    HYBRID = "hybrid"              # 混合：规则先切，LLM 补全


@dataclass
class SubQuery:
    """子查询"""
    id: str
    text: str
    focus: str              # 关注点，如 "component", "parameter", "step"
    priority: int = 1       # 优先级，高优先先检索


@dataclass
class TreeReasoningResult:
    """树状推理最终结果"""
    original_query: str
    sub_queries: List[SubQuery]
    retrieval_results: Dict[str, List[RetrievalResult]]  # sub_query_id -> results
    aggregated_answer: str
    confidence: float
    evidence_map: Dict[str, List[str]] = field(default_factory=dict)  # 答案片段 -> 证据来源


class TreeReasoningRetriever:
    """
    Tier 2 专用检索器：树状分解 → 并行检索 → 结构化聚合
    
    适用场景：
    - "List three components of X and explain each"
    - "What are the steps to do Y?"
    - "Compare A and B on dimensions C, D, E"
    """
    
    def __init__(
        self,
        multi_index: MultiIndexRetrieval,
        decomposition_strategy: DecompositionStrategy = DecompositionStrategy.RULE_BASED,
        max_sub_queries: int = 5,
        top_k_per_sub: int = 3,
        max_workers: int = 4,
    ):
        self.multi_index = multi_index
        self.strategy = decomposition_strategy
        self.max_sub_queries = max_sub_queries
        self.top_k_per_sub = top_k_per_sub
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _word_to_num(self, word: str) -> int:
        """英文数字词转数字"""
        num_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'first': 1, 'second': 2, 'third': 3
        }
        return num_words.get(word.lower(), 1)
    
    # ============================================================
    # 1. Query Decomposition（查询分解）
    # ============================================================
    
    def decompose(self, query: str) -> List[SubQuery]:
        """主入口：根据策略分解查询"""
        if self.strategy == DecompositionStrategy.RULE_BASED:
            return self._rule_decompose(query)
        elif self.strategy == DecompositionStrategy.LLM_BASED:
            return self._llm_decompose(query)
        else:
            # Hybrid: 规则先切，不够再用 LLM 补
            rule_subs = self._rule_decompose(query)
            if len(rule_subs) < 2:
                return self._llm_decompose(query)
            return rule_subs
    
    def _rule_decompose(self, query: str) -> List[SubQuery]:
        """规则分解：基于关键词模式，零 token 成本"""
        subs = []
        q_lower = query.lower()
        
        # 模式 1: "list N components/parts/steps of X" (支持数字词 three, two 等)
        m = re.search(
            r'\b(list|enumerate|identify)\b.*?\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b.*?\b(components?|parts?|steps?|elements?|factors?|aspects?)\b.*?\b(of|for)\b\s+([^,\.]+?)(?:\s+(?:and|,|;|each|every|all|explain|describe)|\s*$)',
            q_lower, re.I
        )
        if m:
            count = self._word_to_num(m.group(2))
            item_type = m.group(3)
            target = m.group(5).strip(' ?。，')
            for i in range(1, min(count + 1, self.max_sub_queries)):
                subs.append(SubQuery(
                    id=f"sq_{i}",
                    text=f"{target} {item_type} {i}",
                    focus=item_type.rstrip('s'),
                    priority=10 - i
                ))
            return subs[:self.max_sub_queries]
        
        # 模式 1b: "list the three main components of X" (main 等修饰词)
        m = re.search(
            r'\b(list|enumerate|identify)\b.*?\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b.*?(main\s+)?\b(components?|parts?|steps?|elements?|factors?|aspects?)\b.*?\b(of|for)\b\s+([^,\.]+?)(?:\s+(?:and|,|;|each|every|all|explain|describe)|\s*$)',
            q_lower, re.I
        )
        if m:
            count = self._word_to_num(m.group(2))
            item_type = m.group(4)
            target = m.group(6).strip(' ?。，')
            for i in range(1, min(count + 1, self.max_sub_queries)):
                subs.append(SubQuery(
                    id=f"sq_{i}",
                    text=f"{target} {item_type} {i}",
                    focus=item_type.rstrip('s'),
                    priority=10 - i
                ))
            return subs[:self.max_sub_queries]
        
        # 模式 2: "what are the steps to ..."
        m = re.search(r'\b(what|how)\b.*?\b(steps?|procedure|process)\b.*?\b(to|for)\b\s+(.+)', q_lower, re.I)
        if m:
            target = m.group(4).strip(' ?。，')
            step_keywords = ["prepare", "configure", "install", "run", "test", "deploy", "verify"]
            for i, kw in enumerate(step_keywords[:self.max_sub_queries]):
                subs.append(SubQuery(
                    id=f"sq_{i}",
                    text=f"{kw} step for {target}",
                    focus="step",
                    priority=10 - i
                ))
            return subs
        
        # 模式 3: 多个 WH-词连用 "what X and what Y" / "what X, what Y"
        wh_matches = list(re.finditer(r'\b(what|who|when|where|how|why)\b[^?，。]*?[?？]', query, re.I))
        if len(wh_matches) >= 2:
            for i, m in enumerate(wh_matches[:self.max_sub_queries]):
                subs.append(SubQuery(
                    id=f"sq_{i}",
                    text=m.group(0).strip(),
                    focus="fact",
                    priority=10 - i
                ))
            return subs
        
        # 模式 4: "explain/describe each ..."
        m = re.search(r'\b(explain|describe|detail)\b.*?\b(each|every|all)\b.*?\b(components?|parts?|steps?|elements?)\b', q_lower, re.I)
        if m:
            subs.append(SubQuery(
                id="sq_1",
                text=query,
                focus="overview",
                priority=10
            ))
            return subs
        
        # 兜底：整体查询
        return [SubQuery(id="sq_1", text=query, focus="general", priority=10)]
    
    def _llm_decompose(self, query: str) -> List[SubQuery]:
        """LLM 分解：需要外部 LLM Client，这里留接口"""
        # TODO: 接入 LLM，Prompt 示例：
        # "将复杂问题分解为 3-5 个可独立检索的子问题，每个子问题聚焦一个事实点。
        # 返回 JSON: [{id, text, focus, priority}]"
        # 这里返回规则兜底，避免依赖
        return self._rule_decompose(query)
    
    # ============================================================
    # 2. Parallel Retrieval（并行检索）
    # ============================================================
    
    def retrieve_parallel(self, sub_queries: List[SubQuery]) -> Dict[str, List[RetrievalResult]]:
        """并行执行多个子查询检索，返回 {sub_query_id: [results]}"""
        future_to_sq = {}
        
        for sq in sub_queries:
            future = self.executor.submit(self._retrieve_single, sq)
            future_to_sq[future] = sq.id
        
        results_map = {}
        for future in as_completed(future_to_sq):
            sq_id = future_to_sq[future]
            try:
                results_map[sq_id] = future.result(timeout=10)
            except Exception as e:
                print(f"[TreeReasoning] 子查询 {sq_id} 检索失败: {e}")
                results_map[sq_id] = []
        
        return results_map
    
    def _retrieve_single(self, sub_query: SubQuery) -> List[RetrievalResult]:
        """单子查询检索：走 Multi-Index 融合检索"""
        return self.multi_index.search(sub_query.text, top_k=self.top_k_per_sub)
    
    # ============================================================
    # 3. Result Aggregation（结果聚合）
    # ============================================================
    
    def aggregate(
        self,
        original_query: str,
        sub_queries: List[SubQuery],
        retrieval_results: Dict[str, List[RetrievalResult]]
    ) -> TreeReasoningResult:
        """
        聚合逻辑：
        1. 按子查询分组保留结构
        2. 全局去重（同一文档可能被多个子查询命中）
        3. 评分融合：子查询优先级 * 检索分数
        4. 生成结构化答案骨架
        """
        doc_sources: Dict[str, Dict[str, Any]] = {}
        
        for sq in sub_queries:
            results = retrieval_results.get(sq.id, [])
            for r in results:
                doc_id = r.document.id
                if doc_id not in doc_sources:
                    doc_sources[doc_id] = {
                        "document": r.document,
                        "scores": [],
                        "sub_query_ids": [],
                        "domains": []
                    }
                weighted_score = r.score * (sq.priority / 10.0)
                doc_sources[doc_id]["scores"].append(weighted_score)
                doc_sources[doc_id]["sub_query_ids"].append(sq.id)
                doc_sources[doc_id]["domains"].append(r.domain)
        
        fused_results = []
        for doc_id, info in doc_sources.items():
            max_score = max(info["scores"]) if info["scores"] else 0
            coverage_bonus = len(set(info["sub_query_ids"])) * 0.05
            final_score = min(max_score + coverage_bonus, 1.0)
            
            fused_results.append({
                "document": info["document"],
                "score": final_score,
                "sub_query_ids": info["sub_query_ids"],
                "domains": list(set(info["domains"]))
            })
        
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        
        aggregated_answer = self._build_structured_answer(
            original_query, sub_queries, fused_results, retrieval_results
        )
        
        evidence_map = self._build_evidence_map(sub_queries, fused_results)
        
        coverage = len([sq for sq in sub_queries if retrieval_results.get(sq.id)]) / len(sub_queries)
        avg_score = sum(r["score"] for r in fused_results) / len(fused_results) if fused_results else 0
        confidence = (coverage * 0.6 + avg_score * 0.4)
        
        return TreeReasoningResult(
            original_query=original_query,
            sub_queries=sub_queries,
            retrieval_results=retrieval_results,
            aggregated_answer=aggregated_answer,
            confidence=confidence,
            evidence_map=evidence_map
        )
    
    def _build_structured_answer(
        self,
        query: str,
        sub_queries: List[SubQuery],
        fused_results: List[Dict],
        retrieval_results: Dict[str, List[RetrievalResult]]
    ) -> str:
        """构建结构化答案：按子问题组织，附带证据引用"""
        parts = [f"## 综合回答: {query}\n"]
        
        for sq in sub_queries:
            results = retrieval_results.get(sq.id, [])
            if not results:
                parts.append(f"### {sq.focus}: (未检索到相关信息)")
                continue
            
            parts.append(f"### {sq.focus} (子问题: {sq.text})")
            for i, r in enumerate(results[:2], 1):
                doc = r.document
                parts.append(f"  {i}. {doc.content[:200]}... [来源: {r.domain}, 分数: {r.score:.3f}]")
            parts.append("")
        
        cross_docs = [r for r in fused_results if len(r["sub_query_ids"]) > 1]
        if cross_docs:
            parts.append("### 关联知识 (多子问题共同指向)")
            for r in cross_docs[:2]:
                parts.append(f"  - {r['document'].content[:200]}... [覆盖: {r['sub_query_ids']}]")
        
        return "\n".join(parts)
    
    def _build_evidence_map(
        self,
        sub_queries: List[SubQuery],
        fused_results: List[Dict]
    ) -> Dict[str, List[str]]:
        """构建证据映射：用于后续 Citation Verification（引用验证）"""
        evidence_map = {}
        for r in fused_results:
            doc = r["document"]
            key = f"{doc.id}:{doc.content[:50]}"
            evidence_map[key] = [
                f"sub_query:{sq_id}" for sq_id in r["sub_query_ids"]
            ] + [f"domain:{d}" for d in r["domains"]]
        return evidence_map
    
    # ============================================================
    # 一站式调用
    # ============================================================
    
    def retrieve_and_reason(self, query: str) -> TreeReasoningResult:
        """对外统一入口：分解 → 并行检索 → 聚合"""
        sub_queries = self.decompose(query)
        print(f"[TreeReasoning] 分解为 {len(sub_queries)} 个子查询: {[sq.text for sq in sub_queries]}")
        
        retrieval_results = self.retrieve_parallel(sub_queries)
        
        result = self.aggregate(query, sub_queries, retrieval_results)
        print(f"[TreeReasoning] 聚合完成，置信度: {result.confidence:.2f}")
        
        return result
    
    def close(self):
        self.executor.shutdown(wait=True)


# ============================================================
# 集成到 SimpleAgent 的最小改动示例
# ============================================================

def integrate_into_agent(agent):
    """把 TreeReasoningRetriever 挂载到 SimpleAgent，按策略分发"""
    from .enhanced.query_router import RetrievalStrategy
    
    tree_retriever = TreeReasoningRetriever(
        multi_index=agent._multi_index,
        decomposition_strategy=DecompositionStrategy.HYBRID,
    )
    agent._tree_retriever = tree_retriever
    
    original_pipeline = agent._run_enhanced_pipeline
    
    def enhanced_pipeline_with_tree_reasoning(user_input: str):
        from .enhanced.query_router import DynamicRouter
        router = DynamicRouter()
        analysis = router.route_query(user_input)
        
        if analysis.strategy == RetrievalStrategy.TREE_REASONING:
            print(f"[Agent] 检测到 Tier 2 Multi-Fact，启用 Tree Reasoning 检索")
            tree_result = tree_retriever.retrieve_and_reason(user_input)
            
            extra_context = f"[Tree Reasoning 检索结果]\n{tree_result.aggregated_answer}"
            return tree_result.aggregated_answer, {"tree_reasoning": True}
        
        return original_pipeline(user_input)
    
    agent._run_enhanced_pipeline = enhanced_pipeline_with_tree_reasoning
    return agent


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    multi_index = MultiIndexRetrieval()
    
    test_docs = [
        Document(
            id="doc1",
            content="The three main components of a car engine are: cylinder block, crankshaft connecting rod mechanism, and valve train. The cylinder block is the skeleton of the engine.",
            metadata={"domain": "automotive"},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        ),
        Document(
            id="doc2",
            content="Cylinder block: withstands piston motion and combustion pressure, usually made of cast iron or aluminum alloy. Contains cylinders, water jackets, oil passages.",
            metadata={"domain": "automotive"},
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
        ),
        Document(
            id="doc3",
            content="Crankshaft connecting rod mechanism: converts piston reciprocating motion to crankshaft rotation. Includes piston, connecting rod, crankshaft, flywheel.",
            metadata={"domain": "automotive"},
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
        ),
        Document(
            id="doc4",
            content="Valve train: controls intake and exhaust valve opening and closing. Includes camshaft, tappet, rocker arm, timing chain/belt.",
            metadata={"domain": "automotive"},
            embedding=[0.4, 0.5, 0.6, 0.7, 0.8],
        ),
        Document(
            id="doc5",
            content="Engine cooling system: water pump, radiator, thermostat, fan. Prevents engine overheating.",
            metadata={"domain": "automotive"},
            embedding=[0.5, 0.6, 0.7, 0.8, 0.9],
        ),
    ]
    
    for doc in test_docs:
        multi_index.add_document(doc)
    
    retriever = TreeReasoningRetriever(multi_index)
    
    test_query = "List the three main components of a car engine and explain each one"
    
    print("=" * 80)
    print("Tree Reasoning Retriever 测试")
    print("=" * 80)
    print(f"\n原始查询: {test_query}\n")
    
    result = retriever.retrieve_and_reason(test_query)
    
    print("\n" + "=" * 80)
    print("最终聚合答案:")
    print("=" * 80)
    print(result.aggregated_answer)
    
    print("\n" + "=" * 80)
    print(f"置信度: {result.confidence:.2f}")
    print(f"证据映射: {json.dumps(result.evidence_map, ensure_ascii=False, indent=2)}")
    
    retriever.close()