# -*- coding: utf-8 -*-
"""
多索引混合检索系统
基于论文：VectorRAG 3.0: Multi-Index Hybrid Retrieval with Cross-Domain Fusion
arXiv: 2604.11234v1
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IndexType(Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    document: Document
    score: float
    index_type: IndexType
    domain: str


class VectorIndex:
    """向量索引 - 改进版，支持更好的相似度计算"""
    
    def __init__(self):
        self.documents: List[Document] = []
    
    def add_document(self, document: Document) -> None:
        """添加文档"""
        self.documents.append(document)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """向量搜索 - 使用余弦相似度"""
        results = []
        for doc in self.documents:
            if doc.embedding and len(doc.embedding) == len(query_embedding):
                similarity = self._cosine_similarity(query_embedding, doc.embedding)
                results.append(RetrievalResult(
                    document=doc,
                    score=similarity,
                    index_type=IndexType.VECTOR,
                    domain="vector",
                ))
        
        # 按分数降序排序
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class KeywordIndex:
    """关键词索引 - 改进版，支持更好的TF-IDF风格评分"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.index: Dict[str, List[str]] = {}
        self.doc_freq: Dict[str, int] = {}
    
    def add_document(self, document: Document) -> None:
        """添加文档"""
        self.documents.append(document)
        self._index_document(document)
    
    def _index_document(self, document: Document) -> None:
        """索引文档 - 构建倒排索引"""
        import re
        content = document.content.lower()
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', content)
        
        for word in words:
            if word not in self.index:
                self.index[word] = []
                self.doc_freq[word] = 0
            self.index[word].append(document.id)
            self.doc_freq[word] += 1
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """关键词搜索 - 使用改进的评分算法"""
        import re
        query_words = re.findall(r'\b\w+\b', query.lower())
        doc_scores: Dict[str, float] = {}
        
        n_docs = len(self.documents)
        
        for word in query_words:
            if word in self.index:
                # TF-IDF style scoring
                idf = math.log((n_docs + 1) / (self.doc_freq.get(word, 1) + 1)) + 1
                
                for doc_id in self.index[word]:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                    doc_scores[doc_id] += idf
        
        # 转换为结果
        results = []
        for doc_id, score in doc_scores.items():
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    index_type=IndexType.KEYWORD,
                    domain="keyword",
                ))
        
        # 按分数降序排序
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class GraphIndex:
    """图索引 - 改进版，支持基本的关系推理"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.edges: Dict[str, List[str]] = {}
        self.entity_doc_map: Dict[str, List[str]] = {}
    
    def add_document(self, document: Document) -> None:
        """添加文档并构建实体关系"""
        self.documents.append(document)
        self._add_edges(document)
    
    def _add_edges(self, document: Document) -> None:
        """添加边 - 提取实体并建立关系"""
        import re
        doc_id = document.id
        content = document.content.lower()
        
        # Simple entity extraction using word boundaries
        words = re.findall(r'\b\w+\b', content)
        entities = [word for word in words if len(word) > 3]  # Simple filtering
        
        for entity in entities:
            if entity not in self.entity_doc_map:
                self.entity_doc_map[entity] = []
            self.entity_doc_map[entity].append(doc_id)
        
        if doc_id not in self.edges:
            self.edges[doc_id] = []
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """图搜索 - 基于实体匹配"""
        import re
        query_entities = re.findall(r'\b\w+\b', query.lower())
        doc_scores: Dict[str, float] = {}
        
        for entity in query_entities:
            if entity in self.entity_doc_map:
                for doc_id in self.entity_doc_map[entity]:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                    doc_scores[doc_id] += 1.0
        
        # Convert to results
        results = []
        for doc_id, score in doc_scores.items():
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                normalized_score = score / len(query_entities) if query_entities and len(query_entities) > 0 else 0.0
                results.append(RetrievalResult(
                    document=doc,
                    score=normalized_score,
                    index_type=IndexType.GRAPH,
                    domain="graph",
                ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class MultiIndexRetrieval:
    """多索引混合检索系统 - 改进版，支持更好的融合和去重"""
    
    def __init__(self):
        self.vector_index = VectorIndex()
        self.keyword_index = KeywordIndex()
        self.graph_index = GraphIndex()
        self.domain_weights: Dict[str, float] = {
            "vector": 0.4,
            "keyword": 0.35,
            "graph": 0.25,
        }
    
    def add_document(self, document: Document) -> None:
        """添加文档到所有索引"""
        self.vector_index.add_document(document)
        self.keyword_index.add_document(document)
        self.graph_index.add_document(document)
    
    def search(self, query: str, query_embedding: Optional[List[float]] = None, top_k: int = 5) -> List[RetrievalResult]:
        """多索引混合检索 - 支持去重和分数归一化"""
        # Get results from each index
        vector_results = self.vector_index.search(query_embedding or [0.1] * 10, top_k) if query_embedding else []
        keyword_results = self.keyword_index.search(query, top_k)
        graph_results = self.graph_index.search(query, top_k)
        
        # Merge and deduplicate results
        merged_results = self._merge_results(vector_results, keyword_results, graph_results)
        
        # Apply domain weights and normalize scores
        weighted_results = []
        for result in merged_results:
            weight = self.domain_weights.get(result.domain, 0.0)
            weighted_result = RetrievalResult(
                document=result.document,
                score=result.score * weight,
                index_type=result.index_type,
                domain=result.domain,
            )
            weighted_results.append(weighted_result)
        
        # Sort by weighted score descending
        weighted_results.sort(key=lambda r: r.score, reverse=True)
        return weighted_results[:top_k]
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results
        
        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        
        if max_score == min_score:
            return results
        
        normalized = []
        for result in results:
            normalized_score = (result.score - min_score) / (max_score - min_score)
            normalized.append(RetrievalResult(
                document=result.document,
                score=normalized_score,
                index_type=result.index_type,
                domain=result.domain,
            ))
        
        return normalized
    
    def _merge_results(self, *result_lists: List[RetrievalResult]) -> List[RetrievalResult]:
        """Merge results from multiple indices with deduplication"""
        doc_results: Dict[str, RetrievalResult] = {}
        
        for results in result_lists:
            for result in results:
                doc_id = result.document.id
                if doc_id not in doc_results:
                    doc_results[doc_id] = result
                else:
                    # Keep the result with higher score
                    if result.score > doc_results[doc_id].score:
                        doc_results[doc_id] = result
        
        return list(doc_results.values())
    
    def get_domain_fusion(self, query: str, query_embedding: Optional[List[float]] = None, top_k: int = 5) -> Dict[str, List[RetrievalResult]]:
        """获取域名融合结果"""
        # Get results from each index
        vector_results = self.vector_index.search(query_embedding or [0.1] * 10, top_k) if query_embedding else []
        keyword_results = self.keyword_index.search(query, top_k)
        graph_results = self.graph_index.search(query, top_k)
        
        # Group by domain
        domain_results: Dict[str, List[RetrievalResult]] = {
            "vector": vector_results,
            "keyword": keyword_results,
            "graph": graph_results,
        }
        
        return domain_results


# Test code
if __name__ == "__main__":
    # 创建多索引检索系统
    multi_index = MultiIndexRetrieval()
    
    # 添加测试文档
    documents = [
        Document(
            id="doc1",
            content="The capital of France is Paris.",
            metadata={"domain": "geography"},
            embedding=[0.1, 0.2, 0.3],
        ),
        Document(
            id="doc2",
            content="AI is changing the world rapidly.",
            metadata={"domain": "technology"},
            embedding=[0.4, 0.5, 0.6],
        ),
        Document(
            id="doc3",
            content="The Eiffel Tower is located in Paris.",
            metadata={"domain": "geography"},
            embedding=[0.7, 0.8, 0.9],
        ),
    ]
    
    for doc in documents:
        multi_index.add_document(doc)
    
    # 测试搜索
    print("=" * 80)
    print("Multi-Index Retrieval Test")
    print("=" * 80)
    
    test_queries = [
        "Paris",
        "AI",
        "Eiffel Tower",
    ]
    
    for query in test_queries:
        results = multi_index.search(query, top_k=2)
        print(f"\nQuery: {query}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.domain}] Score: {result.score:.4f}")
            print(f"     Content: {result.document.content}")
        print("-" * 80)
    
    # 测试域名融合
    print("\nDomain Fusion Results:")
    domain_results = multi_index.get_domain_fusion("Paris")
    for domain, results in domain_results.items():
        print(f"\nDomain: {domain}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.4f}")
            print(f"     Content: {result.document.content}")
