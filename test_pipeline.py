# -*- coding: utf-8 -*-
"""
增强流水线集成测试脚本
演示6个增强模块的端到端执行流程（纯模块级，不依赖 LLM API）
"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_query_router():
    """测试查询复杂度分类器"""
    from my_agent.enhanced.query_router import DynamicRouter, QueryTier
    
    router = DynamicRouter()
    
    # Tier 1: 简单问题
    r = router.route_query("What is the capital of France?")
    assert r.tier == QueryTier.TIER_1_SIMPLE, f"Expected tier_1_simple, got {r.tier}"
    print(f"  ✅ 简单查询 → {r.tier.value}")
    
    # Tier 2: 多事实问题
    r = router.route_query("List the three main components and explain each one")
    assert r.tier == QueryTier.TIER_2_MULTI_FACT, f"Expected tier_2_multi_fact, got {r.tier}"
    print(f"  ✅ 多事实查询 → {r.tier.value}")
    
    # Tier 3: 交叉引用
    r = router.route_query("Compare the economic policies of the US and China")
    assert r.tier == QueryTier.TIER_3_CROSS_REF, f"Expected tier_3_cross_ref, got {r.tier}"
    print(f"  ✅ 交叉引用查询 → {r.tier.value}")
    
    # Tier 4: 综合合成
    r = router.route_query("Based on research papers, synthesize a comprehensive analysis of AI impact")
    assert r.tier == QueryTier.TIER_4_SYNTHESIS, f"Expected tier_4_synthesis, got {r.tier}"
    print(f"  ✅ 综合查询 → {r.tier.value}")


def test_persona_memory():
    """测试 Persona 记忆提取"""
    from my_agent.enhanced.persona_memory import PersonaMemory, CognitiveDomain
    
    memory = PersonaMemory()
    
    # 添加事实
    from my_agent.enhanced.persona_memory import PersonaFact
    f1 = PersonaFact(
        domain=CognitiveDomain.BIOGRAPHY,
        fact="My name is John",
        confidence=0.9,
        timestamp="2025-01-01",
        source="chat"
    )
    memory.add_fact(f1)
    
    facts = memory.get_all_facts()
    assert len(facts) == 1, f"Expected 1 fact, got {len(facts)}"
    assert facts[0].domain == CognitiveDomain.BIOGRAPHY
    print(f"  ✅ Persona 记忆: 添加和检索正常 ({len(facts)} 条)")


def test_hallucination_detector():
    """测试幻觉检测"""
    from my_agent.enhanced.hallucination_detector import HallucinationDetector
    
    detector = HallucinationDetector()
    
    # 正常文本
    r = detector.detect("Python is a programming language created by Guido van Rossum.")
    print(f"  ✅ 正常文本: hallucination={r.is_hallucination}")
    
    # 可能幻觉的文本（过度自信）
    r = detector.detect("As proven by the 2099 Nobel Prize committee, AI will replace all jobs by 2025.")
    print(f"  ✅ 可疑文本: hallucination={r.is_hallucination}, type={r.hallucination_type}")


def test_deterministic_citation():
    """测试确定性引用"""
    from my_agent.enhanced.deterministic_citation import DeterministicCitation
    
    cit = DeterministicCitation()
    
    # 包含引用的文本
    text = 'According to Smith et al. (2023), "deep learning has transformed NLP."'
    result = cit.extract_citations(text)
    print(f"  ✅ 引用提取: has_citation={result.has_citation}, count={len(result.citations)}")


def test_multi_index_retrieval():
    """测试多索引检索"""
    from my_agent.enhanced.multi_index_retrieval import Document, MultiIndexRetrieval
    
    idx = MultiIndexRetrieval()
    
    docs = [
        Document(id="d1", content="Python is a high-level programming language.", metadata={"domain": "tech"}),
        Document(id="d2", content="JavaScript is used for web development.", metadata={"domain": "tech"}),
        Document(id="d3", content="The weather in Tokyo is rainy today.", metadata={"domain": "weather"}),
    ]
    
    for doc in docs:
        idx.add_document(doc)
    
    results = idx.search("programming language", top_k=2)
    assert len(results) > 0, "Expected at least one result"
    print(f"  ✅ 多索引检索: {len(results)} 结果, best={results[0].score:.3f}")


def test_agent_enhanced_modules():
    """测试 SimpleAgent 增强模块初始化"""
    from my_agent.agent import SimpleAgent
    
    agent = SimpleAgent(enable_enhanced=True)
    
    # 验证增强模块已初始化
    assert agent._router is not None, "Router should be initialized"
    assert agent._persona_memory is not None, "Persona memory should be initialized"
    assert agent._hallucination_detector is not None, "Hallucination detector should be initialized"
    assert agent._citation_system is not None, "Citation system should be initialized"
    assert agent._multi_index is not None, "Multi-index should be initialized"
    
    # 测试多索引属性访问（修复 Bug: agent.multi_index → agent._multi_index）
    from my_agent.enhanced.multi_index_retrieval import Document
    doc = Document(id="t1", content="Test document for agent integration.", metadata={"domain": "test"})
    agent._multi_index.add_document(doc)
    
    results = agent._multi_index.search("test document", top_k=1)
    assert len(results) > 0, "Should find test document"
    
    print(f"  ✅ SimpleAgent 增强模块全部初始化正常")
    
    agent.close()


def test_memory_store():
    """测试记忆存储"""
    from my_agent.agent import SimpleAgent
    
    agent = SimpleAgent(enable_enhanced=False)
    
    # 测试记忆提取
    result = agent.memory_store.recall("hello")
    assert result is not None
    print(f"  ✅ 记忆存储: recall 正常, {len(result.matched_lessons)} lessons matched")
    
    agent.close()


def main():
    print("=" * 80)
    print("SimpleAgent 增强流水线集成测试（纯模块级）")
    print("=" * 80)
    
    tests = [
        ("查询复杂度分类", test_query_router),
        ("Persona 记忆", test_persona_memory),
        ("幻觉检测", test_hallucination_detector),
        ("确定性引用", test_deterministic_citation),
        ("多索引检索", test_multi_index_retrieval),
        ("Agent 增强模块初始化", test_agent_enhanced_modules),
        ("记忆存储", test_memory_store),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n{'─' * 60}")
        print(f"测试: {name}")
        print("─" * 60)
        try:
            test_fn()
            print(f"  ✅ {name}: 通过")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: 失败 - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
