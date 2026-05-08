# -*- coding: utf-8 -*-
"""
SimpleAgent 增强模块测试脚本
测试所有增强模块的功能
"""
import sys
import os
import io

# 设置标准输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from my_agent.enhanced import (
    QueryComplexityClassifier,
    DynamicRouter,
    QueryTier,
    RetrievalStrategy,
    PersonaExtractor,
    PersonaMemory,
    CategoryRAG,
    CognitiveDomain,
    HallucinationDetector,
    DeterministicCitation,
    MultiIndexRetrieval,
    VectorIndex,
    KeywordIndex,
    GraphIndex,
    StreamingOutput,
)
import asyncio


def test_query_router():
    """测试查询复杂度分类器"""
    print("=" * 80)
    print("测试查询复杂度分类器")
    print("=" * 80)
    
    router = DynamicRouter()
    
    test_queries = [
        ("What is the capital of France?", QueryTier.TIER_1_SIMPLE, RetrievalStrategy.VECTOR_RAG),
        ("List the three main components of a car engine and explain each one", QueryTier.TIER_2_MULTI_FACT, RetrievalStrategy.TREE_REASONING),
        ("Compare the economic policies of the US and China", QueryTier.TIER_3_CROSS_REF, RetrievalStrategy.HYBRID_AHR),
        ("Based on the research papers from 2020-2025, synthesize a comprehensive analysis of AI's impact on healthcare, education, and finance", QueryTier.TIER_4_SYNTHESIS, RetrievalStrategy.HYBRID_TREE_ENSEMBLE),
    ]
    
    all_passed = True
    for query, expected_tier, expected_strategy in test_queries:
        analysis = router.route_query(query)
        print(f"\nQuery: {query}")
        print(f"  Tier: {analysis.tier.value}")
        print(f"  Strategy: {analysis.strategy.value}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Complexity Score: {analysis.complexity_score:.2f}")
        print(f"  Indicators: {analysis.indicators}")
        
        # 验证基本逻辑
        if analysis.tier != expected_tier:
            print(f"  ❌ 错误:期望 {expected_tier.value}，实际 {analysis.tier.value}")
            all_passed = False
        elif analysis.strategy != expected_strategy:
            print(f"  ❌ 错误:期望 {expected_strategy.value}，实际 {analysis.strategy.value}")
            all_passed = False
        else:
            print(f"  ✅ 通过:{expected_tier.value} 策略正确")
    
    print(f"\n查询复杂度分类器测试: {'✅ 通过' if all_passed else '❌ 失败'}")
    return all_passed


def test_persona_memory():
    """测试 Persona 记忆提取"""
    print("\n" + "=" * 80)
    print("Test Persona Memory Extraction")
    print("=" * 80)
    
    persona_memory = PersonaMemory()
    extractor = PersonaExtractor(use_llm=False)  # Use rule-based for testing
    category_rag = CategoryRAG(persona_memory)
    
    test_text = """
    My name is John. I was born in 1990 in New York. I work as a software engineer at Google.
    I love playing basketball and I have a friend named Mike who also works at Google.
    I prefer working in the morning and I am very anxious about deadlines.
    """
    
    # 提取事实
    facts = extractor.extract_facts(test_text, "test_conversation")
    for fact in facts:
        persona_memory.add_fact(fact)
    
    # 打印摘要
    print(persona_memory.get_summary())
    
    # 检索测试
    biography_facts = category_rag.retrieve("Who is John?", CognitiveDomain.BIOGRAPHY)
    print("\nRetrieving Biography facts:")
    for fact in biography_facts:
        print(f"  - {fact.fact}")
    
    work_facts = category_rag.retrieve("Where does John work?", CognitiveDomain.WORK)
    print("\nRetrieving Work facts:")
    for fact in work_facts:
        print(f"  - {fact.fact}")
    
    all_facts = category_rag.retrieve_all("Tell me about John")
    print("\nRetrieving all facts:")
    for fact in all_facts:
        print(f"  - [{fact.domain.value}] {fact.fact}")
    
    # Verify that we got some facts
    if not facts:
        print("❌ 错误:没有提取到任何事实")
        return False
    
    print("✅ Persona 记忆提取测试通过")
    return True


def test_hallucination_detector():
    """测试幻觉检测"""
    print("\n" + "=" * 80)
    print("Test Hallucination Detection")
    print("=" * 80)
    
    detector = HallucinationDetector(use_llm=False, use_search=False)
    
    # Test cases with expected outcomes
    test_cases = [
        ("Python 3.12 强制要求所有变量声明类型，不声明就报错。", True, "factual_error"),
        ("Python is a dynamically typed language.", False, "none"),
        ("In 1999, the study will show that AI will definitely change the world.", True, "temporal_inconsistency"),
        ("Because X happened, therefore Y must be true - this has been proved beyond doubt.", True, "causal_fallacy"),
        ("Experts say that 847% of people believe in aliens according to a recent study.", True, "fabrication"),
    ]
    
    all_passed = True
    for text, expected_hallucination, expected_type in test_cases:
        result = detector.detect(text)
        print(f"\nText: {text}")
        print(f"  Is Hallucination: {result.is_hallucination}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Hallucination Type: {result.hallucination_type}")
        print(f"  Evidence: {result.evidence}")
        print(f"  Correction: {result.correction_suggestion}")
        
        # 验证基本逻辑
        if result.is_hallucination != expected_hallucination:
            print(f"  ❌ 错误:期望幻觉={expected_hallucination}，实际={result.is_hallucination}")
            all_passed = False
        elif result.is_hallucination and result.hallucination_type != expected_type:
            print(f"  ❌ 错误:期望类型={expected_type}，实际={result.hallucination_type}")
            all_passed = False
        elif result.is_hallucination and not result.evidence:
            print("  ❌ 错误:检测到幻觉但没有证据")
            all_passed = False
        else:
            print(f"  ✅ 通过:检测结果正确")
    
    print(f"\n幻觉检测测试: {'✅ 通过' if all_passed else '❌ 失败'}")
    return all_passed


def test_deterministic_citation():
    """测试确定性引用"""
    print("\n" + "=" * 80)
    print("Test Deterministic Citation")
    print("=" * 80)
    
    citation_system = DeterministicCitation()
    
    # Test cases with expected outcomes
    test_cases = [
        ('"The capital of France is Paris" is a well-known fact.', True),
        ('According to a recent study, AI will change the world.', True),
        ('The capital of France is Paris.', False),
        ('As stated by John Doe et al., 2023, machine learning is transforming healthcare.', True),
    ]
    
    all_passed = True
    for text, expected_has_citation in test_cases:
        result = citation_system.extract_citations(text)
        print(f"\nText: {text}")
        print(f"  Has Citation: {result.has_citation}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Verification Status: {result.verification_status}")
        if result.citations:
            for citation in result.citations:
                print(f"  Citation: [{citation.citation_type}] {citation.content}")
        
        # 验证基本逻辑
        if result.has_citation != expected_has_citation:
            print(f"  ❌ 错误:期望引用={expected_has_citation}，实际={result.has_citation}")
            all_passed = False
        elif result.has_citation and not result.citations:
            print("  ❌ 错误:检测到引用但没有引用内容")
            all_passed = False
        else:
            print(f"  ✅ 通过:检测结果正确")
    
    print(f"\n确定性引用测试: {'✅ 通过' if all_passed else '❌ 失败'}")
    return all_passed


def test_multi_index_retrieval():
    """测试多索引混合检索"""
    print("\n" + "=" * 80)
    print("Test Multi-Index Retrieval")
    print("=" * 80)
    
    multi_index = MultiIndexRetrieval()
    
    documents = [
        {
            "id": "doc1",
            "content": "The capital of France is Paris.",
            "metadata": {"domain": "geography"},
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "id": "doc2",
            "content": "AI is changing the world rapidly.",
            "metadata": {"domain": "technology"},
            "embedding": [0.4, 0.5, 0.6],
        },
        {
            "id": "doc3",
            "content": "The Eiffel Tower is located in Paris.",
            "metadata": {"domain": "geography"},
            "embedding": [0.7, 0.8, 0.9],
        },
    ]
    
    for doc_data in documents:
        from my_agent.enhanced.multi_index_retrieval import Document
        doc = Document(
            id=doc_data["id"],
            content=doc_data["content"],
            metadata=doc_data["metadata"],
            embedding=doc_data["embedding"],
        )
        multi_index.add_document(doc)
    
    test_queries = ["Paris", "AI", "Eiffel Tower"]
    
    all_passed = True
    for query in test_queries:
        results = multi_index.search(query, top_k=2)
        print(f"\nQuery: {query}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.domain}] Score: {result.score:.4f}")
            print(f"     Content: {result.document.content}")
        
        # 验证基本逻辑
        if not results:
            print("  ❌ 错误:未返回任何结果")
            all_passed = False
        else:
            print("  ✅ 通过:返回了结果")
    
    print(f"\n多索引混合检索测试: {'✅ 通过' if all_passed else '❌ 失败'}")
    return all_passed


async def test_streaming_output():
    """测试流式输出"""
    print("\n" + "=" * 80)
    print("Test Streaming Output")
    print("=" * 80)
    
    streaming_output = StreamingOutput()
    
    # 测试响应流式输出
    response = "This is a test response. It contains multiple chunks of text."
    chunks = []
    async for event in streaming_output.stream_response(response, chunk_size=10):
        chunks.append(event)
        print(f"Event: {event.event_type.value} | Content: {event.content}")
    
    # 测试工具调用流式输出
    tool_events = []
    async for event in streaming_output.stream_tool_call("get_time", {}):
        tool_events.append(event)
        print(f"Event: {event.event_type.value} | Content: {event.content}")
    
    # 测试错误流式输出
    error_events = []
    async for event in streaming_output.stream_error("Test error"):
        error_events.append(event)
        print(f"Event: {event.event_type.value} | Content: {event.content}")
    
    # 打印事件摘要
    print("\nEvent Summary:")
    print(streaming_output.get_event_summary())
    
    # Verify we got expected events
    if not chunks or not tool_events or not error_events:
        print("❌ 错误:没有收到预期的事件")
        return False
    
    print("✅ 流式输出测试通过")
    return True


def main():
    """运行所有测试"""
    print("=" * 80)
    print("SimpleAgent 增强模块测试")
    print("=" * 80)
    
    results = []
    
    # 运行同步测试
    results.append(("Query Router", test_query_router()))
    results.append(("Persona Memory", test_persona_memory()))
    results.append(("Hallucination Detection", test_hallucination_detector()))
    results.append(("Deterministic Citation", test_deterministic_citation()))
    results.append(("Multi-Index Retrieval", test_multi_index_retrieval()))
    
    # 运行异步测试
    results.append(("流式输出", asyncio.run(test_streaming_output())))
    
    # 打印总结
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print(f"\n总体结果: {'✅ 所有测试通过' if all_passed else '❌ 部分测试失败'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
