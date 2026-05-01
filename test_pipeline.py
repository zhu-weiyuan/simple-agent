"""
增强流水线集成测试脚本
演示6个增强模块的端到端执行流程
"""
import sys
import os
import io

# 设置标准输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from my_agent.agent import SimpleAgent
from my_agent.core.pipeline import IntegratedPipeline


def test_integrated_pipeline():
    """测试增强流水线集成"""
    print("=" * 80)
    print("SimpleAgent 增强流水线集成测试")
    print("=" * 80)
    
    # 创建 Agent 实例（启用增强模块）
    agent = SimpleAgent(enable_enhanced=True)
    
    # 添加一些测试数据到多索引
    from my_agent.enhanced.multi_index_retrieval import Document
    
    test_docs = [
        Document(
            id="doc1",
            content="John is a software engineer at Google.",
            metadata={"domain": "biography"},
        ),
        Document(
            id="doc2",
            content="John prefers Python over Java for backend development.",
            metadata={"domain": "preferences"},
        ),
        Document(
            id="doc3",
            content="John lives in San Francisco with his wife Sarah.",
            metadata={"domain": "biography"},
        ),
    ]
    
    print("\nAdding test documents to multi-index...")
    for doc in test_docs:
        agent.multi_index.add_document(doc)
    print(f"Added {len(test_docs)} documents")
    
    # 测试1: 简单查询
    print("\n" + "-" * 80)
    print("测试1: 简单查询 (Tier 1)")
    print("-" * 80)
    
    test_input_1 = "What time is it?"
    result_1, context_1 = IntegratedPipeline(agent).execute(test_input_1)
    
    print(f"\n用户输入: {test_input_1}")
    print(f"查询层级: {context_1.query_tier}")
    print(f"路由策略: {context_1.routing_strategy}")
    print(f"幻觉检测: 未触发")
    print(f"引用验证: 无")
    print(f"\n最终回复:\n{result_1}")
    
    # 测试2: Persona记忆匹配
    print("\n" + "-" * 80)
    print("测试2: Persona记忆匹配 (Tier 2)")
    print("-" * 80)
    
    test_input_2 = "Tell me about John's work and preferences."
    result_2, context_2 = IntegratedPipeline(agent).execute(test_input_2)
    
    print(f"\n用户输入: {test_input_2}")
    print(f"查询层级: {context_2.query_tier}")
    print(f"路由策略: {context_2.routing_strategy}")
    print(f"Persona事实:")
    for fact in context_2.persona_facts:
        print(f"  [{fact['domain']}] {fact['fact']} ({fact['confidence']:.2f})")
    
    if context_2.retrieval_results:
        print("检索结果:")
        for r in context_2.retrieval_results:
            print(f"  [{r['domain']}] Score:{r['score']:.3f} {r['content']}")
    
    # 测试3: 幻觉检测触发
    print("\n" + "-" * 80)
    print("测试3: 幻觉检测触发 (复杂查询)")
    print("-" * 80)
    
    test_input_3 = "Python 3.12 强制要求所有变量声明类型，不声明就报错。这个说法对吗？"
    result_3, context_3 = IntegratedPipeline(agent).execute(test_input_3)
    
    print(f"\n用户输入: {test_input_3}")
    print(f"查询层级: {context_3.query_tier}")
    print(f"路由策略: {context_3.routing_strategy}")
    print(f"幻觉检测: {'触发' if context_3.is_hallucination else '未触发'}")
    if context_3.is_hallucination:
        print(f"  类型: {context_3.hallucination_type}")
        print(f"  证据: {context_3.hallucination_evidence}")
        print(f"  纠正建议: {context_3.hallucination_correction}")
    
    # 测试4: 引用验证触发
    print("\n" + "-" * 80)
    print("测试4: 引用验证触发 (复杂查询)")
    print("-" * 80)
    
    test_input_4 = "As stated by John Doe et al., 2023, machine learning is transforming healthcare."
    result_4, context_4 = IntegratedPipeline(agent).execute(test_input_4)
    
    print(f"\n用户输入: {test_input_4}")
    print(f"查询层级: {context_4.query_tier}")
    print(f"路由策略: {context_4.routing_strategy}")
    print(f"引用验证: {'触发' if context_4.has_citations else '未触发'}")
    if context_4.has_citations:
        print("提取的引用:")
        for cit in context_4.citations_extracted:
            print(f"  [{cit['source']}] {cit['content']} ({cit['confidence']:.2f})")
    
    # 打印流水线执行摘要
    print("\n" + "=" * 80)
    print("流水线执行摘要")
    print("=" * 80)
    
    test_cases = [
        ("简单查询", context_1),
        ("Persona匹配", context_2),
        ("幻觉检测", context_3),
        ("引用验证", context_4),
    ]
    
    for name, ctx in test_cases:
        print(f"\n{name}:")
        print(f"  查询层级: {ctx.query_tier}")
        print(f"  路由策略: {ctx.routing_strategy}")
        print(f"  Persona事实: {len(ctx.persona_facts)}条")
        print(f"  检索结果: {len(ctx.retrieval_results)}条")
        print(f"  幻觉检测: {'触发' if ctx.is_hallucination else '未触发'}")
        print(f"  引用验证: {'触发' if ctx.has_citations else '未触发'}")


if __name__ == "__main__":
    test_integrated_pipeline()
