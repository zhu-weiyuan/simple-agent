# -*- coding: utf-8 -*-
"""
查询复杂度分类器 + 动态路由
基于论文:Adaptive Query Routing: A Tier-Based Framework for Hybrid Retrieval
arXiv: 2604.14222v1
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class QueryTier(Enum):
    TIER_1_SIMPLE = "tier_1_simple"
    TIER_2_MULTI_FACT = "tier_2_multi_fact"
    TIER_3_CROSS_REF = "tier_3_cross_reference"
    TIER_4_SYNTHESIS = "tier_4_synthesis"


class RetrievalStrategy(Enum):
    VECTOR_RAG = "vector_rag"
    TREE_REASONING = "tree_reasoning"
    HYBRID_AHR = "hybrid_ahr"
    HYBRID_TREE_ENSEMBLE = "hybrid_tree_ensemble"


@dataclass
class QueryAnalysis:
    tier: QueryTier
    strategy: RetrievalStrategy
    confidence: float
    complexity_score: float
    indicators: List[str]


class QueryComplexityClassifier:
    """四级查询复杂度分类器 - 改进版，支持更准确的分级"""
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, Dict]:
        """Compile patterns for each query tier with better differentiation"""
        return {
            QueryTier.TIER_1_SIMPLE: {
                "patterns": [
                    # Simple factual questions
                    re.compile(r'\b(what|who|when|where|how)\b.*\b(is|are|was|were)\b', re.I),
                    re.compile(r'\b(defin|mean|explain|describe)\b.*\b(a|an|the)\b', re.I),
                    re.compile(r'\b(list|summarize|outline)\b.*\b(single|one|first)\b', re.I),
                    # Simple yes/no questions
                    re.compile(r'\b(is|are|was|were|do|does|did)\b.*\b(a|an|the)\b', re.I),
                ],
                "negative_patterns": [
                    # Exclude complex multi-part questions
                    re.compile(r'\b(and|,|;)\b.*\b(what|who|when|where|how)\b', re.I),
                    re.compile(r'\b(compare|contrast|analyz|evaluat|assess)\b', re.I),
                ],
                "weight": 1.0,
            },
            QueryTier.TIER_2_MULTI_FACT: {
                "patterns": [
                    # Multi-part questions
                    re.compile(r'\b(list|enumerate|identify)\b.*\b(three|four|five|multiple|several)\b', re.I),
                    re.compile(r'\b(what|who|when|where|how)\b.*\b(and|,|;)\b.*\b(what|who|when|where|how)\b', re.I),
                    # Questions requiring multiple facts
                    re.compile(r'\b(explain|describe|detail)\b.*\b(each|every|all)\b', re.I),
                    re.compile(r'\b(components|parts|elements|factors|aspects)\b', re.I),
                ],
                "negative_patterns": [
                    # Exclude simple single-fact questions
                    re.compile(r'\b(simple|basic|single|one)\b', re.I),
                    re.compile(r'\b(what is|who is|when is|where is)\b.*\b(a|an)\b', re.I),
                ],
                "weight": 1.2,
            },
            QueryTier.TIER_3_CROSS_REF: {
                "patterns": [
                    # Comparison and contrast questions
                    re.compile(r'\b(compare|contrast|versus|vs\.?)\b', re.I),
                    re.compile(r'\b(similar|different|difference|similarity)\b', re.I),
                    # Cross-referencing questions
                    re.compile(r'\b(between|across|among)\b.*\b(two|multiple|various|different)\b', re.I),
                    re.compile(r'\b(relate|connect|link|associate)\b', re.I),
                ],
                "negative_patterns": [
                    # Exclude simple comparisons
                    re.compile(r'\b(simple|basic|single)\b', re.I),
                    re.compile(r'\b(what is|who is)\b', re.I),
                ],
                "weight": 1.4,
            },
            QueryTier.TIER_4_SYNTHESIS: {
                "patterns": [
                    # Complex synthesis questions
                    re.compile(r'\b(based on|drawing from|combining|merging)\b', re.I),
                    re.compile(r'\b(implication|consequence|impact|effect)\b.*\b(analysi|review|summary)\b', re.I),
                    re.compile(r'\b(comprehensive|thorough|extensive)\b.*\b(analysi|review|summary)\b', re.I),
                    re.compile(r'\b(synthesi|integrat)\b.*\b(comprehens|thorough|extens)\b', re.I),
                    # Multi-domain analysis
                    re.compile(r'\b(cross-domain|interdisciplinary|multidisciplinary)\b', re.I),
                ],
                "negative_patterns": [],
                "weight": 1.5,
            },
        }
    
    def classify(self, query: str) -> QueryAnalysis:
        """Classify query complexity with improved scoring"""
        scores = {}
        indicators = []
        
        for tier, compiled in self.compiled_patterns.items():
            tier_score = 0.0
            tier_indicators = []
            
            # Check positive patterns
            for pattern in compiled["patterns"]:
                if pattern.search(query):
                    tier_score += 1.0
                    tier_indicators.append(f"pos:{pattern.pattern[:50]}")
            
            # Check negative patterns (reduce score)
            for pattern in compiled["negative_patterns"]:
                if pattern.search(query):
                    tier_score -= 0.5
                    tier_indicators.append(f"neg:{pattern.pattern[:50]}")
            
            # Apply weight
            tier_score *= compiled["weight"]
            scores[tier] = tier_score
            indicators.extend(tier_indicators)
        
        # Determine best tier
        best_tier = max(scores, key=scores.get)
        best_score = scores[best_tier]
        
        # Calculate confidence based on score margin
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            margin = sorted_scores[0] - sorted_scores[1]
            confidence = min(0.5 + margin * 0.3, 0.95)
        else:
            confidence = 0.5
        
        # Determine retrieval strategy
        strategy = self._get_strategy(best_tier)
        
        # Normalize complexity score (0-1 range)
        tier_index = list(QueryTier).index(best_tier)
        complexity_score = tier_index / 3.0
        
        return QueryAnalysis(
            tier=best_tier,
            strategy=strategy,
            confidence=confidence,
            complexity_score=complexity_score,
            indicators=indicators[:5],
        )
    
    def _get_strategy(self, tier: QueryTier) -> RetrievalStrategy:
        """Map tier to appropriate retrieval strategy"""
        strategy_map = {
            QueryTier.TIER_1_SIMPLE: RetrievalStrategy.VECTOR_RAG,
            QueryTier.TIER_2_MULTI_FACT: RetrievalStrategy.TREE_REASONING,
            QueryTier.TIER_3_CROSS_REF: RetrievalStrategy.HYBRID_AHR,
            QueryTier.TIER_4_SYNTHESIS: RetrievalStrategy.HYBRID_TREE_ENSEMBLE,
        }
        return strategy_map.get(tier, RetrievalStrategy.VECTOR_RAG)


class DynamicRouter:
    """动态路由系统"""
    
    def __init__(self):
        self.classifier = QueryComplexityClassifier()
    
    def route_query(self, query: str) -> QueryAnalysis:
        """Route query to appropriate retrieval strategy"""
        analysis = self.classifier.classify(query)
        return analysis


# Test code
if __name__ == "__main__":
    router = DynamicRouter()
    
    test_queries = [
        "What is the capital of France?",
        "List the three main components of a car engine and explain each one",
        "Compare the economic policies of the US and China",
        "Based on the research papers from 2020-2025, synthesize a comprehensive analysis of AI's impact on healthcare, education, and finance",
    ]
    
    print("=" * 80)
    print("Query Complexity Classifier Test")
    print("=" * 80)
    
    for query in test_queries:
        analysis = router.route_query(query)
        print(f"\nQuery: {query}")
        print(f"  Tier: {analysis.tier.value}")
        print(f"  Strategy: {analysis.strategy.value}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Complexity Score: {analysis.complexity_score:.2f}")
        print(f"  Indicators: {analysis.indicators}")
        print("-" * 80)
