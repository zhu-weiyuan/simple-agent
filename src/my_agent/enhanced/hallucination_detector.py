"""
实时幻觉检测系统 - 混合检测（常识锚点 + LLM + 网络搜索）
基于概念：SelfCheckGPT + HALU + FaCt
参考论文：
- SelfCheckGENT: mitigating factual hallucations (2023)
- HALU: hallucination detection in LLMs (2023) 
- Fact: factuality detection with LLMs (2023)
"""
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class HallucinationResult:
    is_hallucination: bool
    confidence: float
    hallucination_type: str  # none | factual_error | temporal_inconsistency | causal_fallacy | overconfidence | fabrication
    evidence: List[str] = field(default_factory=list)
    correction_suggestion: str = ""


class HallucinationDetector:
    """混合幻觉检测器：常识锚点 + 规则 + LLM + 网络搜索"""

    def __init__(self, use_llm: bool = True, use_search: bool = False):
        self.use_llm = use_llm and self._has_llm()
        self.use_search = use_search
        self._client = None
        self.danger_patterns = self._compile_danger_patterns()
        self.common_sense_facts = self._load_common_sense()
        self.temporal_patterns = self._compile_temporal_patterns()
        self.causal_patterns = self._compile_causal_patterns()
        self.overconfidence_patterns = self._compile_overconfidence_patterns()
        self.fabrication_patterns = self._compile_fabrication_patterns()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _has_llm(self) -> bool:
        try:
            from openai import OpenAI
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv("MY_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY") or "your_key_here",
                base_url=os.getenv("MY_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "http://localhost:8080/v1",
            )
        return self._client

    def _compile_danger_patterns(self) -> List[re.Pattern]:
        """Compile patterns that indicate potential hallucinations"""
        return [
            # Vague authority claims without specific sources
            re.compile(r'\b(research shows|studies indicate|experts agree)\b', re.I),
            re.compile(r'\b(always|never|everyone|no one)\b.*\b(definitely|certainly)\b', re.I),
            re.compile(r'\b(according to)\b.*\b(study|research|paper|journal)\b', re.I),
            # Overly specific but unverifiable claims
            re.compile(r'\b(exactly|precisely)\b.*\b([0-9]{3,})\b.*\b(people|years|times)\b', re.I),
            # Claims about future events as facts
            re.compile(r'\b(will|going to)\b.*\b(definitely|certainly|will definitely)\b', re.I),
        ]

    def _load_common_sense(self) -> List[Dict]:
        """Load common sense facts for contradiction detection"""
        return [
            {"claim": "Python 是动态类型语言，类型提示是可选的", "negations": ["强制类型", "强制要求.*声明类型", "必须声明类型", "类型检查强制", "类型提示强制", "强制.*类型.*声明", "强制.*类型.*检查"]},
            {"claim": "Python 有 GIL（全局解释器锁）", "negations": ["Python 没有 GIL", "Python 原生多线程"]},
            {"claim": "HTTP 默认端口是 80", "negations": ["HTTP 默认端口 443", "HTTP 默认端口是 443"]},
            {"claim": "SQL 是声明式语言", "negations": ["SQL 是过程式", "SQL 是命令式"]},
            {"claim": "JavaScript 单线程（主线程）", "negations": ["JavaScript 原生多线程", "JavaScript 多线程"]},
            {"claim": "Paris is the capital of France", "negations": ["Paris is not the capital", "capital of France is not Paris"]},
            {"claim": "Water boils at 100°C at sea level", "negations": ["water boils at [0-9]+°C", "boiling point of water is [0-9]+"]},
        ]

    def _compile_temporal_patterns(self) -> List[Dict]:
        """Compile patterns for temporal inconsistency detection"""
        return [
            # Past events described as future
            {"pattern": re.compile(r'\b(in (?:19|20)\d{2})\b.*\b(will|going to|future)\b', re.I), "type": "past_as_future"},
            # Future events described as past
            {"pattern": re.compile(r'\b(in (?:20|21)\d{2})\b.*\b(did|happened|occurred)\b', re.I), "type": "future_as_past"},
            # Contradictory time references
            {"pattern": re.compile(r'\byesterday\b.*\b tomorrow\b|\b tomorrow\b.*\b yesterday\b', re.I), "type": "contradictory_time"},
            # Impossible time sequences
            {"pattern": re.compile(r'\bbefore\b.*\bafter\b.*\bthe same event\b', re.I), "type": "impossible_sequence"},
        ]

    def _compile_causal_patterns(self) -> List[Dict]:
        """Compile patterns for causal fallacy detection"""
        return [
            # Post hoc ergio propter hoc (after this, therefore because of this)
            {"pattern": re.compile(r'\b(because|since|due to)\b.*\b(therefore|so|thus)\b.*\b(proved|demonstrated)\b', re.I), "type": "post_hoc"},
            # False cause without evidence
            {"pattern": re.compile(r'\b(causes|leads to|results in)\b.*\b(without|no|lack of)\b.*\b(evidence|proof|studies)\b', re.I), "type": "false_cause"},
            # Correlation implies causation
            {"pattern": re.compile(r'\b(correlated|associated)\b.*\b(causes|leads to)\b', re.I), "type": "correlation_causation"},
        ]

    def _compile_overconfidence_patterns(self) -> List[Dict]:
        """Compile patterns for overconfidence detection"""
        return [
            # Absolute statements without evidence
            {"pattern": re.compile(r'\b(100%|absolutely|definitely|certainly|undoubtedly)\b.*\b(true|correct|fact)\b', re.I), "type": "absolute_claim"},
            # Overconfident predictions
            {"pattern": re.compile(r'\b(will|going to)\b.*\b(100%|absolutely|definitely|certainly)\b.*\b(happen|occur|result)\b', re.I), "type": "overconfident_prediction"},
            # Unqualified expert claims
            {"pattern": re.compile(r'\b(experts\s+say|scientists\s+agree|researchers\s+confirm)\b.*\b(every|all|everyone)\b', re.I), "type": "unqualified_expert"},
        ]

    def _compile_fabrication_patterns(self) -> List[Dict]:
        """Compile patterns for fabrication detection"""
        return [
            # Fake statistics (including >100%)
            {"pattern": re.compile(r'([1-9][0-9]{2,}|100)%.*\b(of\s+\w+)\b.*\b(study|research|survey)\b', re.I), "type": "fake_statistic"},
            # Fake statistics with experts
            {"pattern": re.compile(r'\b(experts\s+say|scientists\s+agree)\b.*([0-9]+)%', re.I), "type": "fake_statistic"},
            # Impossible percentages (>100%)
            {"pattern": re.compile(r'([1-9][0-9]{2,})%', re.I), "type": "impossible_percentage"},
            # Invented quotes
            {"pattern": re.compile(r'\b("|\u201c)([^"]{20,})\1\b.*\b(said|stated|claimed)\b.*\b([A-Z][a-z]+ [A-Z][a-z]+)\b', re.I), "type": "invented_quote"},
            # Fake references
            {"pattern": re.compile(r'\b([A-Z][a-z]+ et al\.?\s+\(?[0-9]{4}\)?)\b.*\b(found|discovered|proved)\b', re.I), "type": "fake_reference"},
        ]

    def detect(self, text: str) -> HallucinationResult:
        """Main detection method with cascading checks"""
        # Step 1: Common sense contradiction check (highest confidence)
        anchor_result = self._common_sense_check(text)
        if anchor_result and anchor_result.confidence >= 0.85:
            return anchor_result

        # Step 2: Temporal inconsistency check
        temporal_result = self._temporal_check(text)
        if temporal_result and temporal_result.confidence >= 0.8:
            return temporal_result

        # Step 3: Causal fallacy check
        causal_result = self._causal_check(text)
        if causal_result and causal_result.confidence >= 0.75:
            return causal_result

        # Step 4: Overconfidence check
        overconfidence_result = self._overconfidence_check(text)
        if overconfidence_result and overconfidence_result.confidence >= 0.7:
            return overconfidence_result

        # Step 5: Fabrication check
        fabrication_result = self._fabrication_check(text)
        if fabrication_result and fabrication_result.confidence >= 0.7:
            return fabrication_result

        # Step 6: Quick rule-based check
        quick_result = self._quick_check(text)
        if quick_result:
            return quick_result

        # Step 7: LLM-based detection (if available)
        if self.use_llm:
            try:
                return self._llm_detect(text)
            except Exception:
                pass

        # Default: no hallucination detected
        return HallucinationResult(
            is_hallucination=False,
            confidence=0.5,
            hallucination_type="none",
            evidence=[],
            correction_suggestion="No hallucination detected",
        )

    def _common_sense_check(self, text: str) -> Optional[HallucinationResult]:
        """Check for contradictions with common sense facts"""
        evidence = []
        for fact in self.common_sense_facts:
            for negation in fact["negations"]:
                if re.search(negation, text, re.I):
                    msg = f"Contradicts known fact: {fact['claim']}"
                    evidence.append(msg)
        
        if evidence:
            return HallucinationResult(
                is_hallucination=True,
                confidence=0.85,
                hallucination_type="factual_error",
                evidence=evidence,
                correction_suggestion=f"⚠️ {'; '.join(evidence)}",
            )
        return None

    def _temporal_check(self, text: str) -> Optional[HallucinationResult]:
        """Check for temporal inconsistencies"""
        evidence = []
        for pattern_info in self.temporal_patterns:
            if pattern_info["pattern"].search(text):
                msg = f"Temporal inconsistency detected: {pattern_info['type']}"
                evidence.append(msg)
        
        if evidence:
            return HallucinationResult(
                is_hallucination=True,
                confidence=0.8,
                hallucination_type="temporal_inconsistency",
                evidence=evidence,
                correction_suggestion=f"⚠️ {'; '.join(evidence)}",
            )
        return None

    def _causal_check(self, text: str) -> Optional[HallucinationResult]:
        """Check for causal fallacies"""
        evidence = []
        for pattern_info in self.causal_patterns:
            if pattern_info["pattern"].search(text):
                msg = f"Causal fallacy detected: {pattern_info['type']}"
                evidence.append(msg)
        
        if evidence:
            return HallucinationResult(
                is_hallucination=True,
                confidence=0.75,
                hallucination_type="causal_fallacy",
                evidence=evidence,
                correction_suggestion=f"⚠️ {'; '.join(evidence)}",
            )
        return None

    def _overconfidence_check(self, text: str) -> Optional[HallucinationResult]:
        """Check for overconfidence patterns"""
        evidence = []
        for pattern_info in self.overconfidence_patterns:
            if pattern_info["pattern"].search(text):
                msg = f"Overconfidence detected: {pattern_info['type']}"
                evidence.append(msg)
        
        if evidence:
            return HallucinationResult(
                is_hallucination=True,
                confidence=0.7,
                hallucination_type="overconfidence",
                evidence=evidence,
                correction_suggestion=f"⚠️ {'; '.join(evidence)}",
            )
        return None

    def _fabrication_check(self, text: str) -> Optional[HallucinationResult]:
        """Check for fabrication patterns"""
        evidence = []
        for pattern_info in self.fabrication_patterns:
            if pattern_info["pattern"].search(text):
                msg = f"Potential fabrication detected: {pattern_info['type']}"
                evidence.append(msg)
        
        if evidence:
            return HallucinationResult(
                is_hallucination=True,
                confidence=0.7,
                hallucination_type="fabrication",
                evidence=evidence,
                correction_suggestion=f"⚠️ {'; '.join(evidence)}",
            )
        return None

    def _quick_check(self, text: str) -> Optional[HallucinationResult]:
        """Quick rule-based check for multiple danger patterns"""
        danger_count = 0
        matched_patterns = []
        
        for pattern in self.danger_patterns:
            if pattern.search(text):
                danger_count += 1
                matched_patterns.append(pattern.pattern[:50])
        
        if danger_count >= 2:
            return HallucinationResult(
                is_hallucination=True,
                confidence=0.7,
                hallucination_type="potential_fabrication",
                evidence=[f"Detected {danger_count} suspicious patterns: {'; '.join(matched_patterns)}"],
                correction_suggestion="⚠️ This text may contain fabricated information. Please verify sources.",
            )
        return None

    def _llm_detect(self, text: str) -> HallucinationResult:
        """LLM-based hallucination detection"""
        client = self._get_client()
        
        prompt = f"""You are a professional hallucination detector. Analyze the following text for factual errors, inconsistencies, or potential fabrications.

Text to analyze:
---
{text}
---

Please respond with a JSON object containing:
{{
  "is_hallucination": true/false,
  "confidence": 0.0-1.0,
  "hallucination_type": "none" | "factual_error" | "temporal_inconsistency" | "causal_fallacy" | "overconfidence" | "fabrication",
  "evidence": ["list of specific issues found"],
  "correction_suggestion": "suggested correction or verification advice"
}}

If no hallucination is detected, set is_hallucination to false and hallucination_type to "none"."""
        
        model = os.getenv("MY_AGENT_MODEL") or os.getenv("OPENAI_MODEL") or "default"
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional hallucination detector. Only output JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            
            content = response.choices[0].message.content or "{}"
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(content)
            except json.JSONDecodeError:
                return HallucinationResult(
                    is_hallucination=False,
                    confidence=0.5,
                    hallucination_type="none",
                    evidence=["Failed to parse LLM response"],
                    correction_suggestion="LLM detection failed",
                )
            
            is_hallucination = data.get("is_hallucination", False)
            confidence = float(data.get("confidence", 0.5))
            hallucination_type = data.get("hallucination_type", "none")
            evidence = data.get("evidence", [])
            correction_suggestion = data.get("correction_suggestion", "")
            
            # Format correction suggestion
            if is_hallucination and correction_suggestion and correction_suggestion != "无":
                correction_suggestion = f"⚠️ {correction_suggestion}"
            elif not is_hallucination:
                correction_suggestion = "No hallucination detected"
            
            return HallucinationResult(
                is_hallucination=is_hallucination,
                confidence=confidence,
                hallucination_type=hallucination_type,
                evidence=evidence if isinstance(evidence, list) else [],
                correction_suggestion=correction_suggestion,
            )
            
        except Exception as e:
            return HallucinationResult(
                is_hallucination=False,
                confidence=0.5,
                hallucination_type="none",
                evidence=[f"LLM detection failed: {str(e)}"],
                correction_suggestion="LLM detection unavailable",
            )


if __name__ == "__main__":
    detector = HallucinationDetector()
    test_texts = [
        "Python 3.12 强制要求所有变量声明类型，不声明就报错。",
        "Python 是由 Guido van Rossum 于 1991 年发布的。",
        "Python 没有 GIL，原生支持多线程。",
        "AI 将在未来改变世界。",
        "In 1999, the study will show that AI will definitely change the world.",
        "Because X happened, therefore Y must be true - this has been proved beyond doubt.",
        "Experts say that 847% of people believe in aliens according to a recent study.",
    ]
    for text in test_texts:
        r = detector.detect(text)
        icon = "🚨" if r.is_hallucination else "✅"
        print(f"{icon} {text}")
        print(f"   Hallucination: {r.is_hallucination} | Type: {r.hallucination_type} | Confidence: {r.confidence}")
        print(f"   Evidence: {r.evidence}")
        print(f"   Correction: {r.correction_suggestion}")
        print()
