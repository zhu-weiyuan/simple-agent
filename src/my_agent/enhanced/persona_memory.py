"""
Persona 记忆提取系统 - LLM 驱动版
基于论文：Synthius-Mem: Brain-Inspired Hallucination-Resistant Persona Memory
arXiv: 2604.11563v1
"""
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class CognitiveDomain(Enum):
    BIOGRAPHY = "biography"
    EXPERIENCES = "experiences"
    PREFERENCES = "preferences"
    SOCIAL_CIRCLE = "social_circle"
    WORK = "work"
    PSYCHOMETRICS = "psychometrics"


@dataclass
class PersonaFact:
    domain: CognitiveDomain
    fact: str
    confidence: float
    timestamp: str
    source: str


@dataclass
class PersonaMemory:
    """结构化 Persona 记忆存储"""
    facts: List[PersonaFact] = field(default_factory=list)
    domain_counts: Dict[CognitiveDomain, int] = field(default_factory=lambda: {
        domain: 0 for domain in CognitiveDomain
    })
    
    def add_fact(self, fact: PersonaFact) -> None:
        self.facts.append(fact)
        self.domain_counts[fact.domain] += 1
    
    def get_domain_facts(self, domain: CognitiveDomain) -> List[PersonaFact]:
        return [f for f in self.facts if f.domain == domain]
    
    def get_all_facts(self) -> List[PersonaFact]:
        return self.facts
    
    def get_summary(self) -> str:
        summary_parts = []
        for domain, count in self.domain_counts.items():
            if count > 0:
                summary_parts.append(f"- {domain.value}: {count} facts")
        return "\n".join(summary_parts) if summary_parts else "No facts stored"


class PersonaExtractor:
    """LLM 驱动的 Persona 记忆提取器 - 改进版，支持更好的规则和兜底"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and self._has_llm()
        self._client = None
    
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
    
    def extract_facts(self, text: str, source: str = "conversation") -> List[PersonaFact]:
        """Extract Persona facts from text (LLM driven with rule fallback)"""
        if self.use_llm:
            try:
                llm_facts = self._llm_extract(text, source)
                if llm_facts:
                    return llm_facts
            except Exception:
                pass
        
        # Fallback to rule-based extraction
        return self._rule_extract(text, source)
    
    def _llm_extract(self, text: str, source: str) -> List[PersonaFact]:
        """LLM driven extraction with better error handling"""
        client = self._get_client()
        
        prompt = f"""Extract user Persona information (personal profile facts) from the following text. Include:
- BIOGRAPHY: name, age, birthplace, gender, nationality, etc.
- EXPERIENCES: work experience, education, graduation schools, completed projects, etc.
- PREFERENCES: preferred tools, languages, styles, habits, etc.
- SOCIAL_CIRCLE: friends, family, colleagues, partners, etc.
- WORK: profession, position, company, organization, etc.
- PSYCHOMETRICS: personality, traits, behavior patterns, habits, etc.

Text:
---
{text}
---

Respond with a JSON array where each element is:
[
  {{
    "domain": "domain_name",
    "fact": "extracted fact (concise description)",
    "confidence": 0.0-1.0
  }}
]

If no Persona information can be extracted, return an empty array [].
Output only JSON, nothing else."""
        
        model = os.getenv("MY_AGENT_MODEL") or os.getenv("OPENAI_MODEL") or "default"
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional Persona information extractor. Extract user profile facts from conversation text. Output only JSON array, nothing else."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            
            content = response.choices[0].message.content or "[]"
            
            # Parse JSON
            facts = []
            try:
                # Try to extract JSON array from response
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(content)
            except json.JSONDecodeError:
                return facts
            
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            domain_map = {
                "biography": CognitiveDomain.BIOGRAPHY,
                "experiences": CognitiveDomain.EXPERIENCES,
                "preferences": CognitiveDomain.PREFERENCES,
                "social_circle": CognitiveDomain.SOCIAL_CIRCLE,
                "work": CognitiveDomain.WORK,
                "psychometrics": CognitiveDomain.PSYCHOMETRICS,
            }
            
            for item in data if isinstance(data, list) else []:
                domain_name = item.get("domain", "preferences")
                domain = domain_map.get(domain_name, CognitiveDomain.PREFERENCES)
                fact_text = item.get("fact", "")
                confidence = float(item.get("confidence", 0.8))
                
                if fact_text and len(fact_text) > 5:
                    facts.append(PersonaFact(
                        domain=domain,
                        fact=fact_text,
                        confidence=confidence,
                        timestamp=timestamp,
                        source=source,
                    ))
            
            return facts
            
        except Exception as e:
            raise Exception(f"LLM extraction failed: {str(e)}")
    
    def _rule_extract(self, text: str, source: str) -> List[PersonaFact]:
        """Improved rule-based extraction with better patterns"""
        facts = []
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Comprehensive rule patterns for different domains
        rules = [
            # Biography patterns
            (CognitiveDomain.BIOGRAPHY, [
                r'(?:我叫|我的名字是|I am|My name is)\s+([A-Za-z\u4e00-\u9fff]+)',
                r'(?:我来自|我住在|I live in|I am from)\s+([A-Za-z\u4e00-\u9fff]+)',
                r'(?:我出生于|I was born in)\s+(\d{4})',
            ]),
            # Work patterns
            (CognitiveDomain.WORK, [
                r'(?:我是|I work as|I am a)\s+([A-Za-z\u4e00-\u9fff]+(?:工程师|开发|设计师|经理|developer|engineer|designer|manager))',
                r'(?:我在|I work at)\s+([A-Za-z\u4e00-\u9fff&]+)',
            ]),
            # Preferences patterns
            (CognitiveDomain.PREFERENCES, [
                r'(?:我喜欢|I like|I prefer|偏爱)\s+([A-Za-z\u4e00-\u9fff]+)',
                r'(?:我喜欢用|I use)\s+([A-Za-z\u4e00-\u9fff]+)',
            ]),
            # Social Circle patterns
            (CognitiveDomain.SOCIAL_CIRCLE, [
                r'(?:我有一个朋友|I have a friend)\s+([A-Za-z\u4e00-\u9fff]+)',
                r'(?:我的朋友|My friend)\s+([A-Za-z\u4e00-\u9fff]+)',
            ]),
            # Psychometrics patterns
            (CognitiveDomain.PSYCHOMETRICS, [
                r'(?:我性格|My personality|I am)\s+([A-Za-z\u4e00-\u9fff]+(?:开朗|外向|内向|安静|outgoing|introvert|quiet))',
                r'(?:我是个|I am a)\s+([A-Za-z\u4e00-\u9fff]+(?:人|person))',
            ]),
        ]
        
        for domain, patterns in rules:
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.I)
                for match in matches:
                    fact_text = match.group(0).strip()
                    if len(fact_text) > 5:
                        facts.append(PersonaFact(
                            domain=domain,
                            fact=fact_text,
                            confidence=0.6,
                            timestamp=timestamp,
                            source=source,
                        ))
        
        return facts


class CategoryRAG:
    """CategoryRAG 检索系统 - 改进版，支持更好的检索逻辑"""
    
    def __init__(self, persona_memory: PersonaMemory):
        self.persona_memory = persona_memory
    
    def retrieve(self, query: str, domain: CognitiveDomain, top_k: int = 5) -> List[PersonaFact]:
        """Retrieve facts from specific domain with query relevance"""
        domain_facts = self.persona_memory.get_domain_facts(domain)
        
        # Simple keyword matching for relevance scoring
        query_lower = query.lower()
        scored_facts = []
        
        for fact in domain_facts:
            score = self._calculate_relevance_score(query_lower, fact.fact.lower())
            scored_facts.append((fact, score))
        
        # Sort by relevance score and confidence
        scored_facts.sort(key=lambda x: (x[1] * 0.7 + x[0].confidence * 0.3), reverse=True)
        
        return [fact for fact, score in scored_facts[:top_k]]
    
    def retrieve_all(self, query: str, top_k: int = 10) -> List[PersonaFact]:
        """Retrieve facts from all domains with query relevance"""
        all_facts = self.persona_memory.get_all_facts()
        
        # Score all facts by relevance
        query_lower = query.lower()
        scored_facts = []
        
        for fact in all_facts:
            score = self._calculate_relevance_score(query_lower, fact.fact.lower())
            scored_facts.append((fact, score))
        
        # Sort by combined relevance and confidence
        scored_facts.sort(key=lambda x: (x[1] * 0.7 + x[0].confidence * 0.3), reverse=True)
        
        return [fact for fact, score in scored_facts[:top_k]]
    
    def _calculate_relevance_score(self, query: str, fact: str) -> float:
        """Calculate relevance score between query and fact"""
        # Simple keyword overlap scoring
        query_words = set(query.split())
        fact_words = set(fact.split())
        
        if not query_words or not fact_words:
            return 0.1
        
        overlap = len(query_words & fact_words)
        total = len(query_words | fact_words)
        
        return overlap / total if total > 0 else 0.1
