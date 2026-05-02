# -*- coding: utf-8 -*-
"""
确定性引用机制
基于论文：AgriIR: A Scalable Framework for Domain-Specific Knowledge Retrieval
arXiv: 2604.16353v1
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Citation:
    source: str
    content: str
    confidence: float
    timestamp: str = "2026-04-27"
    citation_type: str = "unknown"


@dataclass
class CitationResult:
    has_citation: bool
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    verification_status: str = "not_verified"


class DeterministicCitation:
    """确定性引用系统 - 支持直接引用、参考文献引用和事实声明验证"""
    
    def __init__(self):
        self.citation_patterns = self._compile_citation_patterns()
        self.citations: List[Citation] = []
    
    def _compile_citation_patterns(self) -> Dict[str, List[Dict]]:
        """Compile comprehensive citation patterns"""
        return {
            "direct_quote": [
                # Direct quotes with quotation marks
                {"pattern": re.compile(r'["\u201c\u201d](.+?)["\u201c\u201d]', re.I), "extract": "content"},
                # Quotes with attribution
                {"pattern": re.compile(r'(?:said|stated|mentioned|quoted|wrote)\s+["\u201c\u201d](.+?)["\u201c\u201d]', re.I), "extract": "content"},
                # Chinese quotes
                {"pattern": re.compile(r'[''\u2018\u2019](.+?)[''\u2018\u2019]', re.I), "extract": "content"},
            ],
            "reference": [
                # Academic references
                {"pattern": re.compile(r'(?:according to|as stated by|based on|as per)\s+([^\.,;]+)', re.I), "extract": "source"},
                # Citation formats
                {"pattern": re.compile(r'\(([A-Z][a-z]+ et al\.?\s*,?\s*\d{4})\)', re.I), "extract": "citation"},
                # Journal/paper references
                {"pattern": re.compile(r'(?:in|from)\s+([A-Z][a-zA-Z\s&]+(?:Journal|Magazine|News|Review|Study))', re.I), "extract": "source"},
            ],
            "fact_claim": [
                # Fact claims without sources
                {"pattern": re.compile(r'(?:it is a fact that|the fact is|clearly|obviously)\s+(.+?)[\.\!]', re.I), "extract": "claim"},
                # Authoritative claims
                {"pattern": re.compile(r'(?:experts say|researchers confirm|scientists agree)\s+that\s+(.+?)[\.\!]', re.I), "extract": "claim"},
            ],
        }
    
    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the collection"""
        self.citations.append(citation)
    
    def extract_citations(self, text: str) -> CitationResult:
        """Extract citations from text with comprehensive pattern matching"""
        citations = []
        
        # Extract direct quotes
        quotes = self._extract_direct_quotes(text)
        citations.extend(quotes)
        
        # Extract references
        references = self._extract_references(text)
        citations.extend(references)
        
        # Extract fact claims
        claims = self._extract_fact_claims(text)
        citations.extend(claims)
        
        # Calculate confidence based on citation types and counts
        has_citation = len(citations) > 0
        confidence = self._calculate_confidence(citations) if has_citation else 0.0
        
        # Determine verification status
        verification_status = self._determine_verification_status(citations)
        
        return CitationResult(
            has_citation=has_citation,
            citations=citations,
            confidence=confidence,
            verification_status=verification_status,
        )
    
    def _extract_direct_quotes(self, text: str) -> List[Citation]:
        """Extract direct quotes from text"""
        citations = []
        
        for pattern_info in self.citation_patterns["direct_quote"]:
            pattern = pattern_info["pattern"]
            matches = pattern.finditer(text)
            
            for match in matches:
                content = match.group(1).strip()
                if len(content) > 5:  # Ignore very short quotes
                    citation = Citation(
                        source="direct_quote",
                        content=content,
                        confidence=0.9,
                        citation_type="quote",
                    )
                    citations.append(citation)
        
        return citations
    
    def _extract_references(self, text: str) -> List[Citation]:
        """Extract references and sources from text"""
        citations = []
        
        for pattern_info in self.citation_patterns["reference"]:
            pattern = pattern_info["pattern"]
            matches = pattern.finditer(text)
            
            for match in matches:
                source_text = match.group(1).strip()
                if len(source_text) > 3:  # Ignore very short sources
                    citation = Citation(
                        source=source_text,
                        content=f"Reference to {source_text}",
                        confidence=0.8,
                        citation_type="reference",
                    )
                    citations.append(citation)
        
        return citations
    
    def _extract_fact_claims(self, text: str) -> List[Citation]:
        """Extract fact claims that need verification"""
        citations = []
        
        for pattern_info in self.citation_patterns["fact_claim"]:
            pattern = pattern_info["pattern"]
            matches = pattern.finditer(text)
            
            for match in matches:
                claim_text = match.group(1).strip()
                if len(claim_text) > 10:  # Ignore very short claims
                    citation = Citation(
                        source="fact_claim",
                        content=claim_text,
                        confidence=0.6,
                        citation_type="claim",
                    )
                    citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, citations: List[Citation]) -> float:
        """Calculate overall confidence based on citation types and quality"""
        if not citations:
            return 0.0
        
        # Higher confidence for direct quotes and academic references
        type_weights = {
            "quote": 0.9,
            "reference": 0.8,
            "claim": 0.6,
            "unknown": 0.5,
        }
        
        total_confidence = sum(type_weights.get(c.citation_type, 0.5) for c in citations)
        avg_confidence = total_confidence / len(citations)
        
        # Boost confidence for multiple citations
        if len(citations) > 1:
            avg_confidence = min(avg_confidence + 0.1, 0.95)
        
        return avg_confidence
    
    def _determine_verification_status(self, citations: List[Citation]) -> str:
        """Determine verification status based on citation types"""
        if not citations:
            return "not_verified"
        
        # Check if we have verifiable sources
        has_academic_ref = any(c.citation_type == "reference" for c in citations)
        has_direct_quote = any(c.citation_type == "quote" for c in citations)
        
        if has_academic_ref or has_direct_quote:
            return "verified"
        elif len(citations) > 0:
            return "partially_verified"
        else:
            return "not_verified"
    
    def get_citation_summary(self) -> str:
        """Get a summary of all citations"""
        if not self.citations:
            return "No citations found"
        
        summary_parts = []
        for i, citation in enumerate(self.citations, 1):
            summary_parts.append(f"{i}. [{citation.citation_type}] {citation.content}")
        
        return "\n".join(summary_parts)
    
    def verify_citation(self, citation: Citation) -> bool:
        """Verify a single citation"""
        # Basic validation checks
        if not citation.content or len(citation.content) < 5:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            re.compile(r'\b(fake|fabrication|hallucination)\b', re.I),
            re.compile(r'\b(unknown|anonymous|no source)\b', re.I),
        ]
        
        for pattern in suspicious_patterns:
            if pattern.search(citation.content):
                return False
        
        return True
    
    def verify_all_citations(self) -> bool:
        """Verify all citations"""
        return all(self.verify_citation(citation) for citation in self.citations)


# Test code
if __name__ == "__main__":
    citation_system = DeterministicCitation()
    
    test_texts = [
        '"The capital of France is Paris" is a well-known fact.',
        'According to a recent study, AI will change the world.',
        'The capital of France is Paris.',
        'As stated by John Doe et al., 2023, machine learning is transforming healthcare.',
        'Experts say that artificial intelligence will revolutionize every industry.',
    ]
    
    print("=" * 80)
    print("Deterministic Citation Test")
    print("=" * 80)
    
    for text in test_texts:
        result = citation_system.extract_citations(text)
        print(f"\nText: {text}")
        print(f"  Has Citation: {result.has_citation}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Verification Status: {result.verification_status}")
        if result.citations:
            for citation in result.citations:
                print(f"  Citation: [{citation.citation_type}] {citation.content}")
        print("-" * 80)
