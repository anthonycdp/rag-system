"""Hallucination detection for RAG systems."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.constants import (
    DEFAULT_HALLUCINATION_THRESHOLD,
    DEFAULT_LLM_MODEL,
    MAX_CLAIMS_SHOWN,
    MAX_CONTEXT_RATIO,
    MAX_DISPLAY_CLAIMS,
    MAX_RESPONSE_LENGTH,
    MIN_CLAIM_LENGTH,
)
from src.utils import create_llm, extract_json_from_response, format_context_text


class HallucinationLevel(str, Enum):
    """Severity levels for detected hallucinations."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClaimAnalysis:
    """Analysis of a single claim."""

    claim: str
    is_grounded: bool
    supporting_evidence: str | None = None
    confidence: float = 0.0


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    is_hallucination: bool
    level: HallucinationLevel
    overall_score: float
    claims: list[ClaimAnalysis] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw_response: str = ""


class HallucinationConfig(BaseModel):
    """Configuration for hallucination detection."""

    model_name: str = DEFAULT_LLM_MODEL
    api_key: str | None = None
    threshold: float = Field(default=DEFAULT_HALLUCINATION_THRESHOLD, ge=0.0, le=1.0)
    enable_suggestions: bool = True


class HallucinationDetector:
    """Detects hallucinations in generated responses."""

    CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer.
Return as a JSON list of claim strings.

Answer: {answer}

Return format: ["claim1", "claim2", ...]"""

    VERIFICATION_PROMPT = """You are a factual verification system. Your task is to verify if claims are grounded in the provided context.

Context (Retrieved Documents):
{context}

Claims to Verify:
{claims}

For each claim:
1. Check if it is directly supported by the context
2. Check if it can be reasonably inferred from the context
3. A claim is UNFOUNDED if it contains information NOT in the context

Return your analysis as JSON:
{{
    "overall_groundedness_score": <float 0-1>,
    "claims": [
        {{
            "claim": "<claim text>",
            "is_grounded": <true/false>,
            "supporting_evidence": "<quote from context if grounded, null if not>",
            "confidence": <float 0-1>
        }},
        ...
    ],
    "hallucination_level": "<none/low/medium/high/critical>",
    "suggestions": ["<suggestion1>", ...]
}}

Scoring guide:
- 1.0: All claims fully grounded
- 0.7-0.9: Most claims grounded, minor unsupported details
- 0.4-0.6: Mixed grounded and ungrounded claims
- 0.1-0.3: Mostly ungrounded claims
- 0.0: Complete hallucination

Level guide:
- none: No hallucination detected (score >= 0.9)
- low: Minor details unsupported (score 0.7-0.9)
- medium: Some claims unsupported (score 0.4-0.7)
- high: Many claims unsupported (score 0.1-0.4)
- critical: Response is largely fabricated (score < 0.1)"""

    def __init__(self, config: HallucinationConfig | None = None) -> None:
        self.config = config or HallucinationConfig()
        self._llm: BaseChatModel | None = None

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = create_llm(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
            )
        return self._llm

    def _extract_claims_from_answer(self, answer: str) -> list[str]:
        """Extract factual claims from an answer using LLM.

        Args:
            answer: Generated answer.

        Returns:
            List of claim strings.
        """
        llm = self._get_llm()
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        response = llm.invoke(prompt)

        try:
            claims = extract_json_from_response(response.content)
            return claims if isinstance(claims, list) else []
        except ValueError:
            return self._split_into_sentences(answer)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences as fallback claim extraction.

        Args:
            text: Text to split.

        Returns:
            List of sentences that meet minimum length requirement.
        """
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > MIN_CLAIM_LENGTH]

    def _format_claims_for_verification(self, claims: list[str]) -> str:
        """Format claims for the verification prompt.

        Args:
            claims: List of claims to format.

        Returns:
            Formatted claims string.
        """
        return "\n".join(f"{i+1}. {claim}" for i, claim in enumerate(claims))

    def _parse_claim_analyses(self, claims_data: list[dict]) -> tuple[list[ClaimAnalysis], list[str]]:
        """Parse claim analyses from verification response.

        Args:
            claims_data: List of claim dictionaries from LLM response.

        Returns:
            Tuple of (claim_analyses, ungrounded_claims).
        """
        analyses = []
        ungrounded = []

        for claim_data in claims_data:
            analysis = ClaimAnalysis(
                claim=claim_data.get("claim", ""),
                is_grounded=claim_data.get("is_grounded", False),
                supporting_evidence=claim_data.get("supporting_evidence"),
                confidence=claim_data.get("confidence", 0.0),
            )
            analyses.append(analysis)

            if not analysis.is_grounded:
                ungrounded.append(analysis.claim)

        return analyses, ungrounded

    def _parse_level(self, level_str: str) -> HallucinationLevel:
        """Parse hallucination level from string.

        Args:
            level_str: Level string from LLM response.

        Returns:
            HallucinationLevel enum value.
        """
        try:
            return HallucinationLevel(level_str.lower())
        except ValueError:
            return HallucinationLevel.MEDIUM

    def _create_empty_result(self) -> HallucinationResult:
        """Create result for empty answer."""
        return HallucinationResult(
            is_hallucination=False,
            level=HallucinationLevel.NONE,
            overall_score=1.0,
            raw_response="Empty answer provided",
        )

    def _create_no_claims_result(self) -> HallucinationResult:
        """Create result when no claims are extracted."""
        return HallucinationResult(
            is_hallucination=False,
            level=HallucinationLevel.NONE,
            overall_score=1.0,
            raw_response="No claims extracted",
        )

    def detect(
        self,
        answer: str,
        contexts: list[Document],
    ) -> HallucinationResult:
        """Detect hallucinations in an answer.

        Args:
            answer: Generated answer to check.
            contexts: Retrieved context documents.

        Returns:
            HallucinationResult with analysis.
        """
        if not answer or not answer.strip():
            return self._create_empty_result()

        claims = self._extract_claims_from_answer(answer)

        if not claims:
            return self._create_no_claims_result()

        context_text = format_context_text(contexts)
        claims_text = self._format_claims_for_verification(claims)

        llm = self._get_llm()
        prompt = self.VERIFICATION_PROMPT.format(
            context=context_text,
            claims=claims_text,
        )

        response = llm.invoke(prompt)

        try:
            result = extract_json_from_response(response.content)

            overall_score = float(result.get("overall_groundedness_score", 0.0))
            level = self._parse_level(result.get("hallucination_level", "medium"))

            claim_analyses, ungrounded_claims = self._parse_claim_analyses(
                result.get("claims", [])
            )

            return HallucinationResult(
                is_hallucination=overall_score < self.config.threshold,
                level=level,
                overall_score=overall_score,
                claims=claim_analyses,
                ungrounded_claims=ungrounded_claims,
                suggestions=result.get("suggestions", []) if self.config.enable_suggestions else [],
                raw_response=response.content,
            )

        except ValueError:
            return HallucinationResult(
                is_hallucination=True,
                level=HallucinationLevel.HIGH,
                overall_score=0.0,
                ungrounded_claims=claims,
                raw_response=f"Failed to parse response",
            )

    def quick_check(
        self,
        answer: str,
        contexts: list[Document],
    ) -> tuple[bool, float]:
        """Quick check for hallucinations without detailed analysis.

        Args:
            answer: Generated answer.
            contexts: Retrieved context documents.

        Returns:
            Tuple of (is_grounded, confidence_score).
        """
        result = self.detect(answer, contexts)
        return result.overall_score >= self.config.threshold, result.overall_score


class GuardrailsManager:
    """Manages guardrails for RAG responses."""

    def __init__(
        self,
        hallucination_config: HallucinationConfig | None = None,
        enable_hallucination_detection: bool = True,
    ) -> None:
        self.enable_hallucination_detection = enable_hallucination_detection
        self._detector: HallucinationDetector | None = None
        self._hallucination_config = hallucination_config

    def _get_detector(self) -> HallucinationDetector:
        if self._detector is None:
            self._detector = HallucinationDetector(self._hallucination_config)
        return self._detector

    def _check_empty_response(self, answer: str) -> dict[str, Any]:
        """Check if response is empty.

        Args:
            answer: Generated answer.

        Returns:
            Result dictionary with failure status if empty.
        """
        if not answer or not answer.strip():
            return {
                "passed": False,
                "warnings": ["Empty response generated"],
            }
        return {"passed": True, "warnings": []}

    def _check_hallucination(
        self,
        answer: str,
        contexts: list[Document],
    ) -> tuple[dict[str, Any], list[str]]:
        """Check response for hallucinations.

        Args:
            answer: Generated answer.
            contexts: Retrieved context documents.

        Returns:
            Tuple of (check_result, warnings).
        """
        if not self.enable_hallucination_detection:
            return {}, []

        detector = self._get_detector()
        hallucination_result = detector.detect(answer, contexts)

        check = {
            "passed": hallucination_result.overall_score >= detector.config.threshold,
            "score": hallucination_result.overall_score,
            "level": hallucination_result.level.value,
            "ungrounded_claims": hallucination_result.ungrounded_claims[:MAX_DISPLAY_CLAIMS],
        }

        warnings = []
        if hallucination_result.is_hallucination:
            warnings.append(
                f"Hallucination detected (score: {hallucination_result.overall_score:.2f})"
            )

        return {"hallucination": check}, warnings

    def _check_response_quality(self, answer: str, contexts: list[Document]) -> list[str]:
        """Check response quality metrics.

        Args:
            answer: Generated answer.
            contexts: Retrieved context documents.

        Returns:
            List of quality warnings.
        """
        warnings = []

        if len(answer) > MAX_RESPONSE_LENGTH:
            warnings.append("Response exceeds recommended length")

        if contexts:
            context_text = " ".join(doc.page_content for doc in contexts)
            if context_text and len(answer) > len(context_text) * MAX_CONTEXT_RATIO:
                warnings.append(
                    "Response significantly longer than context - potential hallucination"
                )

        return warnings

    def check_response(
        self,
        answer: str,
        contexts: list[Document],
    ) -> dict[str, Any]:
        """Check response against all guardrails.

        Args:
            answer: Generated answer.
            contexts: Retrieved context documents.

        Returns:
            Dictionary with guardrail results.
        """
        empty_check = self._check_empty_response(answer)
        if not empty_check["passed"]:
            return {
                "passed": False,
                "checks": {},
                "warnings": empty_check["warnings"],
            }

        checks, warnings = self._check_hallucination(answer, contexts)
        warnings.extend(self._check_response_quality(answer, contexts))

        return {
            "passed": not warnings,
            "checks": checks,
            "warnings": warnings,
        }

    def get_safe_response(
        self,
        answer: str,
        contexts: list[Document],
    ) -> str:
        """Get a safe version of the response with warnings.

        Args:
            answer: Generated answer.
            contexts: Retrieved context documents.

        Returns:
            Potentially modified answer with warnings.
        """
        guardrail_result = self.check_response(answer, contexts)

        if guardrail_result["passed"]:
            return answer

        warnings = guardrail_result["warnings"]
        if warnings:
            warning_text = "\n".join(f"- {w}" for w in warnings)
            return f"{answer}\n\n---\n⚠️ **Verification Warnings:**\n{warning_text}"

        return answer
