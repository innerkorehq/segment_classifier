import asyncio
import json
import logging
import litellm
from typing import Any
from segment_classifier.models import (
    InputSegment, ClassifiedSegment, LLMClassificationRequest,
    LLMClassificationResult, ClassificationStage, ComponentType
)
from segment_classifier.utils.html_normalizer import NormalizedSegment
from segment_classifier.config import ClassifierSettings, ModelFeatureConfig

logger = logging.getLogger(__name__)


class LLMBatchClassifier:
    def __init__(self, settings: ClassifierSettings):
        self.settings = settings
        self._semaphore = asyncio.Semaphore(
            settings.litellm_max_concurrent_batches
        )
        self._model_usage: dict[str, int] = {}

        # Set api key globally or per call, litellm supports both
        if settings.litellm_api_key:
            litellm.api_key = settings.litellm_api_key

    def select_model(
        self,
        normalized: NormalizedSegment,
        segment: InputSegment,
    ) -> str:
        """Feature-based model routing."""
        cfg = self.settings.model_routing

        # 1. High complexity
        if (normalized.dom_depth > cfg.high_complexity_dom_depth_threshold or
            normalized.unique_tag_count > cfg.high_complexity_unique_tag_threshold or
            segment.sibling_count == 0):
            return cfg.high_complexity_model

        # 2. Fast
        elif (normalized.dom_depth <= cfg.fast_model_max_dom_depth and
              normalized.unique_tag_count <= 3):
            return cfg.fast_model

        # 3. Standard
        else:
            return cfg.standard_model

    def _build_request(
        self,
        segment: InputSegment,
        normalized: NormalizedSegment,
        fingerprint_hash: str,
    ) -> LLMClassificationRequest:
        """Construct LLMClassificationRequest from segment + normalized data."""
        return LLMClassificationRequest(
            segment_id=segment.segment_id,
            fingerprint_hash=fingerprint_hash,
            normalized_html=normalized.skeleton,
            position_hint=segment.position_hint,
            sibling_count=segment.sibling_count,
            url_hints=segment.url_path_segments,
            dom_depth=normalized.dom_depth,
            child_tag_counts=normalized.child_tag_counts,
            text_density_ratio=normalized.text_density_ratio
        )

    async def _call_litellm(
        self,
        model: str,
        requests: list[LLMClassificationRequest],
    ) -> list[LLMClassificationResult]:
        """
        Make one LiteLLM acompletion call for a batch.
        Parse response. Return list of results.
        On error: return UNKNOWN for all items.
        """
        prompt = """You are an expert UI component classifier. Given a list of HTML segment descriptors, classify each into exactly one component type.

Available component types:
"""
        prompt += ", ".join([c.value for c in ComponentType]) + "\n\n"
        prompt += """For each segment respond with valid JSON array only (no markdown, no explanation):
[
  {
    "segment_id": "...",
    "component_type": "...",
    "confidence": 0.0-1.0,
    "reasoning": "one sentence"
  }
]

Rules:
- Use normalized_html structure only, ignore content values
- sibling_count >= 3 strongly suggests a collection item
- position_hint=top/bottom suggests layout components
- url_hints provide page context
- Respond ONLY with the JSON array. No preamble."""

        user_content = f"Classify these {len(requests)} segments:\n"
        user_content += json.dumps([r.model_dump() for r in requests], indent=2)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                timeout=self.settings.litellm_timeout_seconds,
            )

            # Record usage
            self._model_usage[model] = self._model_usage.get(model, 0) + 1

            raw_response = response.choices[0].message.content
            # Strip markdown
            raw_response = raw_response.strip()
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            elif raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]

            parsed = json.loads(raw_response.strip())

            results = []
            for item in parsed:
                try:
                    results.append(LLMClassificationResult.model_validate(item))
                except Exception as e:
                    logger.warning(f"Error validating LLM response item: {e}")
                    results.append(LLMClassificationResult(
                        segment_id=item.get("segment_id", ""),
                        component_type=ComponentType.UNKNOWN,
                        confidence=0.0,
                        reasoning=f"Parse error: {e}"
                    ))

            # Ensure all segments are accounted for
            parsed_ids = {r.segment_id for r in results}
            for req in requests:
                if req.segment_id not in parsed_ids:
                    results.append(LLMClassificationResult(
                        segment_id=req.segment_id,
                        component_type=ComponentType.UNKNOWN,
                        confidence=0.0,
                        reasoning="Missing from LLM response"
                    ))

            return results

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            self._model_usage[model] = self._model_usage.get(model, 0) + 1 # Still count as call
            return [
                LLMClassificationResult(
                    segment_id=r.segment_id,
                    component_type=ComponentType.UNKNOWN,
                    confidence=0.0,
                    reasoning=f"LLM Error: {e}"
                )
                for r in requests
            ]

    async def classify_batch(
        self,
        items: list[tuple[InputSegment, NormalizedSegment, str]],
    ) -> list[ClassifiedSegment]:
        """
        1. Group items by selected model
        2. Split each group into sub-batches of litellm_batch_size
        3. asyncio.gather all sub-batches under semaphore
        4. Return flat list of ClassifiedSegments
        """
        # Group by model
        grouped: dict[str, list[tuple[InputSegment, NormalizedSegment, str]]] = {}
        for item in items:
            segment, normalized, _ = item
            model = self.select_model(normalized, segment)
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(item)

        all_classified: list[ClassifiedSegment] = []

        async def process_subbatch(model_name: str, subbatch: list[tuple[InputSegment, NormalizedSegment, str]]):
            async with self._semaphore:
                requests = [self._build_request(s, n, h) for s, n, h in subbatch]
                results = await self._call_litellm(model_name, requests)

                # Map results back
                result_map = {r.segment_id: r for r in results}

                for segment, normalized, fp_hash in subbatch:
                    res = result_map.get(segment.segment_id)
                    if res:
                        all_classified.append(
                            ClassifiedSegment(
                                segment_id=segment.segment_id,
                                page_url=segment.page_url,
                                page_slug=segment.page_slug,
                                raw_html=segment.raw_html,
                                text_content=segment.text_content,
                                position_hint=segment.position_hint,
                                component_type=res.component_type,
                                classification_stage=ClassificationStage.LLM,
                                confidence=res.confidence,
                                fingerprint_hash=fp_hash,
                                llm_model_used=model_name,
                                llm_raw_response=res.reasoning # Hack: put reasoning here
                            )
                        )
                    else:
                        all_classified.append(
                            ClassifiedSegment(
                                segment_id=segment.segment_id,
                                page_url=segment.page_url,
                                page_slug=segment.page_slug,
                                raw_html=segment.raw_html,
                                text_content=segment.text_content,
                                position_hint=segment.position_hint,
                                component_type=ComponentType.UNKNOWN,
                                classification_stage=ClassificationStage.LLM,
                                confidence=0.0,
                                fingerprint_hash=fp_hash,
                                llm_model_used=model_name
                            )
                        )

        tasks = []
        for model_name, group_items in grouped.items():
            # Split into sub-batches
            for i in range(0, len(group_items), self.settings.litellm_batch_size):
                subbatch = group_items[i:i + self.settings.litellm_batch_size]
                tasks.append(process_subbatch(model_name, subbatch))

        await asyncio.gather(*tasks)

        # Ensure the returned list is in the exact order as the input items
        result_map = {res.segment_id: res for res in all_classified}
        ordered_classified = []
        for segment, _, _ in items:
            if segment.segment_id in result_map:
                ordered_classified.append(result_map[segment.segment_id])
            else:
                # Fallback, though process_subbatch should populate it
                ordered_classified.append(
                    ClassifiedSegment(
                        segment_id=segment.segment_id,
                        page_url=segment.page_url,
                        page_slug=segment.page_slug,
                        raw_html=segment.raw_html,
                        text_content=segment.text_content,
                        position_hint=segment.position_hint,
                        component_type=ComponentType.UNKNOWN,
                        classification_stage=ClassificationStage.LLM,
                        confidence=0.0,
                        fingerprint_hash=""
                    )
                )
        return ordered_classified

    @property
    def model_usage(self) -> dict[str, int]:
        return dict(self._model_usage)
