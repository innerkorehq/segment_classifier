import asyncio
import logging
from segment_classifier.models import (
    InputSegment, ClassifiedSegment, PipelineResult, ClassificationStage, FingerprintRecord
)
from segment_classifier.config import ClassifierSettings
from segment_classifier.utils.html_normalizer import NormalizedSegment
from segment_classifier.stages.rule_based import RuleBasedClassifier
from segment_classifier.stages.fingerprint import compute_fingerprint
from segment_classifier.stages.fuzzy_cluster import FuzzyClusterStage
from segment_classifier.stages.llm_classifier import LLMBatchClassifier
from segment_classifier.cache.l1_cache import L1FingerprintCache
from segment_classifier.cache.l2_cache import L2FuzzyCache

logger = logging.getLogger(__name__)

class ClassifierPipeline:
    def __init__(self, settings: ClassifierSettings):
        self.settings = settings
        self.rule_classifier = RuleBasedClassifier(
            confidence_threshold=settings.rule_based_confidence_threshold
        )
        self.l1_cache = L1FingerprintCache(
            cache_path=settings.cache.l1_cache_path,
            auto_persist_every=50
        )
        self.l2_cache = L2FuzzyCache(
            cache_path=settings.cache.l2_cache_path,
            embeddings_path=settings.cache.l2_embeddings_path,
            similarity_threshold=settings.cache.l2_similarity_threshold,
            max_cluster_size=settings.cache.l2_max_cluster_size,
            persist_on_update=settings.cache.persist_on_update
        )
        self.fuzzy_stage = FuzzyClusterStage(self.l2_cache, settings.cache)
        self.llm_classifier = LLMBatchClassifier(settings)

    async def initialize(self) -> None:
        """Load caches from disk."""
        await asyncio.gather(
            self.l1_cache.load(),
            self.l2_cache.load()
        )

    async def shutdown(self) -> None:
        """Persist caches to disk."""
        await asyncio.gather(
            self.l1_cache.persist(),
            self.l2_cache.persist()
        )

    async def run(self, segments: list[InputSegment]) -> PipelineResult:
        total_segments = len(segments)
        classified: list[ClassifiedSegment] = []

        # Precompute fingerprints concurrently
        loop = asyncio.get_running_loop()

        # We can run compute_fingerprint in an executor or direct async
        # To avoid blocking event loop for too long if many segments, use executor
        fingerprints_list = await asyncio.gather(
            *[loop.run_in_executor(None, compute_fingerprint, seg) for seg in segments]
        )

        fingerprints: dict[str, tuple[NormalizedSegment, str]] = {
            seg.segment_id: fp for seg, fp in zip(segments, fingerprints_list)
        }

        pending = segments.copy()

        # Stage 1: Rule-based
        next_pending = []
        for segment in pending:
            normalized, fp_hash = fingerprints[segment.segment_id]
            result = self.rule_classifier.classify(segment, normalized)
            if result:
                classified.append(result)
                await self.l1_cache.set(fp_hash, FingerprintRecord(
                    fingerprint_hash=fp_hash,
                    component_type=result.component_type,
                    confidence=result.confidence,
                    example_segment_id=segment.segment_id
                ))
            else:
                next_pending.append(segment)
        pending = next_pending

        # Stage 2: L1 Exact Cache
        next_pending = []
        for segment in pending:
            normalized, fp_hash = fingerprints[segment.segment_id]
            record = await self.l1_cache.get(fp_hash)
            if record and record.confidence >= self.settings.l1_min_confidence:
                classified.append(ClassifiedSegment(
                    segment_id=segment.segment_id,
                    page_url=segment.page_url,
                    page_slug=segment.page_slug,
                    raw_html=segment.raw_html,
                    text_content=segment.text_content,
                    position_hint=segment.position_hint,
                    component_type=record.component_type,
                    classification_stage=ClassificationStage.L1_EXACT_CACHE,
                    confidence=record.confidence,
                    fingerprint_hash=fp_hash
                ))
                await self.l1_cache.increment_hit(fp_hash)
            else:
                next_pending.append(segment)
        pending = next_pending

        # Stage 3: L2 Fuzzy Cache
        next_pending = []
        for segment in pending:
            normalized, fp_hash = fingerprints[segment.segment_id]
            result = await self.fuzzy_stage.classify(segment, normalized, fp_hash)
            if result and result.confidence >= self.settings.l2_min_confidence:
                classified.append(result)
            else:
                next_pending.append(segment)
        pending = next_pending

        # Stage 4: LLM Batch
        llm_calls_made = 0
        if pending:
            llm_items = [
                (seg, fingerprints[seg.segment_id][0], fingerprints[seg.segment_id][1])
                for seg in pending
            ]
            llm_results = await self.llm_classifier.classify_batch(llm_items)

            # For each LLM result, register in L1 + L2
            for seg, result in zip(pending, llm_results):
                normalized, fp_hash = fingerprints[seg.segment_id]
                await self.l1_cache.set(fp_hash, FingerprintRecord(
                    fingerprint_hash=fp_hash,
                    component_type=result.component_type,
                    confidence=result.confidence,
                    example_segment_id=seg.segment_id
                ))
                await self.fuzzy_stage.register(
                    fingerprint_hash=fp_hash,
                    normalized=normalized,
                    component_type=result.component_type,
                    confidence=result.confidence
                )

            classified.extend(llm_results)

            # Calculate total LLM batch calls
            grouped_by_model: dict[str, int] = {}
            for item in llm_items:
                seg, norm, _ = item
                model = self.llm_classifier.select_model(norm, seg)
                grouped_by_model[model] = grouped_by_model.get(model, 0) + 1

            for count in grouped_by_model.values():
                import math
                llm_calls_made += math.ceil(count / self.settings.litellm_batch_size)

        stage_breakdown = {stage: 0 for stage in ClassificationStage}
        for c in classified:
            stage_breakdown[c.classification_stage] += 1

        cache_hits = stage_breakdown[ClassificationStage.L1_EXACT_CACHE] + stage_breakdown[ClassificationStage.L2_FUZZY_CACHE]
        cache_hit_rate = cache_hits / total_segments if total_segments > 0 else 0.0

        return PipelineResult(
            total_segments=total_segments,
            classified=classified,
            stage_breakdown=stage_breakdown,
            llm_calls_made=llm_calls_made,
            llm_model_usage=self.llm_classifier.model_usage,
            cache_hit_rate=cache_hit_rate
        )
