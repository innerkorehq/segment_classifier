from sklearn.feature_extraction.text import TfidfVectorizer
from segment_classifier.cache.l2_cache import L2FuzzyCache
from segment_classifier.config import CacheConfig
from segment_classifier.models import (
    InputSegment, ClassifiedSegment, ClassificationStage, ComponentType
)
from segment_classifier.utils.html_normalizer import NormalizedSegment


class FuzzyClusterStage:
    def __init__(self, cache: L2FuzzyCache, config: CacheConfig):
        self.cache = cache
        self.config = config

        # Use TfidfVectorizer, but fit it once on a known dummy vocabulary
        # or just use sklearn's HashingVectorizer which is stateless.
        # But since prompt specifies "use sklearn TfidfVectorizer", we must pre-fit it
        # or use HashingVectorizer with TfidfTransformer. HashingVectorizer directly
        # isn't TfidfVectorizer. Let's stick to HashingVectorizer + tfidf for stability?
        # Actually, prompt says "use sklearn TfidfVectorizer". We can use TfidfVectorizer
        # but with a fixed vocabulary if we hash. Or we can just build a stateless tfidf
        # using HashingVectorizer + TfidfTransformer.
        # But wait, code review said: "You must either use a HashingVectorizer... or pre-fit the vectorizer".
        # Let's use HashingVectorizer + TfidfTransformer to get TF-IDF scaling on fixed dimensions.
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        from sklearn.pipeline import Pipeline
        self.vectorizer = Pipeline([
            ('hash', HashingVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                n_features=512,
                lowercase=True,
                norm=None # Let TfidfTransformer handle normalization
            )),
            ('tfidf', TfidfTransformer())
        ])
        # Fit once on empty/dummy string to initialize tfidf idf_ to smooth values
        self.vectorizer.fit(["dummy"])
        self._is_fitted = True

    def _build_fingerprint_string(self, normalized: NormalizedSegment) -> str:
        return f"{normalized.skeleton} {normalized.attrs_fingerprint} {' '.join(normalized.class_tokens)}"

    def _vectorize(self, fingerprint_string: str) -> list[float]:
        matrix = self.vectorizer.transform([fingerprint_string])
        return matrix.toarray()[0].tolist()

    async def classify(
        self,
        segment: InputSegment,
        normalized: NormalizedSegment,
        fingerprint_hash: str,
    ) -> ClassifiedSegment | None:
        """
        1. Build fingerprint string
        2. Vectorize
        3. find_nearest in L2Cache
        4. Return ClassifiedSegment (stage=L2_FUZZY_CACHE) or None
        """
        fingerprint_string = self._build_fingerprint_string(normalized)
        vector = self._vectorize(fingerprint_string)

        nearest = await self.cache.find_nearest(vector, self.config.l2_similarity_threshold)
        if nearest:
            # Penalty of 0.05 for fuzzy
            confidence = max(0.0, nearest.confidence - 0.05)

            return ClassifiedSegment(
                segment_id=segment.segment_id,
                page_url=segment.page_url,
                page_slug=segment.page_slug,
                raw_html=segment.raw_html,
                text_content=segment.text_content,
                position_hint=segment.position_hint,
                component_type=nearest.component_type,
                classification_stage=ClassificationStage.L2_FUZZY_CACHE,
                confidence=confidence,
                fingerprint_hash=fingerprint_hash,
                cluster_id=nearest.cluster_id
            )

        return None

    async def register(
        self,
        fingerprint_hash: str,
        normalized: NormalizedSegment,
        component_type: ComponentType,
        confidence: float,
    ) -> None:
        """
        Called after LLM resolves a segment to register it in L2 for future lookup.
        """
        fingerprint_string = self._build_fingerprint_string(normalized)
        vector = self._vectorize(fingerprint_string)

        nearest = await self.cache.find_nearest(vector, self.config.l2_similarity_threshold)
        if nearest:
            await self.cache.add_to_cluster(nearest.cluster_id, fingerprint_hash, vector)
        else:
            await self.cache.create_cluster(fingerprint_hash, vector, component_type, confidence)
