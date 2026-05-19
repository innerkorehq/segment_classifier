from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ClassificationStage(str, Enum):
    RULE_BASED = "rule_based"
    L1_EXACT_CACHE = "l1_exact_cache"
    L2_FUZZY_CACHE = "l2_fuzzy_cache"
    LLM = "llm"


class SegmentPosition(str, Enum):
    TOP = "top"           # top 5% of page
    BOTTOM = "bottom"     # bottom 10% of page
    MIDDLE = "middle"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    # Layout
    LAYOUT_HEADER = "layout.header"
    LAYOUT_FOOTER = "layout.footer"
    LAYOUT_NAV = "layout.nav"
    LAYOUT_SIDEBAR = "layout.sidebar"
    LAYOUT_BREADCRUMB = "layout.breadcrumb"

    # Collections
    COLLECTION_PRODUCT_CARD = "collection.product_card"
    COLLECTION_PRODUCT_LIST = "collection.product_list"
    COLLECTION_BLOG_CARD = "collection.blog_card"
    COLLECTION_BLOG_LIST = "collection.blog_list"
    COLLECTION_NEWS_ITEM = "collection.news_item"
    COLLECTION_NEWS_LIST = "collection.news_list"

    # Sections
    SECTION_HERO = "section.hero"
    SECTION_FEATURE_GRID = "section.feature_grid"
    SECTION_TESTIMONIAL = "section.testimonial"
    SECTION_CTA = "section.cta"
    SECTION_FAQ = "section.faq"
    SECTION_PRICING = "section.pricing"

    # UI Elements
    UI_FORM = "ui.form"
    UI_MODAL = "ui.modal"
    UI_TABLE = "ui.table"
    UI_CAROUSEL = "ui.carousel"
    UI_PAGINATION = "ui.pagination"
    UI_SEARCH = "ui.search"

    # Content
    CONTENT_ARTICLE = "content.article"
    CONTENT_RICH_TEXT = "content.rich_text"
    CONTENT_MEDIA = "content.media"

    UNKNOWN = "unknown"


class InputSegment(BaseModel):
    """Raw segment from the page-segmenter tool."""
    segment_id: str
    page_url: str
    page_slug: str
    raw_html: str
    text_content: str
    position_hint: SegmentPosition = SegmentPosition.UNKNOWN
    dom_position: str = ""                    # CSS selector path e.g. "main > section:nth-child(2)"
    sibling_count: int = 0                    # how many same-fingerprint siblings on same page
    url_path_segments: list[str] = Field(default_factory=list)  # e.g. ["products", "shoes"]


class ClassifiedSegment(BaseModel):
    """Segment with classification result and metadata."""
    segment_id: str
    page_url: str
    page_slug: str
    raw_html: str
    text_content: str
    position_hint: SegmentPosition

    # Classification output
    component_type: ComponentType
    classification_stage: ClassificationStage
    confidence: float = Field(ge=0.0, le=1.0)

    # Fingerprint computed during pipeline
    fingerprint_hash: str = ""
    cluster_id: str | None = None

    # LLM metadata (populated only for stage=LLM)
    llm_model_used: str | None = None
    llm_raw_response: str | None = None


class FingerprintRecord(BaseModel):
    """Stored in L1 cache: fingerprint → classification."""
    fingerprint_hash: str
    component_type: ComponentType
    confidence: float
    hit_count: int = 1
    example_segment_id: str = ""


class ClusterRecord(BaseModel):
    """Stored in L2 cache: cluster of similar fingerprints."""
    cluster_id: str
    centroid_vector: list[float]
    component_type: ComponentType
    confidence: float
    member_fingerprints: list[str] = Field(default_factory=list)


class LLMClassificationRequest(BaseModel):
    """Batch item sent to LLM. Use the provided raw HTML in normalized_html to understand purpose and content."""
    segment_id: str
    fingerprint_hash: str
    normalized_html: str          # raw HTML content of the segment
    position_hint: SegmentPosition
    sibling_count: int
    url_hints: list[str]
    dom_depth: int
    child_tag_counts: dict[str, int]
    text_density_ratio: float


class LLMClassificationResult(BaseModel):
    """Parsed result from LLM for one segment."""
    segment_id: str
    component_type: ComponentType
    confidence: float
    reasoning: str


class PipelineResult(BaseModel):
    """Final output of the full classification pipeline run."""
    total_segments: int
    classified: list[ClassifiedSegment]
    stage_breakdown: dict[ClassificationStage, int]
    llm_calls_made: int
    llm_model_usage: dict[str, int]   # model_name → call count
    cache_hit_rate: float
