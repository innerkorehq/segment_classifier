from dataclasses import dataclass
from typing import Callable
from segment_classifier.models import (
    InputSegment, ComponentType, ClassificationStage, ClassifiedSegment, SegmentPosition
)
from segment_classifier.utils.html_normalizer import NormalizedSegment


@dataclass
class ClassificationRule:
    name: str
    condition: Callable[[InputSegment, NormalizedSegment], bool]
    component_type: ComponentType
    confidence: float
    priority: int = 50


class RuleBasedClassifier:
    """
    Apply ordered classification rules to a segment.
    Rules evaluated highest priority first.
    Returns ClassifiedSegment with stage=RULE_BASED or None if no match.
    """

    def __init__(self, confidence_threshold: float = 0.90):
        self.confidence_threshold = confidence_threshold
        self.rules: list[ClassificationRule] = self._build_rules()

    def _build_rules(self) -> list[ClassificationRule]:
        """
        Instantiate all rules described above.
        Sort by priority descending before returning.
        """
        rules = [
            # Position rules
            ClassificationRule(
                name="top_header_nav",
                condition=lambda s, n: s.position_hint == SegmentPosition.TOP and n.root_tag in {"header", "nav"},
                component_type=ComponentType.LAYOUT_HEADER,
                confidence=0.97,
                priority=100
            ),
            ClassificationRule(
                name="bottom_footer",
                condition=lambda s, n: s.position_hint == SegmentPosition.BOTTOM and n.root_tag in {"footer"},
                component_type=ComponentType.LAYOUT_FOOTER,
                confidence=0.97,
                priority=99
            ),
            ClassificationRule(
                name="top_has_nav",
                condition=lambda s, n: s.position_hint == SegmentPosition.TOP and n.child_tag_counts.get("nav", 0) > 0,
                component_type=ComponentType.LAYOUT_NAV,
                confidence=0.93,
                priority=98
            ),

            # Tag-based rules
            ClassificationRule(
                name="root_footer",
                condition=lambda s, n: n.root_tag == "footer",
                component_type=ComponentType.LAYOUT_FOOTER,
                confidence=0.99,
                priority=90
            ),
            ClassificationRule(
                name="root_header",
                condition=lambda s, n: n.root_tag == "header",
                component_type=ComponentType.LAYOUT_HEADER,
                confidence=0.99,
                priority=90
            ),
            ClassificationRule(
                name="nav_element",
                condition=lambda s, n: n.root_tag == "nav" or (n.child_tag_counts.get("nav", 0) > 0 and n.child_tag_counts.get("a", 0) > 3),
                component_type=ComponentType.LAYOUT_NAV,
                confidence=0.95,
                priority=85
            ),
            ClassificationRule(
                name="form_element",
                condition=lambda s, n: n.root_tag == "form" or n.child_tag_counts.get("input", 0) > 1,
                component_type=ComponentType.UI_FORM,
                confidence=0.93,
                priority=84
            ),
            ClassificationRule(
                name="table_element",
                condition=lambda s, n: n.child_tag_counts.get("table", 0) > 0,
                component_type=ComponentType.UI_TABLE,
                confidence=0.90,
                priority=83
            ),
            ClassificationRule(
                name="modal_element",
                condition=lambda s, n: n.child_tag_counts.get("dialog", 0) > 0 or "modal" in n.class_tokens,
                component_type=ComponentType.UI_MODAL,
                confidence=0.91,
                priority=82
            ),

            # Sibling repetition rules
            ClassificationRule(
                name="product_card_repetition",
                condition=lambda s, n: s.sibling_count >= 3 and "card" in n.class_tokens,
                component_type=ComponentType.COLLECTION_PRODUCT_CARD,
                confidence=0.88,
                priority=70
            ),
            ClassificationRule(
                name="product_card_img_header",
                condition=lambda s, n: s.sibling_count >= 3 and n.child_tag_counts.get("img", 0) > 0 and (n.child_tag_counts.get("h2", 0) > 0 or n.child_tag_counts.get("h3", 0) > 0),
                component_type=ComponentType.COLLECTION_PRODUCT_CARD,
                confidence=0.85,
                priority=69
            ),
            ClassificationRule(
                name="blog_card_repetition",
                condition=lambda s, n: s.sibling_count >= 3 and "article" in n.class_tokens,
                component_type=ComponentType.COLLECTION_BLOG_CARD,
                confidence=0.87,
                priority=68
            ),
            ClassificationRule(
                name="nav_list_repetition",
                condition=lambda s, n: s.sibling_count >= 5 and n.child_tag_counts.get("li", 0) > 4,
                component_type=ComponentType.LAYOUT_NAV,
                confidence=0.86,
                priority=67
            ),

            # URL hint rules
            ClassificationRule(
                name="url_product_card",
                condition=lambda s, n: any(x in s.url_path_segments for x in ["product", "shop", "store"]) and s.sibling_count >= 2,
                component_type=ComponentType.COLLECTION_PRODUCT_CARD,
                confidence=0.84,
                priority=60
            ),
            ClassificationRule(
                name="url_blog_card",
                condition=lambda s, n: any(x in s.url_path_segments for x in ["blog", "post", "article"]),
                component_type=ComponentType.COLLECTION_BLOG_CARD,
                confidence=0.83,
                priority=59
            ),
            ClassificationRule(
                name="url_news_item",
                condition=lambda s, n: "news" in s.url_path_segments,
                component_type=ComponentType.COLLECTION_NEWS_ITEM,
                confidence=0.83,
                priority=58
            ),

            # Class token rules
            ClassificationRule(
                name="class_product_price",
                condition=lambda s, n: "price" in n.class_tokens and n.child_tag_counts.get("img", 0) > 0,
                component_type=ComponentType.COLLECTION_PRODUCT_CARD,
                confidence=0.89,
                priority=50
            ),
            ClassificationRule(
                name="class_hero",
                condition=lambda s, n: "hero" in n.class_tokens and n.dom_depth <= 4,
                component_type=ComponentType.SECTION_HERO,
                confidence=0.91,
                priority=49
            ),
            ClassificationRule(
                name="class_testimonial",
                condition=lambda s, n: "testimonial" in n.class_tokens,
                component_type=ComponentType.SECTION_TESTIMONIAL,
                confidence=0.90,
                priority=48
            ),
            ClassificationRule(
                name="class_faq",
                condition=lambda s, n: "faq" in n.class_tokens,
                component_type=ComponentType.SECTION_FAQ,
                confidence=0.90,
                priority=47
            ),
            ClassificationRule(
                name="class_pricing",
                condition=lambda s, n: "pricing" in n.class_tokens,
                component_type=ComponentType.SECTION_PRICING,
                confidence=0.91,
                priority=46
            ),
            ClassificationRule(
                name="class_breadcrumb",
                condition=lambda s, n: "breadcrumb" in n.class_tokens or "breadcrumb" in n.attrs_fingerprint,
                component_type=ComponentType.LAYOUT_BREADCRUMB,
                confidence=0.95,
                priority=45
            ),
            ClassificationRule(
                name="class_carousel",
                condition=lambda s, n: "carousel" in n.class_tokens or n.child_tag_counts.get("swiper-slide", 0) > 0,
                component_type=ComponentType.UI_CAROUSEL,
                confidence=0.89,
                priority=44
            ),
            ClassificationRule(
                name="class_pagination",
                condition=lambda s, n: "pagination" in n.class_tokens,
                component_type=ComponentType.UI_PAGINATION,
                confidence=0.94,
                priority=43
            ),
            ClassificationRule(
                name="class_search",
                condition=lambda s, n: "search" in n.class_tokens and n.child_tag_counts.get("input", 0) > 0,
                component_type=ComponentType.UI_SEARCH,
                confidence=0.93,
                priority=42
            ),
            ClassificationRule(
                name="class_cta",
                condition=lambda s, n: "cta" in n.class_tokens,
                component_type=ComponentType.SECTION_CTA,
                confidence=0.88,
                priority=41
            ),

            # Text density rules
            ClassificationRule(
                name="text_article",
                condition=lambda s, n: n.text_density_ratio > 0.6 and n.root_tag in {"article", "main"},
                component_type=ComponentType.CONTENT_ARTICLE,
                confidence=0.85,
                priority=30
            ),
            ClassificationRule(
                name="text_rich_text",
                condition=lambda s, n: n.text_density_ratio > 0.5 and n.dom_depth <= 3,
                component_type=ComponentType.CONTENT_RICH_TEXT,
                confidence=0.80,
                priority=29
            ),

            # Media rules
            ClassificationRule(
                name="media_img",
                condition=lambda s, n: n.child_tag_counts.get("img", 0) > 3 and n.text_density_ratio < 0.1,
                component_type=ComponentType.CONTENT_MEDIA,
                confidence=0.87,
                priority=20
            ),
            ClassificationRule(
                name="media_video",
                condition=lambda s, n: n.child_tag_counts.get("video", 0) > 0,
                component_type=ComponentType.CONTENT_MEDIA,
                confidence=0.92,
                priority=19
            ),
        ]

        rules.sort(key=lambda r: r.priority, reverse=True)
        return rules

    def classify(
        self,
        segment: InputSegment,
        normalized: NormalizedSegment,
    ) -> ClassifiedSegment | None:
        """
        Try each rule in order. Return first match above confidence_threshold.
        Return None if no rule fires.
        """
        for rule in self.rules:
            if rule.condition(segment, normalized):
                if rule.confidence >= self.confidence_threshold:
                    return ClassifiedSegment(
                        segment_id=segment.segment_id,
                        page_url=segment.page_url,
                        page_slug=segment.page_slug,
                        raw_html=segment.raw_html,
                        text_content=segment.text_content,
                        position_hint=segment.position_hint,
                        component_type=rule.component_type,
                        classification_stage=ClassificationStage.RULE_BASED,
                        confidence=rule.confidence,
                        fingerprint_hash=normalized.fingerprint_hash()
                    )
        return None
