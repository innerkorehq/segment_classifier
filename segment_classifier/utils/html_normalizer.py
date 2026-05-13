import hashlib
import json
import re
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag, NavigableString


STRUCTURAL_CLASS_PATTERN = re.compile(
    r'\b(card|grid|list|item|hero|nav|menu|header|footer|sidebar|'
    r'form|modal|badge|price|rating|carousel|pagination|search|'
    r'feature|testimonial|cta|faq|pricing|article|media|table|'
    r'product|blog|news|collection|section|widget)\b',
    re.IGNORECASE
)

PRESENTATIONAL_CLASS_PATTERN = re.compile(
    r'\b(mt|mb|ml|mr|mx|my|pt|pb|pl|pr|px|py|w-|h-|text-|bg-|'
    r'border|rounded|shadow|flex|grid-cols|gap|p-|m-|font-|'
    r'color|opacity|z-|hidden|block|inline)\b'
)

STRUCTURAL_ATTRS = {"role", "type", "aria-label", "aria-role", "data-component", "data-type"}


@dataclass
class NormalizedSegment:
    skeleton: str
    attrs_fingerprint: str
    class_tokens: list[str]
    child_tag_counts: dict[str, int]
    dom_depth: int
    root_tag: str
    text_density_ratio: float
    unique_tag_count: int

    def fingerprint_hash(self) -> str:
        payload = {
            "skeleton": self.skeleton,
            "attrs": self.attrs_fingerprint,
            "classes": sorted(self.class_tokens),
            "counts": self.child_tag_counts,
            "depth": self.dom_depth,
            "root": self.root_tag,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()


def normalize_segment(html: str, text_content: str) -> NormalizedSegment:
    """
    Parse HTML and extract structural fingerprint components.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find()
    if not root or not isinstance(root, Tag):
        return NormalizedSegment("", "", [], {}, 0, "unknown", 0.0, 0)

    skeleton = _extract_skeleton(root)
    attrs_fp = _extract_attrs_fingerprint(root)
    class_tokens = _extract_class_tokens(root)
    child_counts = _count_tags(root)
    depth = _max_depth(root)
    text_ratio = len(text_content) / max(len(html), 1)
    unique_tags = len(child_counts)

    return NormalizedSegment(
        skeleton=skeleton,
        attrs_fingerprint=attrs_fp,
        class_tokens=class_tokens,
        child_tag_counts=child_counts,
        dom_depth=depth,
        root_tag=root.name,
        text_density_ratio=round(text_ratio, 4),
        unique_tag_count=unique_tags,
    )


def _extract_skeleton(tag: Tag, depth: int = 0, max_depth: int = 8) -> str:
    """Recursive tag-name-only skeleton. Siblings joined with '+', children with '>'."""
    if depth >= max_depth:
        return tag.name

    child_skeletons = []
    for child in tag.children:
        if isinstance(child, Tag):
            child_skeletons.append(_extract_skeleton(child, depth + 1, max_depth))

    if not child_skeletons:
        return tag.name

    return f"{tag.name}>" + "+".join(child_skeletons)


def _extract_attrs_fingerprint(tag: Tag) -> str:
    """Walk all tags, keep only STRUCTURAL_ATTRS values and href/src presence booleans."""
    parts = []

    def walk(node: Tag):
        node_parts = []
        for attr in STRUCTURAL_ATTRS:
            val = node.get(attr)
            if val:
                if isinstance(val, list):
                    val = " ".join(val)
                node_parts.append(f"{attr}={val}")

        if node.has_attr("href"):
            node_parts.append("has_href=true")
        if node.has_attr("src"):
            node_parts.append("has_src=true")

        if node_parts:
            parts.append(f"{node.name}[" + ",".join(sorted(node_parts)) + "]")

        for child in node.children:
            if isinstance(child, Tag):
                walk(child)

    walk(tag)
    return "|".join(parts)


def _extract_class_tokens(tag: Tag) -> list[str]:
    """Extract class names matching STRUCTURAL_CLASS_PATTERN from all tags."""
    tokens = set()

    def walk(node: Tag):
        classes = node.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        for c in classes:
            if STRUCTURAL_CLASS_PATTERN.search(c) and not PRESENTATIONAL_CLASS_PATTERN.search(c):
                tokens.add(c)
        for child in node.children:
            if isinstance(child, Tag):
                walk(child)

    walk(tag)
    return list(tokens)


def _count_tags(tag: Tag) -> dict[str, int]:
    """Count occurrences of each tag name in the full subtree."""
    counts = {}

    def walk(node: Tag):
        counts[node.name] = counts.get(node.name, 0) + 1
        for child in node.children:
            if isinstance(child, Tag):
                walk(child)

    # Count tags in the full subtree including the root
    walk(tag)

    return counts


def _max_depth(tag: Tag, current: int = 0) -> int:
    """Return maximum nesting depth."""
    child_depths = [current]
    for child in tag.children:
        if isinstance(child, Tag):
            child_depths.append(_max_depth(child, current + 1))
    return max(child_depths)
