from segment_classifier.models import InputSegment
from segment_classifier.utils.html_normalizer import normalize_segment, NormalizedSegment

def compute_fingerprint(segment: InputSegment) -> tuple[NormalizedSegment, str]:
    """
    Computes NormalizedSegment + fingerprint_hash for a given InputSegment.
    """
    normalized = normalize_segment(segment.raw_html, segment.text_content)
    fingerprint_hash = normalized.fingerprint_hash()
    return normalized, fingerprint_hash
