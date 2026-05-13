# Segment Classifier

An asynchronous Python library that classifies HTML segments extracted by a page-segmenter into structured component types.

## Overview

The `segment_classifier` implements a 4-stage classification pipeline with progressive fallback to optimize for cost and speed:

1. **Rule-based heuristics** — Zero LLM cost. Uses DOM structure, text density, siblings, and attributes.
2. **L1 exact fingerprint cache** — Zero LLM cost. Exact matching on structural DOM fingerprint hashes.
3. **L2 fuzzy cluster cache** — Zero LLM cost. TF-IDF and cosine similarity on fingerprint tokens.
4. **LLM batch classification** — Batched fallback via LiteLLM with feature-based model routing based on segment complexity.

## Installation

You can install the package using poetry:
```bash
poetry install
```

Or via pip (once published):
```bash
pip install segment-classifier
```

## Setup

The library uses `pydantic-settings` to manage configuration via a `.env` file or environment variables.

Required environment variables:
```env
CLASSIFIER_LITELLM_API_KEY="your-api-key"
```

## Usage

```python
import asyncio
from segment_classifier import ClassifierPipeline
from segment_classifier.config import ClassifierSettings
from segment_classifier.models import InputSegment, SegmentPosition

async def main():
    settings = ClassifierSettings()
    pipeline = ClassifierPipeline(settings)
    await pipeline.initialize()

    segments = [
        InputSegment(
            segment_id="seg_001",
            page_url="https://example.com/products",
            page_slug="products",
            raw_html="<div class='product-card'>...</div>",
            text_content="Product Item",
            position_hint=SegmentPosition.MIDDLE,
            sibling_count=3,
        )
    ]

    result = await pipeline.run(segments)
    await pipeline.shutdown()

    for seg in result.classified:
        print(seg.component_type)

asyncio.run(main())
```

## Caching

Caches are stored by default in `.cache/l1_fingerprints.json` and `.cache/l2_clusters.json` / `.cache/l2_embeddings.npy`.

## Stages Breakdown
Every returned `ClassifiedSegment` will be marked with a `classification_stage` indicating which of the 4 stages resolved the query.
