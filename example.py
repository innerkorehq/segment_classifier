import asyncio
from pprint import pprint

from page_segmenter import find_segments

from segment_classifier import ClassifierPipeline
from segment_classifier.config import ClassifierSettings
from segment_classifier.models import (
    InputSegment,
    SegmentPosition,
)


def infer_position(y: float, page_height: float):
    ratio = y / max(page_height, 1)

    if ratio < 0.2:
        return SegmentPosition.TOP

    if ratio > 0.8:
        return SegmentPosition.BOTTOM

    return SegmentPosition.MIDDLE


async def main():

    # url = "https://getaipage.com"
    url = "https://www.simaltiacorporation.com/cable-lug.html"

    raw_segments = await find_segments(url)

    print("raw_segments:", raw_segments)

    if not raw_segments:
        return

    # ---- estimate page height ----

    page_height = max(
        s["boundingBox"]["y"] + s["boundingBox"]["height"]
        for s in raw_segments
    )

    # ---- convert to InputSegment ----

    input_segments = []

    for idx, seg in enumerate(raw_segments):

        bbox = seg.get("boundingBox", {})

        y = bbox.get("y", 0)

        input_seg = InputSegment(
            segment_id=f"seg_{idx}",
            page_url=url,
            page_slug="home",

            # IMPORTANT:
            # use actual outerHTML for normalization
            raw_html=seg.get("rawHtml", ""),

            text_content=(
                f"role={seg.get('role')} "
                f"selector={seg.get('selector')} "
                f"signals={' '.join(seg.get('identitySignals', []))}"
            ),

            position_hint=infer_position(
                y=y,
                page_height=page_height,
            ),

            sibling_count=len(seg.get("children", [])),
        )

        input_segments.append(input_seg)

    print(f"Prepared {len(input_segments)} segments")

    # ---- classifier ----

    settings = ClassifierSettings()

    pipeline = ClassifierPipeline(settings)

    await pipeline.initialize()

    try:

        result = await pipeline.run(input_segments)

        pprint(result)

    finally:

        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())