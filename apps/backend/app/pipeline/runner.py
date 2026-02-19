from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.config import RuntimeConfig
from app.pipeline.conditioning import ConditioningExtractor
from app.pipeline.streamdiffusion_adapter import StreamDiffusionV2Adapter
from app.pipeline.types import PipelineResult


class RealtimePipeline:
    def __init__(self) -> None:
        self._conditioning_extractor = ConditioningExtractor()
        self._adapter = StreamDiffusionV2Adapter()

    def process(self, frame_bytes: bytes, config: RuntimeConfig) -> PipelineResult:
        with Image.open(BytesIO(frame_bytes)) as raw:
            frame = raw.convert("RGB")

        conditioning = self._conditioning_extractor.extract(frame)
        output = self._adapter.generate(frame, config, conditioning)

        out_buffer = BytesIO()
        output.save(out_buffer, format="JPEG", quality=config.jpeg_quality, optimize=True)
        return PipelineResult(image_bytes=out_buffer.getvalue(), conditioning=conditioning)

    @property
    def runtime_status(self) -> dict[str, str | None]:
        return self._adapter.runtime_status
