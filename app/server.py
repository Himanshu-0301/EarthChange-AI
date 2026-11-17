"""Local web server for the EarthChange JavaScript dashboard."""
from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import sys
import tempfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGES = {
    "before": ROOT / "data" / "samples" / "oscd100" / "oscd100-before.png",
    "after": ROOT / "data" / "samples" / "oscd100" / "oscd100-after.png",
}
sys.path.insert(0, str(ROOT / "src"))
from earthchange.agent import analyze


@dataclass(frozen=True)
class DecodedImage:
    pixels: np.ndarray
    metadata: dict
    crs: str | None = None
    transform: tuple[float, ...] | None = None


def decode_image(filename: str, raw: bytes) -> DecodedImage:
    if Path(filename).suffix.lower() in {".tif", ".tiff"}:
        from rasterio.io import MemoryFile
        with MemoryFile(raw) as mem, mem.open() as ds:
            image = ds.read()
            if not np.all(ds.dataset_mask() == 255):
                raise ValueError(f"{filename}: images with nodata/invalid pixels are not supported yet")
            crs = ds.crs.to_string() if ds.crs else None
            transform = tuple(float(value) for value in ds.transform)[:6]
            resolution = None
            projected_meters = bool(ds.crs and ds.crs.is_projected and ds.crs.linear_units.lower() in {"metre", "meter"})
            unrotated = abs(ds.transform.b) < 1e-12 and abs(ds.transform.d) < 1e-12
            if projected_meters and unrotated:
                resolution = [abs(float(ds.transform.a)), abs(float(ds.transform.e))]
            metadata = {
                "filename": Path(filename).name,
                "format": "GeoTIFF",
                "width": ds.width,
                "height": ds.height,
                "bands": ds.count,
                "crs": crs,
                "transform": list(transform),
                "nodata": ds.nodata,
                "resolution_m": resolution,
            }
    else:
        array = np.asarray(Image.open(io.BytesIO(raw)))
        if array.ndim == 2:
            array = array[:, :, None]
        image = np.moveaxis(array, -1, 0)
        crs = None
        transform = None
        metadata = {
            "filename": Path(filename).name,
            "format": Path(filename).suffix.lower().lstrip(".").upper() or "image",
            "width": int(image.shape[2]),
            "height": int(image.shape[1]),
            "bands": int(image.shape[0]),
            "crs": None,
            "transform": None,
            "nodata": None,
            "resolution_m": None,
        }
    if image.ndim != 3:
        raise ValueError("Image must have one or more bands")
    return DecodedImage(np.asarray(image, dtype=np.float32), metadata, crs, transform)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, status: int, data: dict) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(200, {"ready": True})
            return
        if path in ("/api/sample/before", "/api/sample/after"):
            side = path.rsplit("/", 1)[-1]
            payload = SAMPLE_IMAGES[side].read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Disposition", f'inline; filename="oscd100-{side}.png"')
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/":
            path = "/index.html"
        target = (ROOT / "app" / path.lstrip("/")).resolve()
        if not target.is_relative_to((ROOT / "app").resolve()) or not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/interpret":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_384:
                    raise ValueError("Question text must be under 16 KB")
                payload = json.loads(self.rfile.read(length))
                question = payload.get("question", "")
                if not isinstance(question, str):
                    raise TypeError("Question must be text")
                sys.path.insert(0, str(ROOT / "src"))
                from earthchange.nlp.parser import parse_question

                query = parse_question(question)
                intent_labels = {
                    "GENERAL_CHANGE": "General change",
                    "CHANGE_LOCALIZATION": "Locate change",
                    "CHANGE_QUANTIFICATION": "Measure amount",
                    "REGIONAL_COMPARISON": "Compare regions",
                    "MULTI_STEP_ANALYSIS": "Multi-part request",
                    "UNSUPPORTED": "Outside current scope",
                }
                operation_labels = {
                    "DESCRIBE": "Summarize",
                    "LOCATE": "Locate",
                    "COUNT": "Count pixels",
                    "AREA": "Estimate area",
                    "PERCENTAGE": "Measure percentage",
                    "COMPARE": "Compare",
                }
                region_values = query.parameters.get("comparison_regions") or [query.region.value.lower()]
                action_labels = {
                    "validate_input": "Check image compatibility",
                    "change_detection": "Measure pixel-level change",
                    "region_analysis": "Analyze requested regions",
                    "area_calculation": "Calculate pixel and area totals",
                    "generate_visual_evidence": "Prepare visual evidence",
                }
                steps = []
                explanation = "The question is translated to a structured request before CV runs."
                if query.intent.value == "UNSUPPORTED":
                    explanation = (
                        "This target needs semantic land-cover recognition, which the current "
                        "pixel-change pipeline does not provide."
                    )
                else:
                    from earthchange.agent.planner import plan_for_query

                    plan = plan_for_query(query)
                    steps = [
                        {"tool": step.tool.value, "label": action_labels[step.tool.value]}
                        for step in plan.steps
                    ]
                self.send_json(200, {
                    "question": question,
                    "supported": query.intent.value != "UNSUPPORTED",
                    "intent": query.intent.value,
                    "intent_label": intent_labels[query.intent.value],
                    "target": query.target,
                    "region": query.region.value.lower(),
                    "regions": region_values,
                    "operation": query.operation.value,
                    "operation_label": operation_labels[query.operation.value],
                    "steps": steps,
                    "explanation": explanation,
                    "parser": "Deterministic local language parser",
                })
            except Exception as exc:  # noqa: BLE001 - surface question parsing errors to the UI
                self.send_json(400, {"error": str(exc)})
            return
        if path != "/api/analyze":
            self.send_error(404)
            return
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 128 * 1024 * 1024:
                raise ValueError("Upload must be under 128 MB")
            envelope = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode()
            message = BytesParser(policy=policy.default).parsebytes(envelope + self.rfile.read(length))
            fields = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name:
                    fields[name] = (part.get_filename(), part.get_payload(decode=True))
            before_name, before_bytes = fields["before"]
            after_name, after_bytes = fields["after"]
            before = decode_image(before_name, before_bytes)
            after = decode_image(after_name, after_bytes)
            if before.pixels.shape != after.pixels.shape:
                raise ValueError(f"Image dimensions/bands must match ({before.pixels.shape} vs {after.pixels.shape})")
            if (before.crs != after.crs or before.transform != after.transform) and (
                before.crs or after.crs or before.transform or after.transform
            ):
                raise ValueError("GeoTIFFs must use the same CRS and pixel alignment")
            question = fields.get("question", (None, b"Is there any change?"))[1].decode("utf-8")
            size_text = fields.get("pixel_size", (None, b""))[1].decode("ascii").strip()
            size = float(size_text) if size_text else None
            if size is not None and (not math.isfinite(size) or size < 0):
                raise ValueError("Pixel size must be a finite, non-negative number of metres")
            geotiff_resolution = before.metadata.get("resolution_m")
            if geotiff_resolution:
                if size and any(abs(size - value) > max(0.01, value * 0.01) for value in geotiff_resolution):
                    raise ValueError("Entered pixel size conflicts with the GeoTIFF's projected resolution")
                resolution = tuple(geotiff_resolution)
            else:
                resolution = (size, size) if size is not None and size > 0 else None
            with tempfile.TemporaryDirectory(prefix="earthchange-") as output:
                result = analyze(before.pixels, after.pixels, question, pixel_resolution=resolution, output_dir=output)
                stats = result.evidence.change_stats
                viz_path = result.evidence.visualization_paths.get("change_analysis")
                viz = base64.b64encode(Path(viz_path).read_bytes()).decode() if viz_path else None
                self.send_json(200 if result.success else 422, {
                    "success": result.success, "answer": result.answer, "error": result.error,
                    "intent": result.intent, "region": result.region, "operation": result.operation,
                    "target": result.target, "parameters": result.parameters,
                    "plan": result.plan_steps,
                    "stats": ({"changed_pixels": stats.changed_pixels, "total_pixels": stats.total_pixels,
                               "percentage": stats.changed_percentage, "area_m2": stats.changed_area_m2}
                              if stats else None),
                    "regions": {k: {"pixels": v.changed_pixels, "total": v.total_pixels,
                                    "percentage": v.changed_percentage}
                                for k, v in result.evidence.region_stats.items()},
                    "tool_runs": [
                        {
                            "tool": run.tool_name,
                            "parameters": run.parameters,
                            "success": run.success,
                            "error": run.error_message,
                        }
                        for run in result.evidence.tools_executed
                    ],
                    "metadata": result.evidence.metadata,
                    "inputs": {"before": before.metadata, "after": after.metadata,
                               "pixel_resolution_m": list(resolution) if resolution else None},
                    "threshold": result.evidence.threshold_used,
                    "visualization": viz,
                    "dimensions": {"width": int(before.pixels.shape[2]), "height": int(before.pixels.shape[1]), "bands": int(before.pixels.shape[0])},
                })
        except Exception as exc:  # noqa: BLE001 - return malformed upload and pipeline errors to the UI
            self.send_json(400, {"success": False, "error": str(exc)})


if __name__ == "__main__":
    address = ("127.0.0.1", int(os.environ.get("EARTHCHANGE_PORT", "8765")))
    print(f"EarthChange dashboard: http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, Handler).serve_forever()
