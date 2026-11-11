"""Vision API enrichment: one call → concise caption + structured metadata.

API must support OpenAI Responses with image input. Set VISION_API_BASE_URL,
VISION_API_KEY and VISION_MODEL; default is not a real public model name.
"""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import re
from pathlib import Path
from PIL import Image
from config import settings

SCENE_PROMPT = """You analyze a front-facing autonomous driving camera image.
Return ONLY a JSON object containing:
caption (concise 1-2 English sentences), weather (sunny/cloudy/rainy/foggy/snowy/unknown),
lighting (daytime/nighttime/dawn/dusk/tunnel/unknown), road_type
(urban/highway/residential/parking_lot/rural/unknown), scene_type
(intersection/straight_road/curve/crosswalk_area/traffic_jam/unknown),
objects (list of strings), risks (list of strings),
is_cloudy_candidate (0/1), is_daytime (0/1), is_nighttime (0/1).
Only describe visible evidence; do not invent uncertain details. Output valid JSON only."""
FIELDS = ["fileName", "caption", "weather", "lighting", "road_type", "scene_type",
          "objects", "risks", "is_cloudy_candidate", "is_daytime", "is_nighttime"]


def image_to_data_url(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(bytes(image_bytes))) as image:
        img = image.convert("RGB")
        img.thumbnail((1280, 720))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end < start:
            raise ValueError("Vision model did not return JSON")
        result = json.loads(text[start:end+1])
    if not isinstance(result, dict):
        raise ValueError("Vision JSON must be an object")
    if not str(result.get("caption", "")).strip():
        raise ValueError("Empty caption")
    return result


def extract_response_text(response) -> str:
    """Read an OpenAI Responses object, or a custom gateway's SSE text payload.

    Some Responses-compatible proxies return `event:/data:` lines even when
    stream=False. The tutorial encountered this variation, so keep both paths.
    """
    if not isinstance(response, str):
        direct = getattr(response, "output_text", None)
        if direct:
            return str(direct).strip()
        pieces = [part.text for item in getattr(response, "output", [])
                  for part in getattr(item, "content", [])
                  if getattr(part, "type", "") == "output_text"]
        if pieces:
            return "\n".join(pieces).strip()
        raise ValueError("Responses object has no output_text")
    content = response.strip()
    if not content.startswith(("event:", "data:")):
        return content
    for line in content.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "response.failed":
            raise RuntimeError("Vision gateway reported response.failed")
        if event.get("type") == "response.completed":
            result = event.get("response", {})
            pieces = [c.get("text", "") for item in result.get("output", [])
                      for c in item.get("content", []) if c.get("type") == "output_text"]
            if pieces:
                return "\n".join(pieces).strip()
    raise ValueError("No completed response found in SSE payload")


def analyze_scene(image_bytes: bytes) -> dict:
    if not settings.vision_key or not settings.vision_base_url or settings.vision_model == "your-vision-model":
        raise RuntimeError("Configure VISION_API_BASE_URL, VISION_API_KEY, VISION_MODEL")
    from openai import OpenAI
    client = OpenAI(api_key=settings.vision_key, base_url=settings.vision_base_url, timeout=120, max_retries=1)
    response = client.responses.create(
        model=settings.vision_model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": SCENE_PROMPT},
            {"type": "input_image", "image_url": image_to_data_url(image_bytes), "detail": "auto"},
        ]}], stream=False, max_output_tokens=600
    )
    text = extract_response_text(response)
    data = parse_json_response(text)
    out = {
        "caption": str(data["caption"]).strip(),
        "weather": str(data.get("weather", "unknown")),
        "lighting": str(data.get("lighting", "unknown")),
        "road_type": str(data.get("road_type", "unknown")),
        "scene_type": str(data.get("scene_type", "unknown")),
        "objects": data.get("objects", []),
        "risks": data.get("risks", []),
        "is_cloudy_candidate": int(data.get("is_cloudy_candidate", 0)),
        "is_daytime": int(data.get("is_daytime", 0)),
        "is_nighttime": int(data.get("is_nighttime", 0)),
    }
    if not isinstance(out["objects"], list) or not isinstance(out["risks"], list):
        raise ValueError("objects / risks should be lists")
    return out


def annotate_to_csv(records, path: Path) -> int:
    """Resume by fileName and persist each successful API result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            existing = {row["fileName"]: row for row in csv.DictReader(f)}
    done = 0
    for record in records:
        filename = str(record["fileName"])
        if filename in existing:
            continue
        analysis = analyze_scene(bytes(record["picture"]))
        row = {"fileName": filename, **analysis}
        row["objects"] = json.dumps(row["objects"], ensure_ascii=False)
        row["risks"] = json.dumps(row["risks"], ensure_ascii=False)
        existing[filename] = row
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing.values())
        done += 1
        print(f"{filename}: {analysis['caption']}")
    print("New captions:", done, "cached total:", len(existing))
    return done


def main() -> None:
    from .paimon_io import connect_catalog, read_table
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    catalog = connect_catalog()
    df = read_table(catalog.get_table(settings.raw_table))
    if df.empty:
        raise RuntimeError("Raw table empty; first run write-raw")
    annotate_to_csv(df.head(args.limit).to_dict("records"), settings.caption_csv)

if __name__ == "__main__":
    main()
