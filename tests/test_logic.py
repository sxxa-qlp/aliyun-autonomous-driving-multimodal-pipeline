"""CPU-only tests. No DLF/API/GPU/network calls."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

from multimodal_data_processing.caption_metadata import parse_json_response, extract_response_text
from multimodal_data_processing.enrichment import make_enriched_records
from multimodal_data_processing.retrieval import cosine_topk
from multimodal_training_inference.prepare_dataset import (
    select_valid_rows, split_records, write_dataset, load_records, dataset_statistics,
)


def fake_df(n=20):
    # Tiny valid JPEG images with distinct bytes, no external assets.
    import io
    out = []
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (i, 0, 0)).save(buf, "JPEG")
        out.append({"id": f"id-{i:02d}", "fileName": f"image-{i}.jpg", "picture": buf.getvalue(),
                    "caption": f"A test street scene number {i}", "dataset_tag": 1})
    return pd.DataFrame(out)


class TestFlow(unittest.TestCase):
    def test_split_is_order_independent_and_disjoint(self):
        df = select_valid_rows(fake_df())
        a, b = split_records(df, 0.05, 20260520)
        a2, b2 = split_records(df.sample(frac=1, random_state=3), 0.05, 20260520)
        self.assertEqual((len(a), len(b)), (19, 1))
        self.assertEqual(list(a.id), list(a2.id))
        self.assertEqual(list(b.id), list(b2.id))
        self.assertFalse(set(a.id) & set(b.id))

    def test_jsonl_and_manifest(self):
        train, val = split_records(fake_df(), 0.05, 20260520)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_dataset(train, val, base, 20260520)
            self.assertEqual(len(load_records(base / "train.jsonl")), 19)
            self.assertEqual(len(load_records(base / "validation.jsonl")), 1)
            write_dataset(train, val, base, 20260520)  # idempotent
            with self.assertRaises(RuntimeError):
                write_dataset(train, val, base, 99)

    def test_stats_and_bad_data(self):
        df = fake_df(3)
        self.assertEqual(dataset_statistics(df)["samples"], 3)
        df.loc[0, "caption"] = ""
        self.assertEqual(len(select_valid_rows(df)), 2)

    def test_enriched_schema_content(self):
        df = fake_df(2)[["fileName", "picture"]]
        labels = [{"fileName": f"image-{i}.jpg", "caption": "Cloudy road", "weather": "cloudy",
                   "lighting": "daytime", "road_type": "urban", "scene_type": "intersection",
                   "objects": '["car"]', "risks": '["crosswalk"]'} for i in range(2)]
        rows = make_enriched_records(df, labels)
        self.assertEqual(rows[0]["scene_type"], "daytime_cloudy")
        self.assertEqual(rows[0]["dataset_tag"], 1)
        self.assertIsNone(rows[0]["vector"])
        self.assertEqual(rows[0]["metadata"].count("crosswalk"), 1)

    def test_json_response_and_vector_search(self):
        result = parse_json_response('```json\n{"caption":"Cloudy street."}\n```')
        self.assertEqual(result["caption"], "Cloudy street.")
        indices = cosine_topk(np.array([[1,0], [0,1], [.8,.6]], dtype=float),
                              np.array([1,0], dtype=float), k=2, exclude_index=0)
        self.assertEqual([x[0] for x in indices], [2, 1])

    def test_custom_gateway_sse_extraction(self):
        import json
        payload = {"type": "response.completed", "response": {"output": [
            {"content": [{"type": "output_text", "text": '{"caption":"cloudy road"}'}]}]}}
        text = "event: response.completed\ndata: " + json.dumps(payload) + "\n"
        self.assertEqual(parse_json_response(extract_response_text(text))["caption"], "cloudy road")

    def test_no_secrets_required_on_import(self):
        import config
        self.assertTrue(hasattr(config.settings, "catalog_options"))
        with self.assertRaises(RuntimeError):
            config.Settings(access_key_id="", access_key_secret="").catalog_options()


if __name__ == "__main__":
    unittest.main()
