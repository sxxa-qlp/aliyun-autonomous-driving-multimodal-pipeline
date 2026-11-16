"""Notebook publication checks: syntax, notebook shape, and no persisted outputs.

These checks run on CPU without CUDA, DLF, ModelScope or external network.
"""
from __future__ import annotations
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / 'notebooks/01_multimodal_data_processing.ipynb',
    ROOT / 'notebooks/02_multimodal_model_training_inference.ipynb',
)


class TestPublishedNotebooks(unittest.TestCase):
    def test_valid_metadata_and_python_syntax(self):
        for path in NOTEBOOKS:
            with self.subTest(file=path.name):
                notebook = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(notebook['nbformat'], 4)
                self.assertEqual(notebook['metadata']['kernelspec']['name'], 'python3')
                self.assertGreater(len(notebook['cells']), 15)
                for idx, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code':
                        ast.parse(''.join(cell['source']), filename=f'{path.name}:{idx}')
                        self.assertEqual(cell.get('outputs'), [])
                        self.assertIsNone(cell.get('execution_count'))

    def test_user_safety_and_navigation(self):
        for path in NOTEBOOKS:
            notebook = json.loads(path.read_text(encoding='utf-8'))
            sources = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
            self.assertIn('Restart Kernel', sources)
            self.assertIn('config import settings', sources)
            self.assertIn('README.md', sources)
            self.assertNotIn('sk-proj-', sources)
            self.assertNotIn('ALIBABA_CLOUD_ACCESS_KEY_SECRET=你的', sources)


if __name__ == '__main__':
    unittest.main()