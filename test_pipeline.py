"""Fast unit checks for stable configuration and source formatting."""
import unittest

from rag_assistant.pipeline import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K, _format_source, _extractive_fallback


class PipelineConfigTests(unittest.TestCase):
    def test_resume_aligned_retrieval_settings(self):
        self.assertEqual(CHUNK_SIZE, 800)
        self.assertEqual(CHUNK_OVERLAP, 150)
        self.assertEqual(TOP_K, 4)

    def test_page_metadata_is_one_based(self):
        class Doc:
            page_content = "Evidence from page zero"
            metadata = {"source": "sample.pdf", "page": 0}
        self.assertEqual(_format_source(Doc())["page"], 1)

    def test_extractive_answer_cites_the_source(self):
        sources = [{"source": "sample.pdf", "page": 2, "text": "A grounded sentence."}]
        answer = _extractive_fallback("What?", sources)
        self.assertIn("A grounded sentence.", answer)
        self.assertIn("[Source: sample.pdf, p. 2]", answer)

    def test_empty_fallback_admits_missing_evidence(self):
        self.assertIn("could not find", _extractive_fallback("What?", []).lower())


if __name__ == "__main__":
    unittest.main()
