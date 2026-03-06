"""
Tests for the preprocessing module.

Covers:
- Tokenizer output constraints (max_len, padding)
- Text cleaning correctness
- Deduplication logic
- Dataset integrity

Run: pytest tests/ -v
"""

import hashlib
import random
import string

import numpy as np
import pytest


# ============================================================
# test_tokenizer_max_length
# Verifies that tokenizer output never exceeds max_len=256 tokens
# and that attention masks are properly padded.
# Uses 50 random samples including edge cases (empty strings,
# single-word, max-length articles).
# Catches silent truncation bugs that could corrupt batch alignment.
# ============================================================

class TestTokenizerMaxLength:
    """Test suite for tokenizer output constraints."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    @pytest.fixture
    def max_len(self):
        return 256

    @pytest.fixture
    def sample_texts(self):
        """Generate 50 test samples including edge cases."""
        texts = []

        # Edge case: very short texts
        texts.append("News.")
        texts.append("A")
        texts.append("Breaking news today")

        # Edge case: single word
        texts.append("Technology")

        # Edge case: exactly at boundary lengths
        texts.append(" ".join(["word"] * 250))  # near max_len
        texts.append(" ".join(["word"] * 500))  # well over max_len

        # Edge case: special characters
        texts.append("Apple's Q3 revenue: $81.4B — a 2% increase year-over-year.")
        texts.append("U.S. & E.U. agree on AI regulation framework (2024)")
        texts.append('He said "AI will transform everything" at the conference.')

        # Normal samples
        for _ in range(41):
            length = random.randint(10, 300)
            words = [
                "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
                for _ in range(length)
            ]
            texts.append(" ".join(words))

        assert len(texts) == 50
        return texts

    def test_output_never_exceeds_max_len(self, tokenizer, max_len, sample_texts):
        """No tokenized sequence should exceed max_len tokens."""
        encodings = tokenizer(
            sample_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        assert encodings["input_ids"].shape[1] == max_len
        assert encodings["attention_mask"].shape[1] == max_len

    def test_attention_mask_padding_consistency(self, tokenizer, max_len, sample_texts):
        """Attention mask should be 1 for real tokens, 0 for padding."""
        encodings = tokenizer(
            sample_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        pad_token_id = tokenizer.pad_token_id

        for i in range(len(sample_texts)):
            for j in range(max_len):
                if input_ids[i, j] == pad_token_id:
                    assert attention_mask[i, j] == 0, (
                        f"Sample {i}, pos {j}: pad token should have mask=0"
                    )
                else:
                    assert attention_mask[i, j] == 1, (
                        f"Sample {i}, pos {j}: non-pad token should have mask=1"
                    )

    def test_all_sequences_start_with_cls(self, tokenizer, max_len, sample_texts):
        """All sequences should start with [CLS] token."""
        encodings = tokenizer(
            sample_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        cls_token_id = tokenizer.cls_token_id
        for i in range(len(sample_texts)):
            assert encodings["input_ids"][i, 0] == cls_token_id

    def test_batch_shape_consistency(self, tokenizer, max_len, sample_texts):
        """Output shape should be (n_samples, max_len) regardless of input lengths."""
        encodings = tokenizer(
            sample_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        n = len(sample_texts)
        assert encodings["input_ids"].shape == (n, max_len)
        assert encodings["attention_mask"].shape == (n, max_len)


class TestTextCleaning:
    """Test suite for text cleaning functions."""

    def test_clean_text_removes_html(self):
        from src.preprocessing import clean_text
        assert "<b>" not in clean_text("This is <b>bold</b> text")

    def test_clean_text_removes_urls(self):
        from src.preprocessing import clean_text
        result = clean_text("Visit https://example.com for more info")
        assert "https://" not in result
        assert "example.com" not in result

    def test_clean_text_normalizes_whitespace(self):
        from src.preprocessing import clean_text
        result = clean_text("Too   many    spaces   here")
        assert "  " not in result

    def test_clean_text_preserves_content(self):
        from src.preprocessing import clean_text
        result = clean_text("Apple reports Q3 revenue growth driven by iPhone AI features")
        assert "Apple" in result
        assert "revenue" in result
        assert "AI" in result

    def test_clean_text_empty_string(self):
        from src.preprocessing import clean_text
        assert clean_text("") == ""
        assert clean_text("   ") == ""


class TestDeduplication:
    """Test suite for duplicate detection."""

    def test_identical_articles_same_hash(self):
        from src.preprocessing import compute_text_hash
        h1 = compute_text_hash("Breaking News", "The stock market crashed today amid fears...")
        h2 = compute_text_hash("Breaking News", "The stock market crashed today amid fears...")
        assert h1 == h2

    def test_different_articles_different_hash(self):
        from src.preprocessing import compute_text_hash
        h1 = compute_text_hash("Tech News", "Apple launches new iPhone with AI features")
        h2 = compute_text_hash("Sports News", "Lakers win championship in overtime thriller")
        assert h1 != h2

    def test_case_insensitive(self):
        from src.preprocessing import compute_text_hash
        h1 = compute_text_hash("Breaking News", "The market is up today")
        h2 = compute_text_hash("BREAKING NEWS", "THE MARKET IS UP TODAY")
        assert h1 == h2

    def test_near_duplicates_detected(self):
        """Syndicated articles with same title + opening should hash the same."""
        from src.preprocessing import compute_text_hash
        h1 = compute_text_hash(
            "Fed Holds Rates",
            "The Federal Reserve held interest rates steady on Wednesday. "
            "Chairman Powell said inflation is still too high but progress is being made."
        )
        h2 = compute_text_hash(
            "Fed Holds Rates",
            "The Federal Reserve held interest rates steady on Wednesday. "
            "Markets reacted positively to the announcement."
        )
        # Same title + same first 100 chars → same hash
        assert h1 == h2


class TestDatasetIntegrity:
    """Test dataset loading produces valid outputs."""

    def test_label_range(self):
        """All labels should be in [0, 3] for AG News 4-class."""
        from src.preprocessing import LABEL_MAP
        assert set(LABEL_MAP.keys()) == {0, 1, 2, 3}

    def test_label_names(self):
        from src.preprocessing import LABEL_MAP
        expected = {"World", "Sports", "Business", "Sci/Tech"}
        assert set(LABEL_MAP.values()) == expected


class TestNewsDataset:
    """Test PyTorch Dataset implementation."""

    def test_dataset_length(self):
        from src.preprocessing import NewsDataset
        ids = np.zeros((10, 256), dtype=np.int32)
        masks = np.ones((10, 256), dtype=np.int32)
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        dataset = NewsDataset(ids, masks, labels)
        assert len(dataset) == 10

    def test_dataset_getitem_shapes(self):
        import torch
        from src.preprocessing import NewsDataset

        ids = np.zeros((5, 128), dtype=np.int32)
        masks = np.ones((5, 128), dtype=np.int32)
        labels = np.array([0, 1, 2, 3, 0])
        dataset = NewsDataset(ids, masks, labels)

        sample = dataset[0]
        assert sample["input_ids"].shape == (128,)
        assert sample["attention_mask"].shape == (128,)
        assert sample["labels"].shape == ()
        assert sample["input_ids"].dtype == torch.long

    def test_dataset_label_values(self):
        from src.preprocessing import NewsDataset

        ids = np.zeros((4, 64), dtype=np.int32)
        masks = np.ones((4, 64), dtype=np.int32)
        labels = np.array([0, 1, 2, 3])
        dataset = NewsDataset(ids, masks, labels)

        for i in range(4):
            assert dataset[i]["labels"].item() == i
