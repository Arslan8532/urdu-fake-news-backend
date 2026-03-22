"""
Prediction Service
Loads the XLM-RoBERTa model from Hugging Face and runs inference on Urdu text.
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ── Global references (loaded once at startup) ──────────────────
_tokenizer = None
_model = None
_device = None
_label_map = None


def load_model():
    """
    Download (or use cached) model from Hugging Face and keep it in memory.
    Called once when FastAPI starts up.
    """
    global _tokenizer, _model, _device, _label_map

    repo = os.getenv("MODEL_REPO", "MAJ853212/xlm_roberta_fake_news_detection")

    print(f"[model] Loading tokenizer from {repo} …")
    _tokenizer = AutoTokenizer.from_pretrained(repo)

    print(f"[model] Loading model from {repo} …")
    _model = AutoModelForSequenceClassification.from_pretrained(repo)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()

    # Build a human-readable label map from config.json → id2label
    # e.g. { 0: "Fake", 1: "Real" }
    if hasattr(_model.config, "id2label") and _model.config.id2label:
        _label_map = {int(k): v for k, v in _model.config.id2label.items()}
    else:
        # Fallback if config doesn't specify labels
        _label_map = {0: "Fake", 1: "Real"}

    print(f"[model] Ready on {_device}  |  labels: {_label_map}")


def predict(text: str) -> dict:
    """
    Run the model on a single Urdu text string.
    Returns: { "label": "Real"|"Fake", "confidence": 0‒100 }
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # ── Step 1: Tokenize ─────────────────────────────────────────
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    # ── Step 2: Forward pass (no gradient needed) ────────────────
    with torch.no_grad():
        outputs = _model(**inputs)

    # ── Step 3: Convert logits → probabilities ───────────────────
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_class = torch.argmax(probabilities).item()
    confidence = round(probabilities[predicted_class].item() * 100, 2)

    label = _label_map.get(predicted_class, "Unknown")

    return {"label": label, "confidence": confidence}
