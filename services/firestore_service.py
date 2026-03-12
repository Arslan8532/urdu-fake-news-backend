"""
Firestore Service
CRUD operations for detection history stored in Cloud Firestore.
Each document is scoped to a Firebase user UID.
"""

from datetime import datetime, timezone
from firebase_admin import firestore

_db = None


def get_db():
    global _db
    if _db is None:
        _db = firestore.client()
    return _db


# ── Collection name ──────────────────────────────────────────────
COLLECTION = "detections"


def save_detection(
    user_id: str,
    text: str,
    label: str,
    confidence: float,
    model_label: str = "",
    gemini_verdict: str = "",
    gemini_explanation: str = "",
    gemini_sources: str = "",
) -> dict:
    """Save a new detection result and return the created document."""
    db = get_db()
    doc_ref = db.collection(COLLECTION).document()
    data = {
        "userId": user_id,
        "newsText": text,
        "result": label,
        "confidence": confidence,
        "modelLabel": model_label or label,
        "geminiVerdict": gemini_verdict,
        "geminiExplanation": gemini_explanation,
        "geminiSources": gemini_sources,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    doc_ref.set(data)
    return {"id": doc_ref.id, **data}


def get_history(user_id: str) -> list[dict]:
    """Return all detections for a user, newest first."""
    db = get_db()
    docs = (
        db.collection(COLLECTION)
        .where(filter=firestore.FieldFilter("userId", "==", user_id))
        .stream()
    )
    results = [{"id": doc.id, **doc.to_dict()} for doc in docs]
    results.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    return results


def get_detection(user_id: str, doc_id: str) -> dict | None:
    """Return a single detection if it belongs to the user."""
    db = get_db()
    doc = db.collection(COLLECTION).document(doc_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    if data.get("userId") != user_id:
        return None
    return {"id": doc.id, **data}


def delete_detection(user_id: str, doc_id: str) -> bool:
    """Delete a detection only if it belongs to the user. Returns True if deleted."""
    db = get_db()
    doc = db.collection(COLLECTION).document(doc_id).get()
    if not doc.exists:
        return False
    if doc.to_dict().get("userId") != user_id:
        return False
    db.collection(COLLECTION).document(doc_id).delete()
    return True


def get_stats(user_id: str) -> dict:
    """Compute dashboard stats for a user."""
    history = get_history(user_id)
    total = len(history)
    real_count = sum(1 for h in history if h["result"] == "Real")
    fake_count = sum(1 for h in history if h["result"] == "Fake")
    return {
        "totalDetections": total,
        "realCount": real_count,
        "fakeCount": fake_count,
    }


# ── Profile ──────────────────────────────────────────────────────
PROFILES_COLLECTION = "profiles"


def get_profile(user_id: str) -> dict | None:
    """Return the user's profile document."""
    db = get_db()
    doc = db.collection(PROFILES_COLLECTION).document(user_id).get()
    if not doc.exists:
        return None
    return doc.to_dict()


def save_profile(user_id: str, data: dict) -> dict:
    """Create or update the user's profile."""
    db = get_db()
    db.collection(PROFILES_COLLECTION).document(user_id).set(data, merge=True)
    return data
