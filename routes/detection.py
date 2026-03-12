"""
Detection Routes
All /api endpoints for news detection, history, and dashboard stats.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from middleware.auth import get_current_user
from services.prediction import predict
from services.gemini_service import verify_news
from services.firestore_service import (
    save_detection,
    get_history,
    get_detection,
    delete_detection,
    get_stats,
    get_profile,
    save_profile,
)

router = APIRouter(prefix="/api")


# ── Request / Response schemas ───────────────────────────────────
class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000000)


class GeminiVerification(BaseModel):
    verdict: str
    explanation: str
    sources: str


class DetectResponse(BaseModel):
    id: str
    newsText: str
    label: str
    confidence: float
    createdAt: str
    modelLabel: str
    geminiVerdict: str
    geminiExplanation: str
    geminiSources: str


def _combine_verdict(model_label: str, gemini_verdict: str) -> str:
    """
    Combine model prediction with Gemini verification.
    Logic:
      - Both Real  → Real
      - Both Fake  → Fake
      - Model=Real, Gemini=Fake → Fake  (web says it's fake)
      - Model=Fake, Gemini=Real → Fake  (model flagged it)
      - Gemini=Uncertain → trust model
    """
    if gemini_verdict == "Uncertain":
        return model_label
    if model_label == "Real" and gemini_verdict == "Real":
        return "Real"
    # Any other combo: at least one side says Fake
    return "Fake"


# ── POST /api/detect ─────────────────────────────────────────────
@router.post("/detect", response_model=DetectResponse)
async def detect_news(body: DetectRequest, uid: str = Depends(get_current_user)):
    """Run the DL model on the submitted text, verify with Gemini, then persist."""
    # Step 1: ML model prediction
    result = predict(body.text)
    model_label = result["label"]
    model_confidence = result["confidence"]

    # Step 2: Gemini web verification
    gemini = verify_news(body.text)
    gemini_verdict = gemini["verdict"]

    # Step 3: Combine verdicts
    final_label = _combine_verdict(model_label, gemini_verdict)

    # Step 4: Save to Firestore with all metadata
    doc = save_detection(
        user_id=uid,
        text=body.text,
        label=final_label,
        confidence=model_confidence,
        model_label=model_label,
        gemini_verdict=gemini_verdict,
        gemini_explanation=gemini["explanation"],
        gemini_sources=gemini["sources"],
    )

    return DetectResponse(
        id=doc["id"],
        newsText=doc["newsText"],
        label=doc["result"],
        confidence=doc["confidence"],
        createdAt=doc["createdAt"],
        modelLabel=doc["modelLabel"],
        geminiVerdict=doc["geminiVerdict"],
        geminiExplanation=doc["geminiExplanation"],
        geminiSources=doc["geminiSources"],
    )


# ── GET /api/history ─────────────────────────────────────────────
@router.get("/history")
async def list_history(uid: str = Depends(get_current_user)):
    """Return all detection records for the logged-in user."""
    return get_history(uid)


# ── GET /api/detect/{id} ────────────────────────────────────────
@router.get("/detect/{doc_id}")
async def get_single(doc_id: str, uid: str = Depends(get_current_user)):
    """Return a single detection record."""
    doc = get_detection(uid, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Detection not found")
    return doc


# ── GET /api/stats ───────────────────────────────────────────────
@router.get("/stats")
async def dashboard_stats(uid: str = Depends(get_current_user)):
    """Return dashboard statistics for the logged-in user."""
    return get_stats(uid)


# ── DELETE /api/history/{id} ─────────────────────────────────────
@router.delete("/history/{doc_id}")
async def remove_detection(doc_id: str, uid: str = Depends(get_current_user)):
    """Delete a detection record (only if it belongs to the user)."""
    deleted = delete_detection(uid, doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Detection not found")
    return {"detail": "Deleted"}


# ── GET /api/dashboard/stats ────────────────────────────────────
@router.get("/dashboard/stats")
async def dashboard_stats_v2(uid: str = Depends(get_current_user)):
    """Aggregated numbers for the dashboard cards."""
    return get_stats(uid)


# ── Profile ─────────────────────────────────────────────────────
class ProfileRequest(BaseModel):
    fullName: str = Field("", max_length=100)
    phone: str = Field("", max_length=20)
    photo: str = Field("", max_length=500000)


@router.get("/profile")
async def read_profile(uid: str = Depends(get_current_user)):
    """Get the user's saved profile data."""
    profile = get_profile(uid)
    return profile or {"fullName": "", "phone": "", "photo": ""}


@router.put("/profile")
async def update_profile(body: ProfileRequest, uid: str = Depends(get_current_user)):
    """Save or update user profile data."""
    data = {"fullName": body.fullName, "phone": body.phone, "photo": body.photo}
    save_profile(uid, data)
    return data
