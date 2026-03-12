"""
Urdu Fake News Detection — FastAPI Backend
Serves the XLM-RoBERTa model hosted on Hugging Face,
stores results in Firestore, authenticates via Firebase.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # reads .env before anything else

from middleware.auth import init_firebase
from services.prediction import load_model
from routes.detection import router as detection_router


def get_cors_origins() -> list[str]:
    origins = os.getenv("CORS_ORIGINS", "")
    if not origins.strip():
        return ["*"]
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


def is_production() -> bool:
    return (
        os.getenv("RAILWAY_ENVIRONMENT") is not None
        or os.getenv("ENVIRONMENT", "").lower() == "production"
    )


def should_preload_model() -> bool:
    preload = os.getenv("PRELOAD_MODEL", "")
    if preload:
        return preload.lower() in {"1", "true", "yes", "on"}
    return not is_production()


# ── Startup / shutdown lifecycle ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ▸ STARTUP
    init_firebase()          # Firebase Admin SDK
    if should_preload_model():
        load_model()         # Download & cache HF model when enabled
    yield
    # ▸ SHUTDOWN (nothing to clean up)


app = FastAPI(
    title="Urdu Fake News Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — configurable for local/dev/prod environments ─────────
cors_origins = get_cors_origins()
allow_all_origins = cors_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register route groups ───────────────────────────────────────
app.include_router(detection_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Urdu Fake News Detection API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ── Run directly with: python main.py ───────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=not is_production(),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
