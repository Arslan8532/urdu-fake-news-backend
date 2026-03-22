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


# ── Startup / shutdown lifecycle ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ▸ STARTUP
    init_firebase()          # Firebase Admin SDK
    load_model()             # Download & cache HF model
    yield
    # ▸ SHUTDOWN (
        # 
        # nothing to clean up)


app = FastAPI(
    title="Urdu Fake News Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow your React dev server ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
if __name__ == "__main__":
    import uvicorn
    # Use environment variables for host/port, default to Render.com values
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 10000)),
        reload=bool(os.getenv("RELOAD", "0") == "1"),
    )
