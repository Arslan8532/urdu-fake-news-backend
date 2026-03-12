"""
Auth Middleware
Verifies Firebase ID tokens sent from the React frontend.
Extracts the user's UID so every endpoint knows who is calling.
"""

import base64
import json
import os
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Request, HTTPException


def _load_firebase_credentials() -> credentials.Base | None:
    credentials_json = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
    if credentials_json:
        return credentials.Certificate(json.loads(credentials_json))

    credentials_base64 = os.getenv("FIREBASE_CREDENTIALS_BASE64", "").strip()
    if credentials_base64:
        decoded = base64.b64decode(credentials_base64).decode("utf-8")
        return credentials.Certificate(json.loads(decoded))

    cred_path = os.getenv("FIREBASE_CREDENTIALS", "firebase_admin_key.json")
    if os.path.exists(cred_path):
        return credentials.Certificate(cred_path)

    return None


def init_firebase():
    """Initialise Firebase Admin SDK (called once at startup)."""
    if firebase_admin._apps:
        return  # already initialised

    cred = _load_firebase_credentials()

    if cred is not None:
        firebase_admin.initialize_app(cred)
    else:
        # Falls back to Application Default Credentials (useful in Cloud Run / GCE)
        firebase_admin.initialize_app()

    print("[auth] Firebase Admin SDK initialised")


async def get_current_user(request: Request) -> str:
    """
    Dependency that extracts and verifies the Firebase ID token
    from the Authorization header.  Returns the user's UID.
    """
    header = request.headers.get("Authorization", "")

    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = header.split("Bearer ", 1)[1]

    try:
        decoded = auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return decoded["uid"]
