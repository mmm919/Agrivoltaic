"""
auth.py  –  SQLite-backed authentication for AgroVision AI
Endpoints:  POST /auth/signup,  POST /auth/login
"""

import hashlib, secrets, sqlite3, os, logging
from datetime import datetime, timezone
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException

log = logging.getLogger("auth")
router = APIRouter(prefix="/auth", tags=["auth"])

DB_PATH = os.path.join(os.path.dirname(__file__), "storage", "users.db")

# ── DB init ──────────────────────────────────────────────────────────────────
def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            email    TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            salt     TEXT NOT NULL,
            created  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()

# ── Models ───────────────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    email: str = ""

# ── Signup ───────────────────────────────────────────────────────────────────
@router.post("/signup", response_model=AuthResponse)
def signup(req: AuthRequest):
    email = req.email.strip().lower()
    password = req.password

    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email address.")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")

    conn = _get_db()
    existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "Email already registered.")

    salt = secrets.token_hex(16)
    hashed = _hash_pw(password, salt)
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO users (email, password, salt, created) VALUES (?, ?, ?, ?)",
        (email, hashed, salt, now)
    )
    conn.commit()
    conn.close()
    log.info("New user registered: %s", email)
    return AuthResponse(success=True, message="Account created successfully.", email=email)

# ── Login ────────────────────────────────────────────────────────────────────
@router.post("/login", response_model=AuthResponse)
def login(req: AuthRequest):
    email = req.email.strip().lower()
    password = req.password

    conn = _get_db()
    row = conn.execute("SELECT password, salt FROM users WHERE email=?", (email,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(401, "Email not found.")

    stored_hash, salt = row
    if _hash_pw(password, salt) != stored_hash:
        raise HTTPException(401, "Incorrect password.")

    log.info("User logged in: %s", email)
    return AuthResponse(success=True, message="Login successful.", email=email)
