import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY") or ""

if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is not set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
GUEST_TOKEN_EXPIRE_MINUTES = 120

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthPrincipal:
    role: Literal["user", "guest"]
    user_id: Optional[int] = None
    guest_id: Optional[str] = None
    exp: Optional[int] = None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if "sub" in to_encode and to_encode["sub"] is not None:
        to_encode["sub"] = str(to_encode["sub"])

    if "role" not in to_encode:
        to_encode["role"] = "user"

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_guest_token(
    expires_delta: Optional[timedelta] = None,
) -> tuple[str, str, datetime]:
    """Create a stateless, short-lived JWT for a guest session (no DB account)."""
    guest_id = f"guest_{uuid.uuid4().hex[:12]}"
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=GUEST_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": guest_id,
        "role": "guest",
        "exp": expire,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token, guest_id, expire


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_user_from_token(token: str) -> Optional[int]:
    """Extract registered user_id from token (returns None for guest/invalid tokens)."""
    payload = verify_token(token)
    if payload is None:
        return None
    if payload.get("role") == "guest":
        return None
    sub = payload.get("sub")
    if sub is None:
        return None
    try:
        return int(sub)
    except (TypeError, ValueError):
        return None


def _principal_from_token(token: str) -> Optional[AuthPrincipal]:
    payload = verify_token(token)
    if payload is None:
        return None

    role = payload.get("role", "user")
    sub = payload.get("sub")
    exp = payload.get("exp")

    if role == "guest":
        if not isinstance(sub, str) or not sub.startswith("guest_"):
            return None
        return AuthPrincipal(role="guest", guest_id=sub, exp=exp)

    if role == "user":
        try:
            user_id = int(sub)
        except (TypeError, ValueError):
            return None
        return AuthPrincipal(role="user", user_id=user_id, exp=exp)

    return None


def get_current_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> AuthPrincipal:
    """Require a valid Bearer token (either registered user or guest)."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required or session expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    principal = _principal_from_token(credentials.credentials)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return principal


def get_optional_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[AuthPrincipal]:
    """Decode Bearer token if present; reject invalid/expired tokens with 401."""
    if credentials is None or not credentials.credentials:
        return None

    principal = _principal_from_token(credentials.credentials)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_registered_user(
    principal: AuthPrincipal = Depends(get_current_principal),
) -> AuthPrincipal:
    """Reject guest sessions with 403 Forbidden; allow only registered users."""
    if principal.role != "user" or principal.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires a registered account. Please sign in or create an account.",
        )
    return principal
