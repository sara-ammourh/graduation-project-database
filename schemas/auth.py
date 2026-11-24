from pydantic import BaseModel
from typing import Optional


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    phone_number: Optional[str] = None
    preferred_theme: str = "light"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str = "user"
    user_id: int
    username: str
    email: str
    expires_at: Optional[str] = None


class GuestTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str = "guest"
    guest_id: str
    expires_at: str


class SessionStatusResponse(BaseModel):
    role: str
    user_id: Optional[int] = None
    guest_id: Optional[str] = None
    expires_at: Optional[str] = None


class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    preferred_theme: str
    phone_number: Optional[str] = None
    created_at: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
