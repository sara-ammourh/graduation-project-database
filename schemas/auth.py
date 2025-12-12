from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


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
    user_id: int
    username: str
    email: str


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
