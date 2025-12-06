from datetime import date
from typing import Optional, List, Dict, Any
from sqlmodel import Field, Relationship, SQLModel, Column, JSON


class UserAuth(SQLModel, table=True):
    __tablename__ = "user_auth"

    password: str = Field(primary_key=True)
    token: str
    token_created_at: date
    user_id: int = Field(foreign_key="users.user_id", primary_key=True)

    user: "User" = Relationship(back_populates="auth_details")


class UserPost(SQLModel, table=True):
    __tablename__ = "user_post"

    id: Optional[int] = Field(default=None, primary_key=True)
    operation_type: str = Field(max_length=30)
    created_at: date
    status: str = Field(max_length=15)
    user_id: int = Field(foreign_key="users.user_id")

    user: "User" = Relationship(back_populates="posts")


class UsersSavedVisuals(SQLModel, table=True):
    __tablename__ = "users_saved_visuals"

    id: Optional[int] = Field(default=None, primary_key=True)
    saved_visual: Dict[str, Any] = Field(sa_column=Column(JSON))
    type: str = Field(max_length=30)
    updated_at: date
    user_id: int = Field(foreign_key="users.user_id")

    user: "User" = Relationship(back_populates="saved_visuals")


class LabelCorrection(SQLModel, table=True):
    __tablename__ = "label_correction"

    image_path: str = Field(primary_key=True, max_length=256)
    data_structure_type: str = Field(max_length=30)
    wrong_label: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    correct_label: Dict[str, Any] = Field(sa_column=Column(JSON))
    created_at: date
    user_id: int = Field(foreign_key="users.user_id")

    user: "User" = Relationship(back_populates="label_corrections")


class User(SQLModel, table=True):
    __tablename__ = "users"

    user_id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(max_length=64, unique=True)
    email: str = Field(max_length=254)
    preferred_theme: str = Field(max_length=15)
    created_at: date
    phone_number: Optional[str] = Field(default=None, max_length=20)
    saved_vis_num: int = Field(default=0)

    auth_details: List["UserAuth"] = Relationship(back_populates="user")
    posts: List["UserPost"] = Relationship(back_populates="user")
    saved_visuals: List["UsersSavedVisuals"] = Relationship(back_populates="user")
    label_corrections: List["LabelCorrection"] = Relationship(back_populates="user")
