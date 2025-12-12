from typing import Optional, Dict, Any
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from db.config import get_session
from db import crud
from schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    PasswordChangeRequest,
)
from auth.utils import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()


# Authentication Routes


@router.post("/auth/register", response_model=TokenResponse)
def register(request: RegisterRequest, session: Session = Depends(get_session)):
    """Register a new user."""
    user = crud.register_user(
        session=session,
        username=request.username,
        email=request.email,
        password=request.password,
        preferred_theme=request.preferred_theme,
        phone_number=request.phone_number,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists or registration failed",
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_id}, expires_delta=access_token_expires
    )

    return TokenResponse(
        access_token=access_token,
        user_id=user.user_id,
        username=user.username,
        email=user.email,
    )


@router.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, session: Session = Depends(get_session)):
    """Login user with email and password."""
    user = crud.authenticate_user(
        email=request.email, password=request.password, session=session
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_id}, expires_delta=access_token_expires
    )

    return TokenResponse(
        access_token=access_token,
        user_id=user.user_id,
        username=user.username,
        email=user.email,
    )


@router.post("/auth/change-password")
def change_password(
    user_id: int,
    request: PasswordChangeRequest,
    session: Session = Depends(get_session),
):
    """Change user password."""
    success = crud.change_password(
        user_id=user_id,
        old_password=request.current_password,
        new_password=request.new_password,
        session=session,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid current password or user not found",
        )

    return {"message": "Password changed successfully"}


# User API Route

@router.post("/user")
def create_user(username: str,
                email: str,
                preferred_theme: str,
                phone_number: Optional[str],
                session: Session = Depends(get_session)):
    return crud.create_user(username=username,
                            email=email,
                            preferred_theme=preferred_theme,
                            phone_number=phone_number,
                            session=session)


@router.delete("/user/{user_id}")
def delete_user(user_id: int, session: Session = Depends(get_session)):
    return crud.remove_user(user_id=user_id, session=session)


@router.get("/user/{user_id}")
def get_user(user_id: int, session: Session = Depends(get_session)):
    return crud.get_user_by_id(user_id=user_id, session=session)


@router.get("/users")
def get_users(session: Session = Depends(get_session)):
    return crud.get_all_users(session=session)


@router.put("/user/{user_id}")
def update_user(user_id: int, data: dict, session: Session = Depends(get_session)):
    return crud.update_user(user_id=user_id, data=data, session=session)


# UserAuth API Route

@router.post("/user_auth")
def create_user_auth(password: str,
                     token: str,
                     user_id: int,
                     session: Session = Depends(get_session)):
    return crud.create_user_auth(password=password,
                                 token=token,
                                 user_id=user_id,
                                 session=session)


# @router.delete("/user_auth/{user_id}")


# @router.get("/user_auth")


# UserPost API Route
@router.post("/user_post")
def create_user_post(operation_type: str,
                     status: str,
                     user_id: int,
                     session: Session = Depends(get_session)):
    return crud.create_user_post(operation_type=operation_type,
                                 status=status,
                                 user_id=user_id,
                                 session=session)


@router.get("/user_post/{id}")
def get_user_post_by_id(id: int, session: Session = Depends(get_session)):
    return crud.get_user_post_by_id(id=id, session=session)


@router.get("/user_posts/user/{user_id}")
def get_user_posts_by_user_id(user_id: int, session: Session = Depends(get_session)):
    return crud.get_user_posts_by_user_id(user_id=user_id, session=session)


@router.get("/user_posts")
def get_all_user_posts(session: Session = Depends(get_session)):
    return crud.get_all_user_posts(session=session)


# UsersSavedVisuals API Route

@router.post("/saved_visual")
def create_saved_visual(saved_visual: Dict[str, Any],
                        type: str,
                        user_id: int,
                        session: Session = Depends(get_session)):
    return crud.create_saved_visual(saved_visual=saved_visual,
                               type=type,
                               user_id=user_id,
                               session=session)


@router.delete("/saved_visual/{id}")
def delete_saved_visual(id: int, session: Session = Depends(get_session)):
    return crud.remove_saved_visual(id=id, session=session)


@router.get("/saved_visual/{id}")
def get_saved_visual_by_id(id: int, session: Session = Depends(get_session)):
    return crud.get_saved_visual_by_id(id=id, session=session)


@router.get("/saved_visuals")
def get_all_saved_visuals(session: Session = Depends(get_session)):
    return crud.get_all_saved_visuals(session=session)


@router.get("/saved_visuals/user/{user_id}")
def get_saved_visuals_by_user_id(user_id: int, session: Session = Depends(get_session)):
    return crud.get_saved_visuals_by_user_id(user_id=user_id, session=session)


@router.put("/saved_visual/{id}")
def update_saved_visual(id: int, data: dict, session: Session = Depends(get_session)):
    return crud.update_saved_visual(id=id, data=data, session=session)


# LabelCorrection API Route

@router.post("/label_correction")
def create_label_correction(image_path: str,
                            data_structure_type: str,
                            correct_label: Dict[str, Any],
                            user_id: int,
                            wrong_label: Dict[str, Any],
                            session: Session = Depends(get_session)):
    return crud.create_label_correction(image_path=image_path,
                                        data_structure_type=data_structure_type,
                                        correct_label=correct_label,
                                        user_id=user_id,
                                        wrong_label=wrong_label,
                                        session=session)


@router.delete("/label_correction/{image_path}")
def delete_label_correction(image_path: str, session: Session = Depends(get_session)):
    return crud.remove_label_correction(image_path=image_path, session=session)


@router.get("/label_correction/{image_path}")
def get_label_correction_by_path(image_path: str, session: Session = Depends(get_session)):
    return crud.get_label_correction_by_path(image_path=image_path, session=session)


@router.get("/label_corrections")
def get_all_label_corrections(session: Session = Depends(get_session)):
    return crud.get_all_label_corrections(session=session)


@router.get("/label_corrections/user/{user_id}")
def get_label_corrections_by_user_id(user_id: int, session: Session = Depends(get_session)):
    return crud.get_label_corrections_by_user_id(user_id=user_id, session=session)