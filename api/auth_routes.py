from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from auth.utils import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    AuthPrincipal,
    create_access_token,
    create_guest_token,
    get_current_principal,
    require_registered_user,
)
from db import crud
from db.config import get_session
from schemas.auth import (
    GuestTokenResponse,
    LoginRequest,
    PasswordChangeRequest,
    RegisterRequest,
    SessionStatusResponse,
    TokenResponse,
)
from schemas.label_correction import (
    LabelCorrectionRequest,
    LabelCorrectionResponse,
)
from schemas.saved_visual import SavedVisualRequest, SavedVisualResponse

router = APIRouter()


def _assert_same_user(principal: AuthPrincipal, target_user_id: int) -> None:
    if principal.user_id != target_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access or modify another user's data.",
        )


# Authentication Routes


@router.post("/auth/guest", response_model=GuestTokenResponse)
def create_guest_session():
    """Issue a stateless, time-limited guest token without creating a DB account."""
    access_token, guest_id, expires_at = create_guest_token()
    return GuestTokenResponse(
        access_token=access_token,
        guest_id=guest_id,
        expires_at=expires_at.isoformat(),
    )


@router.get("/auth/session", response_model=SessionStatusResponse)
def get_session_status(
    principal: AuthPrincipal = Depends(get_current_principal),
):
    """Validate the current bearer token (registered user or guest)."""
    expires_iso = (
        datetime.fromtimestamp(principal.exp, tz=timezone.utc).isoformat()
        if principal.exp is not None
        else None
    )
    return SessionStatusResponse(
        role=principal.role,
        user_id=principal.user_id,
        guest_id=principal.guest_id,
        expires_at=expires_iso,
    )


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

    if user.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is missing",
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now(timezone.utc) + access_token_expires
    access_token = create_access_token(
        data={"sub": str(user.user_id), "role": "user"},
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=access_token,
        role="user",
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        expires_at=expires_at.isoformat(),
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

    if user.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is missing",
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now(timezone.utc) + access_token_expires
    access_token = create_access_token(
        data={"sub": str(user.user_id), "role": "user"},
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=access_token,
        role="user",
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        expires_at=expires_at.isoformat(),
    )


@router.post("/auth/change-password")
def change_password(
    user_id: int,
    request: PasswordChangeRequest,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Change user password (registered users only)."""
    _assert_same_user(principal, user_id)
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


# User API Route (Restricted to Registered Users)


@router.post("/user")
def create_user(
    username: str,
    email: str,
    preferred_theme: str,
    phone_number: Optional[str],
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    return crud.create_user(
        username=username,
        email=email,
        preferred_theme=preferred_theme,
        phone_number=phone_number,
        session=session,
    )


@router.delete("/user/{user_id}")
def delete_user(
    user_id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    _assert_same_user(principal, user_id)
    return crud.remove_user(user_id=user_id, session=session)


@router.get("/user/{user_id}")
def get_user(
    user_id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    _assert_same_user(principal, user_id)
    return crud.get_user_by_id(user_id=user_id, session=session)


@router.get("/users")
def get_users(
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    return crud.get_all_users(session=session)


@router.put("/user/{user_id}")
def update_user(
    user_id: int,
    data: dict,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    _assert_same_user(principal, user_id)
    return crud.update_user(user_id=user_id, data=data, session=session)


# UserAuth API Route


@router.post("/user_auth")
def create_user_auth(
    password: str,
    token: str,
    user_id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    _assert_same_user(principal, user_id)
    return crud.create_user_auth(
        password=password, token=token, user_id=user_id, session=session
    )


# UserPost API Route (Restricted to Registered Users)


@router.post("/user_post")
def create_user_post(
    operation_type: str,
    status: str,
    user_id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    _assert_same_user(principal, user_id)
    return crud.create_user_post(
        operation_type=operation_type, status=status, user_id=user_id, session=session
    )


@router.get("/user_post/{id}")
def get_user_post_by_id(
    id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    post = crud.get_user_post_by_id(id=id, session=session)
    if post and post.user_id != principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access another user's history.",
        )
    return post


@router.get("/user_posts/user/{user_id}")
def get_user_posts_by_user_id(
    user_id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    _assert_same_user(principal, user_id)
    return crud.get_user_posts_by_user_id(user_id=user_id, session=session)


@router.get("/user_posts")
def get_all_user_posts(
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    return crud.get_user_posts_by_user_id(
        user_id=principal.user_id, session=session  # type: ignore[arg-type]
    )


# UsersSavedVisuals API Route (Restricted to Registered Users)


@router.post("/saved_visual", response_model=SavedVisualResponse)
def create_saved_visual(
    user_id: int,
    request: SavedVisualRequest,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Create and save a new visualization (e.g., graph) for a registered user."""
    _assert_same_user(principal, user_id)
    saved_visual = crud.create_saved_visual(
        saved_visual=request.saved_visual,
        type=request.type,
        user_id=user_id,
        session=session,
    )

    if not saved_visual:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not save visualization. User may have reached max saved visuals (5).",
        )

    return saved_visual


@router.delete("/saved_visual/{id}")
def delete_saved_visual(
    id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Delete a saved visualization by ID (owner only)."""
    existing = crud.get_saved_visual_by_id(id=id, session=session)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved visualization not found",
        )
    _assert_same_user(principal, existing.user_id)

    success = crud.remove_saved_visual(id=id, session=session)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved visualization not found",
        )

    return {"message": "Visualization deleted successfully"}


@router.get("/saved_visual/{id}", response_model=SavedVisualResponse)
def get_saved_visual_by_id(
    id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Get a specific saved visualization by ID (owner only)."""
    saved_visual = crud.get_saved_visual_by_id(id=id, session=session)

    if not saved_visual:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved visualization not found",
        )
    _assert_same_user(principal, saved_visual.user_id)

    return saved_visual


@router.get("/saved_visuals", response_model=List[SavedVisualResponse])
def get_all_saved_visuals(
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Get all saved visualizations for the authenticated user."""
    saved_visuals = crud.get_saved_visuals_by_user_id(
        user_id=principal.user_id, session=session  # type: ignore[arg-type]
    )
    return saved_visuals or []


@router.get("/saved_visuals/user/{user_id}", response_model=List[SavedVisualResponse])
def get_saved_visuals_by_user_id(
    user_id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Get all saved visualizations for a specific registered user."""
    _assert_same_user(principal, user_id)
    saved_visuals = crud.get_saved_visuals_by_user_id(user_id=user_id, session=session)

    if not saved_visuals:
        return []

    return saved_visuals


@router.put("/saved_visual/{id}", response_model=SavedVisualResponse)
def update_saved_visual(
    id: int,
    request: SavedVisualRequest,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Update an existing saved visualization (owner only)."""
    existing = crud.get_saved_visual_by_id(id=id, session=session)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved visualization not found",
        )
    _assert_same_user(principal, existing.user_id)

    updated_visual = crud.update_saved_visual(
        id=id,
        data=request.model_dump(),
        session=session,
    )

    if not updated_visual:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved visualization not found",
        )

    return updated_visual


# LabelCorrection API Route (Restricted to Registered Users)


@router.post("/label_correction", response_model=LabelCorrectionResponse)
def create_label_correction(
    user_id: int,
    request: LabelCorrectionRequest,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Save label corrections for a graph image (registered users only)."""
    _assert_same_user(principal, user_id)
    correction = crud.create_label_correction(
        image_path=request.image_path,
        data_structure_type=request.data_structure_type,
        wrong_label=request.predicted_labels,  # pyright: ignore
        correct_label=request.corrections,  # pyright: ignore
        user_id=user_id,
        session=session,
    )

    if not correction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to save label corrections",
        )

    return correction


@router.delete("/label_correction/{image_path}")
def delete_label_correction(
    image_path: str,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Delete label corrections for an image (owner only)."""
    existing = crud.get_label_correction_by_path(
        image_path=image_path, session=session
    )
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Label correction not found",
        )
    _assert_same_user(principal, existing.user_id)

    success = crud.remove_label_correction(image_path=image_path, session=session)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Label correction not found",
        )

    return {"message": "Label correction deleted successfully"}


@router.get("/label_correction/{image_path}", response_model=LabelCorrectionResponse)
def get_label_correction_by_path(
    image_path: str,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Get label corrections for a specific image (owner only)."""
    correction = crud.get_label_correction_by_path(
        image_path=image_path, session=session
    )

    if not correction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Label correction not found",
        )
    _assert_same_user(principal, correction.user_id)

    return correction


@router.get("/label_corrections", response_model=List[LabelCorrectionResponse])
def get_all_label_corrections(
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    """Get label corrections for the authenticated user."""
    return crud.get_label_corrections_by_user_id(
        user_id=principal.user_id, session=session  # type: ignore[arg-type]
    )


@router.get(
    "/label_corrections/user/{user_id}", response_model=List[LabelCorrectionResponse]
)
def get_label_corrections_by_user_id(
    user_id: int,
    principal: AuthPrincipal = Depends(require_registered_user),
    session: Session = Depends(get_session),
):
    _assert_same_user(principal, user_id)
    return crud.get_label_corrections_by_user_id(user_id=user_id, session=session)
