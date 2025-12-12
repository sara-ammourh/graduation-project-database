from datetime import date
from typing import Optional, List, Dict, Any

from sqlalchemy import update
from sqlmodel import select, Session

from auth.utils import hash_password, verify_password
from db.config import get_session
from db.models import User, UserAuth, UserPost, UsersSavedVisuals, LabelCorrection


# User CRUD

def create_user(
    session: Session,
    username: str,
    email: str,
    preferred_theme: str,
    phone_number: Optional[str] = None,
) -> User:
    new_user = User(
        username=username,
        email=email,
        preferred_theme=preferred_theme,
        created_at=date.today(),
        phone_number=phone_number,
    )

    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user


def remove_user(user_id: int, session: Session) -> bool:
    user = session.exec(select(User).where(User.user_id == user_id)).first()
    if user:
        session.delete(user)
        session.commit()
        return True
    return False


def get_user_by_id(user_id: int, session: Session) -> Optional[User]:
    return session.exec(select(User).where(User.user_id == user_id)).first()


def get_user_by_email(email: str, session: Session) -> Optional[User]:
    return session.exec(select(User).where(User.email == email)).first()


def get_user_by_username(username: str, session: Session) -> Optional[User]:
    return session.exec(select(User).where(User.username == username)).first()


def get_all_users(session: Session) -> List[User]:
    return session.exec(select(User)).all()


def update_user(user_id: int, data: dict, session: Session) -> Optional[User]:
    user = session.exec(select(User).where(User.user_id == user_id)).first()
    if user:
        for key, value in data.items():
            setattr(user, key, value)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    return None


# Authentication CRUD

def register_user(
    session: Session,
    username: str,
    email: str,
    password: str,
    preferred_theme: str = "light",
    phone_number: Optional[str] = None,
) -> Optional[User]:
    """Register a new user with authentication."""
    # Check if user already exists
    if get_user_by_email(email, session) or get_user_by_username(username, session):
        return None

    # Create user
    new_user = create_user(
        session=session,
        username=username,
        email=email,
        preferred_theme=preferred_theme,
        phone_number=phone_number,
    )

    # Create auth record with hashed password
    hashed_pwd = hash_password(password)
    user_auth = UserAuth(
        user_id=new_user.user_id,
        hashed_password=hashed_pwd,
    )
    session.add(user_auth)
    session.commit()

    return new_user


def authenticate_user(email: str, password: str, session: Session) -> Optional[User]:
    """Authenticate user by email and password."""
    user = get_user_by_email(email, session)
    if not user:
        return None

    user_auth = session.exec(
        select(UserAuth).where(UserAuth.user_id == user.user_id)
    ).first()
    if not user_auth:
        return None

    if not verify_password(password, user_auth.hashed_password):
        return None

    return user


def change_password(
    user_id: int, old_password: str, new_password: str, session: Session
) -> bool:
    """Change user password."""
    user_auth = session.exec(
        select(UserAuth).where(UserAuth.user_id == user_id)
    ).first()
    if not user_auth:
        return False

    if not verify_password(old_password, user_auth.hashed_password):
        return False

    user_auth.hashed_password = hash_password(new_password)
    session.add(user_auth)
    session.commit()
    return True


# UserAuth CRUD

def create_user_auth(
        password: str,
        token: str,
        user_id: int,
        session: Session
) -> UserAuth:
    new_user_auth = UserAuth(
        user_id=user_id,
        password=password,
        token=token,
        token_created_at=date.today(),
    )

    session.add(new_user_auth)
    session.commit()
    session.refresh(new_user_auth)
    return new_user_auth


def remove_user_auth(user_id: int, session: Session = next(get_session())) -> bool:
    user_auth = session.exec(select(UserAuth).where(UserAuth.user_id == user_id)).first()
    if user_auth:
        session.delete(user_auth)
        session.commit()
        return True
    return False


def get_user_auth_by_id(
    user_id: int, session: Session = next(get_session())
) -> Optional[UserAuth]:
    return session.exec(select(UserAuth).where(UserAuth.user_id == user_id)).first()


def update_user_auth(user_id: int, data: dict, session: Session = next(get_session())) -> Optional[UserAuth]:
    user_auth = session.exec(select(UserAuth).where(UserAuth.user_id == user_id)).first()
    if user_auth:
        for key, value in data.items():
            setattr(user_auth, key, value)
        session.add(user_auth)
        session.commit()
        session.refresh(user_auth)
        return user_auth
    return None


# UserPost CRUD

def create_user_post(
        operation_type: str,
        status: str,
        user_id: int,
        session: Session
) -> UserPost:
    new_user_post = UserPost(
        operation_type=operation_type,
        created_at=date.today(),
        status=status,
        user_id=user_id,
    )

    session.add(new_user_post)
    session.commit()
    session.refresh(new_user_post)
    return new_user_post


def get_user_post_by_id(id: int, session: Session) -> Optional[UserPost]:
    return session.exec(select(UserPost).where(UserPost.id == id)).first()


def get_user_posts_by_user_id(user_id: int, session: Session) -> List[UserPost]:
    return session.exec(select(UserPost).where(UserPost.user_id == user_id)).all()


def get_all_user_posts(session: Session) -> List[UserPost]:
    return session.exec(select(UserPost)).all()


# UsersSavedVisuals CRUD

def create_saved_visual (
        saved_visual: Dict[str, Any],
        type: str,
        user_id: int,
        session: Session
) -> Optional[UsersSavedVisuals]:
    if session.exec(select(User.saved_vis_num).where(User.user_id == user_id)).first() >= 5:
        print("\nUser Saved Visual Slots are full")
        return None
    new_saved_visual = UsersSavedVisuals(
        saved_visual=saved_visual,
        type=type,
        user_id=user_id,
        updated_at=date.today(),
    )

    session.add(new_saved_visual)
    session.exec(update(User.saved_vis_num)
                  .where(User.user_id == user_id)
                  .values(saved_vis_num=User.saved_vis_num + 1)
                  .execution_options(synchronize_sessions='fetch'))
    session.commit()
    session.refresh(new_saved_visual)
    return new_saved_visual


def remove_saved_visual(id: int, session: Session) -> bool:
    saved_visual = session.exec(select(UsersSavedVisuals).where(UsersSavedVisuals.id == id)).first()
    if saved_visual:
        session.delete(saved_visual)
        session.commit()
        return True
    return False


def get_saved_visual_by_id(id: int, session: Session
) -> Optional[UsersSavedVisuals]:
    return session.exec(select(UsersSavedVisuals).where(UsersSavedVisuals.id == id)).first()


def get_all_saved_visuals(session: Session) -> List[UsersSavedVisuals]:
    return session.exec(select(UsersSavedVisuals)).all()


def get_saved_visuals_by_user_id(user_id: int, session: Session
) -> List[UsersSavedVisuals]:
    return session.exec(select(UsersSavedVisuals.user_id == user_id)).all()


def update_saved_visual(id: int, data: dict, session: Session
                        ) -> Optional[UsersSavedVisuals]:
    saved_visual = session.exec(select(UsersSavedVisuals).where(UsersSavedVisuals.id == id)).first()
    if saved_visual:
        for key, value in data.items():
            setattr(saved_visual, key, value)
        setattr(saved_visual, 'updated_at', date.today())
        session.add(saved_visual)
        session.commit()
        session.refresh(saved_visual)
        return saved_visual
    return None


# LabelCorrection CRUD

def create_label_correction(
        image_path: str,
        data_structure_type: str,
        correct_label: Dict[str, Any],
        user_id: int,
        session: Session,
        wrong_label: Optional[Dict[str, Any]] = None
) -> LabelCorrection:
    new_label_correction = LabelCorrection(
        image_path=image_path,
        data_structure_type=data_structure_type,
        wrong_label=wrong_label,
        correct_label=correct_label,
        created_at=date.today(),
        user_id=user_id,
        session=session,
    )

    session.add(new_label_correction)
    session.commit()
    session.refresh(new_label_correction)
    return new_label_correction


def remove_label_correction(image_path: str, session: Session) -> bool:
    label_correction = session.exec(
        select(LabelCorrection).where(LabelCorrection.image_path == image_path)).first()
    if label_correction:
        session.delete(label_correction)
        session.commit()
        return True
    return False


def get_label_correction_by_path(image_path: str, session: Session
) -> Optional[LabelCorrection]:
    return session.exec(
        select(LabelCorrection).where(LabelCorrection.image_path == image_path)).first()


def get_all_label_corrections(session: Session) -> List[LabelCorrection]:
    return session.exec(select(LabelCorrection)).all()


def get_label_corrections_by_user_id(
        user_id:int, session: Session) -> List[LabelCorrection]:
    return session.exec(select(LabelCorrection).where(LabelCorrection.user_id == user_id)).all()