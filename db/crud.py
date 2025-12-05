from datetime import date
from typing import Optional, List, Dict, Any
from sqlmodel import select, Session
from db.config import get_session
from db.models import User, UserAuth, UserPost, UsersSavedVisuals, LabelCorrection


# User CRUD

def create_user(
    username: str,
    email: str,
    preferred_theme: str,
    phone_number: Optional[str] = None,
    session: Session = next(get_session()),  # Use the dependency to get a session
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


def remove_user(user_id: int, session: Session = next(get_session())) -> bool:
    user = session.exec(select(User).where(User.user_id == user_id)).first()
    if user:
        session.delete(user)
        session.commit()
        return True
    return False


def get_user_by_id(
    user_id: int, session: Session = next(get_session())
) -> Optional[User]:
    return session.exec(select(User).where(User.user_id == user_id)).first()


def get_all_users(session: Session = next(get_session())) -> List[User]:
    return session.exec(select(User)).all()


def update_user(user_id: int, data: dict, session: Session = next(get_session())) -> Optional[User]:
    user = session.exec(select(User).where(User.user_id == user_id)).first()
    if user:
        for key, value in data.items():
            setattr(user, key, value)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    return None


# UserAuth CRUD

def create_user_auth(
        password: str,
        token: str,
        user_id: int,
        session: Session = next(get_session())
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
        session: Session = next(get_session())
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


def get_user_post_by_id(id: int, session: Session = next(get_session())) -> Optional[UserPost]:
    return session.exec(select(UserPost).where(UserPost.id == id)).first()


def get_user_posts_by_user_id(user_id: int, session: Session = next(get_session())) -> List[UserPost]:
    return session.exec(select(UserPost).where(UserPost.user_id == user_id)).all()


def get_all_user_posts(session: Session = next(get_session())) -> List[UserPost]:
    return session.exec(select(UserPost)).all()


# UsersSavedVisuals CRUD

def create_saved_visual (
        saved_visual: str,
        type: str,
        user_id: int,
        session: Session = next(get_session())
) -> UsersSavedVisuals:
    new_saved_visual = UsersSavedVisuals(
        saved_visual=saved_visual,
        type=type,
        user_id=user_id,
        updated_at=date.today(),
    )

    session.add(new_saved_visual)
    session.commit()
    session.refresh(new_saved_visual)
    return new_saved_visual


def remove_saved_visual(id: int, session: Session = next(get_session())) -> bool:
    saved_visual = session.exec(select(UsersSavedVisuals).where(UsersSavedVisuals.id == id)).first()
    if saved_visual:
        session.delete(saved_visual)
        session.commit()
        return True
    return False


def get_saved_visual_by_id(id: int, session: Session = next(get_session())
) -> Optional[UsersSavedVisuals]:
    return session.exec(select(UsersSavedVisuals).where(UsersSavedVisuals.id == id)).first()


def get_all_saved_visuals(session: Session = next(get_session())) -> List[UsersSavedVisuals]:
    return session.exec(select(UsersSavedVisuals)).all()


def get_saved_visuals_by_user_id(
        user_id: int, session: Session = next(get_session())
) -> List[UsersSavedVisuals]:
    return session.exec(select(UsersSavedVisuals.user_id == user_id)).all()


def update_saved_visual(id: int, data: dict, session: Session = next(get_session())
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
        wrong_label: Optional[Dict[str, Any]] = None,
        session: Session = next(get_session())
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


def remove_label_correction(image_path: str, session: Session = next(get_session())) -> bool:
    label_correction = session.exec(
        select(LabelCorrection).where(LabelCorrection.image_path == image_path)).first()
    if label_correction:
        session.delete(label_correction)
        session.commit()
        return True
    return False


def get_label_correction_by_path(
    image_path: str, session: Session = next(get_session())
) -> Optional[LabelCorrection]:
    return session.exec(
        select(LabelCorrection).where(LabelCorrection.image_path == image_path)).first()


def get_all_label_corrections(session: Session = next(get_session())) -> List[LabelCorrection]:
    return session.exec(select(LabelCorrection)).all()


def get_label_corrections_by_user_id(
        user_id:int, session: Session = next(get_session())) -> List[LabelCorrection]:
    return session.exec(select(LabelCorrection).where(LabelCorrection.user_id == user_id)).all()