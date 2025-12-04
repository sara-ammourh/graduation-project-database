from datetime import date
from typing import Optional, List
from sqlmodel import select, Session
from db.config import get_session
from db.models import User


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