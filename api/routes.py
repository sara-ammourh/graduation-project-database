from typing import Optional
from fastapi import APIRouter, Depends
from sqlmodel import Session
from db.config import get_session
from db import crud
from db.crud import get_all_users

router = APIRouter()


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
    return get_all_users(session=session)


@router.put("/user/{user_id}")
def update_user(user_id: int, data: dict, session: Session = Depends(get_session)):
    return crud.update_user(user_id=user_id, data=data, session=session)