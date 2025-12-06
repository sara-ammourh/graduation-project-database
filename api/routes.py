from typing import Optional
from fastapi import APIRouter, Depends
from sqlmodel import Session
from db.config import get_session
from db import crud
from db.crud import get_all_users, get_user_post_by_id

router = APIRouter()

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
    return get_all_users(session=session)


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
    return get_user_post_by_id(id=id, session=session)


@router.get("/user_posts/{user_id}")
def get_user_posts_by_user_id(user_id: int, session: Session = Depends(get_session)):
    return get_user_posts_by_user_id(user_id=user_id, session=session)


@router.get("/user_posts")
def get_all_user_posts(session: Session = Depends(get_session)):
    return get_all_user_posts(session=session)