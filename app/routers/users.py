from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.database.db import get_db
from app.database.models import User
from app.services.user_service import UserCreate, create_user

router = APIRouter(prefix="/users", tags=["Users"])


class UserCreateRequest(BaseModel):
    name: str
    email: str
    subscription_plan: str
    monthly_spend: float
    region: str | None = None
    device_type: str | None = None


@router.post("/")
def create_user_endpoint(user: UserCreateRequest, db: Session = Depends(get_db)):
    try:
        return create_user(
            db,
            UserCreate(
                name=user.name,
                email=user.email,
                subscription_plan=user.subscription_plan,
                monthly_spend=user.monthly_spend,
                region=user.region,
                device_type=user.device_type,
            ),
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="User with this email already exists")


@router.get("/")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).limit(50).all()
    return users


@router.get("/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        return {"error": "User not found"}

    return user
