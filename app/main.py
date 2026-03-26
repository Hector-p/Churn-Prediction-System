from fastapi import FastAPI
from app.database.db import engine, Base
from app.routers.users import router as users_router
from app.routers.churn import router as churn_router
from app.routers.monitoring import router as monitoring_router
from app.routers.feature_store import router as feature_store_router


app = FastAPI(title="AI Customer Intelligence Platform")

Base.metadata.create_all(bind=engine)

app.include_router(monitoring_router)
app.include_router(users_router)
app.include_router(churn_router)
app.include_router(feature_store_router)


@app.get("/")
def root():
    return {"message": "Platform is running successfully"}