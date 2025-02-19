from typing import Annotated

from fastapi import APIRouter, Depends, status

health_router = APIRouter()


@health_router.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {"status": "ok"}
