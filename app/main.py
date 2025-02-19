from typing import List

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

from app.routers.health import health_router
from app.routers.predict import predict_router

# Configure web domain which can access api
origins = [
    "*",
]


def make_middleware() -> List[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
    return middleware


def init_routers(app_: FastAPI) -> None:
    prefix_router = APIRouter(prefix="/api/v1")
    prefix_router.include_router(health_router)
    prefix_router.include_router(predict_router)

    app_.include_router(prefix_router)


def create_app():
    app_ = FastAPI(middleware=make_middleware())
    app_.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    init_routers(app_=app_)

    return app_


app = create_app()
