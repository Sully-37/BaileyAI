from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.bailey import router as bailey_router
from app.api.health import router as health_router
from app.api.mission import router as mission_router
from app.api.startup import router as startup_router
from app.api.websocket import router as websocket_router
from app.config import APP_NAME
from app.logging_config import configure_logging

configure_logging()

app = FastAPI(title=APP_NAME)

# Existing diagnostic and compatibility routes.
app.include_router(health_router)
app.include_router(startup_router)
app.include_router(websocket_router)
app.include_router(mission_router)

# New live Bailey conversation routes.
app.include_router(bailey_router)

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)


@app.get("/")
async def root():
    """
    Serves the Bailey browser interface.
    """

    return FileResponse("app/static/index.html")