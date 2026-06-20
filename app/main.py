from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.health import router as health_router
from app.api.startup import router as startup_router
from app.api.websocket import router as websocket_router
from app.config import settings
from app.logging_config import configure_logging

configure_logging()

app = FastAPI(title=settings.APP_NAME)

app.include_router(health_router)
app.include_router(startup_router)
app.include_router(websocket_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")