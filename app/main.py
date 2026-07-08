import os
# Limit CPU thread usage for math/deep learning libraries to prevent system freezes
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.presentation.api import router as api_router, vector_store
from app.application.memory_daemon import MemoryDaemon
from app.infrastructure.vision_adapter import VisionAdapter
from app.core.config import settings

# Configure logging for the entire app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)

# Initialize Memory Daemon (Producer) and Vision Adapter (Consumer)
memory_daemon = MemoryDaemon()
vision_adapter = VisionAdapter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Screen-capture "contextual memory" is privacy-sensitive and opt-out via config.
    if settings.SCREEN_CAPTURE_ENABLED:
        memory_daemon.start()
        asyncio.create_task(vision_adapter.start_event_consumer(vector_store))
        logging.info("Startup complete: screen-capture memory daemon active.")
    else:
        logging.info("Startup complete: screen-capture disabled (SCREEN_CAPTURE_ENABLED=false).")

    yield

    if settings.SCREEN_CAPTURE_ENABLED:
        memory_daemon.stop()
    logging.info("Shutdown complete.")


app = FastAPI(
    title="RAG API Hub",
    description="A FastAPI RAG application built using Clean Architecture principles.",
    version="1.0.0",
    lifespan=lifespan
)


# Include the API presentation router (MUST be registered before StaticFiles mount)
app.include_router(api_router)

# Serve the frontend UI from the static directory.
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
def root():
    """Redirect root to the frontend UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")
