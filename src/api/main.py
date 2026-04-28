"""
FastAPI application entry point.
Mounts all routers and configures middleware.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.routes import detect, explain, models as model_routes
from src.api.schemas import HealthResponse


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    print("✅ ts-anomaly-llm API starting up")
    yield
    print("🛑 ts-anomaly-llm API shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Time Series Anomaly Detection + LLM Explanations",
        description=(
            "Two-layer system: anomaly detection (Isolation Forest / LSTM-AE / Transformer-AE) "
            "+ LLM-powered plain-English explanations via LangChain."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    app.include_router(detect.router, prefix="/api/v1", tags=["Detection"])
    app.include_router(explain.router, prefix="/api/v1", tags=["Explanation"])
    app.include_router(model_routes.router, prefix="/api/v1", tags=["Models"])

    # --- Root redirect ---
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    # --- Health check ---
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        return HealthResponse(status="ok", version="1.0.0")

    return app


app = create_app()
