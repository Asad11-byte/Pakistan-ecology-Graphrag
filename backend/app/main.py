from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

# ==================================================
# CORS
# ==================================================

ALLOWED_ORIGINS = [
    "http://localhost:5173",

    # Replace with your actual Vercel production domain after first deploy
    "https://pakistan-ecology-graphrag.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# API Router (prefixed with /api to match vercel.json
# routing and the frontend's VITE_API_URL=/api)
# ==================================================

api_router = APIRouter(prefix="/api")

api_router.include_router(ingest_router)
api_router.include_router(query_router)


@api_router.get("/")
async def root():
    return {
        "status": "running",
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@api_router.get("/health")
async def health():
    return {"status": "healthy"}


app.include_router(api_router)
