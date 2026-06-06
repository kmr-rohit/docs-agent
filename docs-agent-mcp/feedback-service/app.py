"""Feedback API for Kubeflow Docs Bot — stores ratings for golden dataset curation."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Kubeflow Docs Feedback API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class FeedbackCreate(BaseModel):
    context_id: str = Field(..., min_length=1, max_length=128)
    message_id: str | None = Field(None, max_length=128)
    query: str = Field(..., min_length=1, max_length=10000)
    response: str = Field(..., min_length=1, max_length=50000)
    citations: list[str] = Field(default_factory=list)
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(None, max_length=2000)
    source: str = Field(default="kubeflow-docs-bot", max_length=128)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/feedback")
def create_feedback(payload: FeedbackCreate) -> dict:
    try:
        saved = db.insert_feedback(payload.model_dump())
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {exc}") from exc
    return {"status": "ok", "feedback": saved}


@app.get("/api/feedback/export")
def export_feedback(
    min_rating: int | None = Query(None, ge=1, le=5),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Export stored feedback rows for golden-dataset curation."""
    rows = db.list_feedback(min_rating=min_rating, limit=limit, offset=offset)
    return {"count": len(rows), "items": rows}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
