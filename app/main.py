"""
FastAPI entrypoint for the mini AI assistant.

Routes:
  POST /ingest        - upload and ingest a PDF/TXT/MD file
  POST /chat          - send a chat message, get a routed + grounded reply
  GET  /chat/{sid}    - inspect a session's conversation history
  DELETE /chat/{sid}  - clear a session's memory
  GET  /health        - liveness check
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.config import UPLOAD_DIR, OPENAI_API_KEY
from app.ingestion.loader import load_document, UnsupportedFileType
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import store_chunks, new_doc_id
from app.agent.planner import handle_chat
from app.memory import session_memory
from app.models.schemas import (
    IngestResponse,
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    HistoryTurn,
)

app = FastAPI(
    title="Mini AI Assistant",
    description="Knowledge ingestion + RAG chat + memory + tool calling demo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    frontend_ok = frontend_dir.exists() and (frontend_dir / "index.html").exists()
    return {
        "status": "ok",
        "openai_key_configured": bool(OPENAI_API_KEY),
        "frontend_mounted": frontend_ok,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    doc_id = new_doc_id()
    dest_path = UPLOAD_DIR / f"{doc_id}{suffix}"

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")
    finally:
        file.file.close()

    try:
        text = load_document(dest_path)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {exc}")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Document appears to be empty or text could not be extracted (e.g. a scanned/image-only PDF).",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No usable text chunks extracted from document.")

    try:
        num_stored = store_chunks(chunks, source=file.filename, doc_id=doc_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding/storage failed: {exc}")

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=num_stored,
        message=f"Ingested '{file.filename}' into the knowledge base as {num_stored} chunks.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )
    try:
        return handle_chat(request.session_id, request.message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat processing failed: {exc}")


@app.get("/chat/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    turns = session_memory.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        turns=[HistoryTurn(role=t["role"], content=t["content"]) for t in turns],
    )


@app.delete("/chat/{session_id}")
async def clear_history(session_id: str):
    session_memory.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


# Serve the minimal demo frontend, if present, at /ui
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_MOUNTED = frontend_dir.exists() and (frontend_dir / "index.html").exists()
if FRONTEND_MOUNTED:
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
else:
    print(
        f"WARNING: frontend directory not found or missing index.html at "
        f"'{frontend_dir}' -- /ui will 404. Check that the 'frontend/' folder "
        f"was included in this deployment."
    )