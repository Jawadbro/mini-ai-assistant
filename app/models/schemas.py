"""Pydantic models for API requests/responses."""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated conversation/session id")
    message: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    source: str
    chunk_id: int
    snippet: str


class ToolCallTrace(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: dict


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    route: Literal["knowledge_base", "tool_call", "direct"]
    sources: List[SourceChunk] = []
    tool_calls: List[ToolCallTrace] = []


class HistoryTurn(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: List[HistoryTurn]
