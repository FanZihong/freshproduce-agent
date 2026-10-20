"""会话与流式对话接口。"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatMessageOut, ChatSessionOut, ChatStreamRequest
from app.services import session_service
from app.services.chat_service import stream_chat

router = APIRouter(tags=["chat"])


@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions() -> list[dict]:
    return session_service.list_sessions()


@router.post("/sessions", response_model=ChatSessionOut)
def create_session() -> dict:
    return session_service.create_session()


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def list_messages(session_id: str) -> list[dict]:
    return session_service.get_messages(session_id)


@router.post("/chat/stream")
async def chat_stream(payload: ChatStreamRequest) -> StreamingResponse:
    """SSE 流式对话：intent -> tool -> token -> done。"""
    return StreamingResponse(
        stream_chat(payload.sessionId, payload.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
