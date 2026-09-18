"""
Chat API route for side assistant replies.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.db.session import get_db
from app.models.chat import ChatMessage as ChatMessageRow
from app.models.user import User
from app.services.chat_assist import get_chat_reply

router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1, max_length=32)
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatResponse:
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        reply = get_chat_reply(req.message, history=history)
        db.add_all([
            ChatMessageRow(user_id=user.id, role="user", content=req.message.strip()[:4000]),
            ChatMessageRow(user_id=user.id, role="assistant", content=reply[:4000]),
        ])
        db.commit()
        return ChatResponse(reply=reply)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}") from e


@router.get("/chat/history")
def chat_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict[str, list[dict[str, str]]]:
    rows = (
        db.query(ChatMessageRow)
        .filter(ChatMessageRow.user_id == user.id)
        .order_by(ChatMessageRow.created_at.asc())
        .limit(200)
        .all()
    )
    return {"messages": [{"role": row.role, "content": row.content} for row in rows]}
