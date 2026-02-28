from pydantic import BaseModel, EmailStr
from datetime import datetime
from uuid import UUID
from typing import Optional, List, Dict,  Any




class NoteCreate(BaseModel):
    title: str = "Untitled Note"
    content: Optional[Dict[str, Any]] = {} # Accepts TipTap JSON

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[Dict[str, Any]] = None

class NoteResponse(BaseModel):
    id: UUID
    workspace_id: UUID
    title: str
    content: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class NoteGlobalResponse(NoteResponse):
    workspace_name: str


class UserCreate(BaseModel):
    name: str # <--- Require the frontend to send a name string
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: UUID
    name: str # <--- Expose the name in API responses
    email: EmailStr
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Token schemas remain unchanged
class Token(BaseModel):
    access_token: str
    token_type: str
    name: str
    email: str

class TokenData(BaseModel):
    user_id: str | None = None


# Used for POST request body validation
class WorkspaceCreate(BaseModel):
    title: str
    description: Optional[str] = None

# Used for standardizing the API response back to React
class WorkspaceResponse(BaseModel):
    id: UUID
    title: str
    description: Optional[str]
    created_at: datetime
    user_id: UUID

    class Config:
        from_attributes = True

# Add to schemas.py
class ChatRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None # Added this field
class ChatResponse(BaseModel):
    reply: str



class PaperResponse(BaseModel):
    id: UUID
    title: str
    source_url: str
    created_at: datetime

    class Config:
        from_attributes = True

class MessageResponse(BaseModel):
    id: UUID
    role: str # 'user' or 'assistant'
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

# The Composite Payload
class WorkspaceDetailResponse(BaseModel):
    id: UUID
    title: str
    description: Optional[str]
    created_at: datetime
    # If nothing is found, these default to empty arrays
    papers: List[PaperResponse] = []
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True


class ArxivPaper(BaseModel):
    title: str
    authors: str
    summary: str
    paper_link: str
    pdf_link: str