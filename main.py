from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List
import json
import os
import shutil
import logging

# Configure logging to show in your CMD/Terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
import uuid
import fitz  # PyMuPDF
from openai import OpenAI
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import text
import feedparser
import re

from sentence_transformers import SentenceTransformer, util

import models
import schemas
import auth
from database import get_db, engine
from ingestion import process_and_embed_pdf
from agent import app as langgraph_agent

# Initialize Database
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="ResearchHub AI API")

# Add CORS middleware to accept all requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploaded_papers", exist_ok=True)

# Initialize OpenAI Client (Routed to Groq)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY_MIND_MAP"),
    base_url="https://api.groq.com/openai/v1",
)

ARXIV_MAP = {
    "NLP": "cs.CL",
    "Computer Vision": "cs.CV",
    "Machine Learning": "cs.LG",
    "Artificial Intelligence": "cs.AI",
    "Robotics": "cs.RO",
    "Multimodal": "cs.AI",
}

client_sum = OpenAI(
    api_key=os.getenv("GROQ_API_KEY_SUMM"),
    base_url="https://api.groq.com/openai/v1",
)
class MindMapResponse(BaseModel):
    markdown: str


# --- AUTHENTICATION API ---

@app.post("/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    
    new_user = models.User(
        name=user.name, 
        email=user.email, 
        password_hash=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login", response_model=schemas.Token)
def login_user(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    
    if not db_user or not auth.verify_password(user.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": str(db_user.id)})
    return {"access_token": access_token, "token_type": "bearer", "name": db_user.name, "email": db_user.email}

@app.get("/users/me", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user


# --- WORKSPACE API ---

@app.post("/workspaces", response_model=schemas.WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(workspace: schemas.WorkspaceCreate, db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    new_workspace = models.Workspace(
        title=workspace.title,
        description=workspace.description,
        user_id=current_user.id
    )
    
    db.add(new_workspace)
    db.commit()
    db.refresh(new_workspace)
    return new_workspace

@app.get("/workspaces", response_model=List[schemas.WorkspaceResponse])
def get_workspaces(db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    workspaces = (
        db.query(models.Workspace)
        .filter(models.Workspace.user_id == current_user.id)
        .order_by(models.Workspace.created_at.desc())
        .all()
    )
    return workspaces

@app.get("/workspaces/{workspace_id}", response_model=schemas.WorkspaceDetailResponse)
def get_workspace_details(
    workspace_id: str, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == current_user.id
    ).first()

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found or unauthorized")

    papers = (
        db.query(models.Paper)
        .filter(models.Paper.workspace_id == workspace_id)
        .order_by(models.Paper.created_at.desc())
        .all()
    )

    messages = (
        db.query(models.Message)
        .filter(models.Message.workspace_id == workspace_id)
        .order_by(models.Message.created_at.asc())
        .all()
    )

    return {
        "id": workspace.id,
        "title": workspace.title,
        "description": workspace.description,
        "created_at": workspace.created_at,
        "papers": papers,
        "messages": messages
    }


# --- PAPER MANAGEMENT API ---

@app.post("/workspaces/{workspace_id}/papers", status_code=status.HTTP_201_CREATED)
async def upload_paper(
    workspace_id: str,
    title: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    logger.info(f"Starting upload for paper: {title} in workspace: {workspace_id}")
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == current_user.id
    ).first()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found or unauthorized")

    file_extension = file.filename.split(".")[-1]
    safe_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join("uploaded_papers", safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    new_paper = models.Paper(
        workspace_id=workspace.id,
        title=title,
        source_url=file_path 
    )
    db.add(new_paper)
    db.commit()
    db.refresh(new_paper)

    try:
        process_and_embed_pdf(
            file_path=file_path, 
            workspace_id=str(workspace.id), 
            paper_id=str(new_paper.id)
        )
    except Exception as e:
        logger.critical(f"Vector ingestion CRASHED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vector ingestion failed: {str(e)}")

    return {"message": "Paper imported successfully", "paper_id": new_paper.id}

@app.delete("/workspaces/{workspace_id}/papers/{paper_id}", status_code=status.HTTP_200_OK)
def delete_paper(
    workspace_id: str,
    paper_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == current_user.id
    ).first()

    if not workspace:
        raise HTTPException(status_code=403, detail="Unauthorized access to workspace")

    paper = db.query(models.Paper).filter(
        models.Paper.id == paper_id,
        models.Paper.workspace_id == workspace_id
    ).first()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    try:
        delete_vectors_sql = text("""
            DELETE FROM data_workspace_embeddings 
            WHERE metadata_->>'paper_id' = :paper_id
        """)
        db.execute(delete_vectors_sql, {"paper_id": str(paper_id)})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete vector embeddings: {str(e)}")
        
    try:
        if os.path.exists(paper.source_url):
            os.remove(paper.source_url)
    except Exception as e:
        print(f"Warning: Could not delete physical file at {paper.source_url}: {str(e)}")

    try:
        db.delete(paper)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete database record: {str(e)}")

    return {"message": "Paper, file, and embeddings successfully deleted."}

@app.get("/workspaces/{workspace_id}/papers/{paper_id}/view")
def view_paper_document(
    workspace_id: str,
    paper_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == current_user.id
    ).first()

    if not workspace:
        raise HTTPException(status_code=403, detail="Unauthorized access")

    paper = db.query(models.Paper).filter(
        models.Paper.id == paper_id,
        models.Paper.workspace_id == workspace_id
    ).first()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    file_path = paper.source_url
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Physical file missing from server")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"{paper.title}.pdf"
    )


# --- GENERATIVE AI API ---

@app.post("/workspaces/{workspace_id}/chat")
async def chat_with_workspace(
    workspace_id: str, 
    request: schemas.ChatRequest, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == current_user.id
    ).first()
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if request.selected_text and request.selected_text.strip():
        final_user_prompt = f"Regarding the following selected text from the document:\n\"{request.selected_text}\"\n\nQuestion: {request.message}"
    else:
        final_user_prompt = request.message

    raw_history = (
        db.query(models.Message)
        .filter(models.Message.workspace_id == workspace_id)
        .order_by(models.Message.created_at.desc())
        .limit(10)
        .all()
    )
    raw_history.reverse() 
    
    formatted_history = []
    for msg in raw_history:
        if msg.role == "user":
            formatted_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            formatted_history.append(AIMessage(content=msg.content))

    current_messages = formatted_history + [HumanMessage(content=final_user_prompt)]
    
    # Initialize the state WITH the loop counter starting at 0
    initial_state = {"workspace_id": workspace_id, "messages": current_messages, "loop_count": 0}

    async def event_generator():
        final_ai_response = ""
        try:
            for event in langgraph_agent.stream(initial_state, config={"recursion_limit": 15}, stream_mode="updates"):
                for node_name, state_update in event.items():
                    messages = state_update.get("messages", [])
                    if not messages:
                        continue
                        
                    last_msg = messages[-1]

                    if node_name == "agent":
                        # Handle parallel tool calling properly
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                tool_name = tool_call['name']
                                yield f"data: {json.dumps({'type': 'step', 'status': f'Agent requested {tool_name}...'})}\n\n"
                        elif last_msg.content:
                            raw_content = last_msg.content
                            if isinstance(raw_content, list):
                                final_ai_response = "".join(chunk.get("text", "") for chunk in raw_content if isinstance(chunk, dict) and "text" in chunk)
                            else:
                                final_ai_response = str(raw_content)
                                
                            yield f"data: {json.dumps({'type': 'final', 'content': final_ai_response})}\n\n"

                    elif node_name == "tools":
                        tool_name = last_msg.name
                        if tool_name == "vector_search":
                            yield f"data: {json.dumps({'type': 'step', 'status': 'Scanning local workspace vectors...'})}\n\n"
                        elif tool_name == "kimi_web_researcher":
                            yield f"data: {json.dumps({'type': 'step', 'status': 'Extracting live web context...'})}\n\n"
                        elif tool_name == "scrape_specific_url":
                            yield f"data: {json.dumps({'type': 'step', 'status': 'Reading URL directly...'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'step', 'status': f'{tool_name} finished execution.'})}\n\n"

            # Save the final interaction to Postgres
            if final_ai_response:
                user_msg_db = models.Message(workspace_id=workspace_id, role="user", content=final_user_prompt)
                ai_msg_db = models.Message(workspace_id=workspace_id, role="assistant", content=final_ai_response)
                db.add(user_msg_db)
                db.add(ai_msg_db)
                db.commit()

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/generate-map", response_model=MindMapResponse)
def generate_map(file: UploadFile = File(...)): 
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDFs are supported.")

    try:
        pdf_bytes = file.file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        text = ""
        for page in doc:
            text += page.get_text()
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        chunk_size = 4000
        overlap = 400
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        context = "\n".join(chunks[:2])

        prompt = f"""
        You are a senior research scientist. Create a clean hierarchical markdown mind map.
        STRICT RULES:
        - Use exactly ONE H1 title (#)
        - Use '-' for bullets
        - Keep each node under 6 words
        - Ensure 2-3 levels deep
        
        Paper text:
        {context}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You output only structured markdown lists for mind maps. No conversational text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2
        )

        markdown_output = response.choices[0].message.content.strip()
        
        if markdown_output.startswith("```"):
            markdown_output = "\n".join(markdown_output.split("\n")[1:-1])

        return MindMapResponse(markdown=markdown_output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


# --- NOTES API ---

def verify_workspace_owner(db: Session, workspace_id: str, user_id: str):
    """
    Security Helper: Ensures the user making the request actually owns the workspace.
    """
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == user_id
    ).first()
    
    if not workspace:
        raise HTTPException(
            status_code=403, 
            detail="Unauthorized: You do not have access to this workspace."
        )
    return workspace

@app.post("/workspaces/{workspace_id}/notes", response_model=schemas.NoteResponse)
def create_note(
    workspace_id: str, 
    note: schemas.NoteCreate, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    verify_workspace_owner(db, workspace_id, str(current_user.id))
    
    db_note = models.Note(
        workspace_id=workspace_id,
        user_id=str(current_user.id),
        title=note.title,
        content=note.content
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note

@app.get("/workspaces/{workspace_id}/notes", response_model=list[schemas.NoteResponse])
def get_notes(
    workspace_id: str, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    verify_workspace_owner(db, workspace_id, str(current_user.id))
    
    return db.query(models.Note).filter(
        models.Note.workspace_id == workspace_id
    ).order_by(models.Note.updated_at.desc()).all()

@app.put("/workspaces/{workspace_id}/notes/{note_id}", response_model=schemas.NoteResponse)
def auto_save_note(
    workspace_id: str, 
    note_id: str, 
    note_update: schemas.NoteUpdate, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    verify_workspace_owner(db, workspace_id, str(current_user.id))
    
    db_note = db.query(models.Note).filter(
        models.Note.id == note_id, 
        models.Note.workspace_id == workspace_id
    ).first()
    
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")

    if note_update.title is not None:
        db_note.title = note_update.title
    if note_update.content is not None:
        db_note.content = note_update.content

    db.commit()
    db.refresh(db_note)
    return db_note

@app.delete("/workspaces/{workspace_id}/notes/{note_id}")
def delete_note(
    workspace_id: str, 
    note_id: str, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    verify_workspace_owner(db, workspace_id, str(current_user.id))
    
    db_note = db.query(models.Note).filter(
        models.Note.id == note_id,
        models.Note.workspace_id == workspace_id
    ).first()
    
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")
        
    db.delete(db_note)
    db.commit()
    return {"message": "Note deleted successfully"}




# Ensure you have your Groq client initialized somewhere at the top of main.py
# client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

@app.post("/api/summarize-pdf")
async def summarize_pdf_endpoint(
    file: UploadFile = File(...),
    length_type: str = Form(...),
    format_type: str = Form(...)
):
    # 1. Read file into memory BEFORE starting the stream to prevent file-closure errors
    pdf_bytes = await file.read()
    
    async def event_generator():
        try:
            # Step 1: Extraction
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting text from PDF...'})}\n\n"
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)

            if len(text.strip()) < 100:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Could not extract sufficient text from PDF.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Step 2: Chunking
            chunk_size = 4000
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            
            # Step 3: Map Step (Partial Summaries)
            partials = []
            for i, chunk in enumerate(chunks):
                yield f"data: {json.dumps({'type': 'status', 'message': f'Analyzing section {i+1} of {len(chunks)}...'})}\n\n"
                
                # Using the standard sync client (in a real prod app, use AsyncGroq, but this works fine for the hackathon)
                response = client_sum.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an expert research paper analyst. Extract the most important technical contributions."},
                        {"role": "user", "content": f"Summarize this research paper section:\n\n{chunk}"},
                    ],
                    temperature=0.15,
                    max_tokens=300,
                )
                partials.append(response.choices[0].message.content)
                await asyncio.sleep(0.1) # Yield control back to event loop

            # Step 4: Reduce Step (Combine)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Synthesizing combined insights...'})}\n\n"
            combined_response = client_sum.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Combine the following into a coherent research summary."},
                    {"role": "user", "content": "\n\n".join(partials)},
                ],
                temperature=0.15,
                max_tokens=500,
            )
            master_summary = combined_response.choices[0].message.content

            # Step 5: Formatting Rules
            length_map = {
                "Short": "120–180 words",
                "Medium": "250–400 words",
                "Long ⭐ Recommended": "500–800 words",
            }
            target_length = length_map.get(length_type, "250-400 words")

            if format_type == "Paragraph":
                format_instruction = "Write in well-structured academic paragraphs."
            elif format_type == "Bullets":
                format_instruction = "Write in clear structured bullet points highlighting key ideas."
            else:
                format_instruction = "Start with a short paragraph overview, then provide structured bullets."

            # Step 6: Final Stream
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating final formatted summary...'})}\n\n"
            
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                stream=True,
                messages=[
                    {"role": "system", "content": "You are a senior research analyst. Write precise, faithful, technical summaries."},
                    {"role": "user", "content": f"Write a high-quality research paper summary of length {target_length}.\n\n{format_instruction}\n\n{master_summary}"},
                ],
                temperature=0.1,
                max_tokens=900,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    # Stream the actual summary tokens to the frontend
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.choices[0].delta.content})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



# --- ENDPOINTS ---
@app.get("/api/arxiv/latest", response_model=List[schemas.ArxivPaper])
def get_latest_papers(genre: str, max_results: int = 6):
    """Fetches the latest papers sorted by submission date."""
    if genre not in ARXIV_MAP:
        raise HTTPException(status_code=400, detail="Invalid genre category")
        
    category = ARXIV_MAP[genre]
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=cat:{category}&"
        f"sortBy=submittedDate&sortOrder=descending&"
        f"max_results={max_results}"
    )
    
    # feedparser is a blocking I/O operation
    feed = feedparser.parse(url)
    
    papers = []
    for entry in feed.entries:
        papers.append(schemas.ArxivPaper(
            title=entry.title,
            authors=", ".join(author.name for author in entry.authors),
            summary=entry.summary.replace("\n", " ")[:300] + "...",
            paper_link=entry.link,
            pdf_link=entry.link.replace("abs", "pdf")
        ))
    return papers

@app.get("/api/arxiv/search", response_model=List[schemas.ArxivPaper])
def search_arxiv(query: str, max_results: int = 8):
    """Searches arXiv by keyword, sorted by relevance."""
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{query}&"
        f"sortBy=relevance&sortOrder=descending&"
        f"max_results={max_results}"
    )
    
    feed = feedparser.parse(url)
    
    papers = []
    for entry in feed.entries:
        papers.append(schemas.ArxivPaper(
            title=entry.title,
            authors=", ".join(author.name for author in entry.authors),
            summary=entry.summary.replace("\n", " ")[:300] + "...",
            paper_link=entry.link,
            pdf_link=entry.link.replace("abs", "pdf")
        ))
    return papers



# --- LOAD VECTOR MODEL (Singleton) ---
# Maps text into a 384-dimensional dense vector space
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def detect_domain(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["translation", "language model", "bert", "nlp", "transformer"]): return "NLP"
    if any(k in t for k in ["image", "vision", "cnn", "segmentation", "object detection"]): return "Computer Vision"
    if any(k in t for k in ["reinforcement", "policy", "agent", "reward"]): return "Reinforcement Learning"
    return "General AI"

def extract_sections(text: str) -> dict:
    sections = {"abstract": "", "introduction": "", "method": "", "results": "", "conclusion": ""}
    text_lower = text.lower()
    for key in sections.keys():
        pattern = rf"{key}(.+?)(?=\n[A-Z ]{{3,}}|\Z)"
        match = re.search(pattern, text_lower, re.DOTALL)
        if match:
            sections[key] = match.group(1)[:2000]
    return sections

@app.post("/api/compare-papers")
async def compare_papers_endpoint(
    file_a: UploadFile = File(...), 
    file_b: UploadFile = File(...)
):
    # Load both files into RAM immediately
    pdf_bytes_a = await file_a.read()
    pdf_bytes_b = await file_b.read()

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Parsing PDFs and extracting sections...'})}\n\n"
            
            doc_a = fitz.open(stream=pdf_bytes_a, filetype="pdf")
            doc_b = fitz.open(stream=pdf_bytes_b, filetype="pdf")
            text_a = "".join(page.get_text() for page in doc_a)
            text_b = "".join(page.get_text() for page in doc_b)

            sec_a = extract_sections(text_a)
            sec_b = extract_sections(text_b)

            # --- PHASE 1: VECTOR MATH (CPU Bound) ---
            yield f"data: {json.dumps({'type': 'status', 'message': 'Computing R^384 cosine similarities...'})}\n\n"
            scores = {}
            for key in sec_a.keys():
                if sec_a[key].strip() and sec_b[key].strip():
                    # Transform strings to 384D Tensors
                    emb_a = embedding_model.encode(sec_a[key], convert_to_tensor=True)
                    emb_b = embedding_model.encode(sec_b[key], convert_to_tensor=True)
                    # O(N) dot product
                    sim = util.cos_sim(emb_a, emb_b).item()
                    scores[key] = round(sim, 2)

            overall = sum(scores.values()) / len(scores) if scores else 0.0

            # --- MULTIPLEXING: Send the Metrics Payload ---
            metrics_payload = {
                "type": "metrics",
                "domain_a": detect_domain(text_a),
                "domain_b": detect_domain(text_b),
                "overall": round(overall, 2),
                "scores": scores
            }
            yield f"data: {json.dumps(metrics_payload)}\n\n"

            # --- PHASE 2: LLM SYNTHESIS (I/O Bound) ---
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating expert LLM synthesis...'})}\n\n"
            
            prompt = (
                "Compare the following two research papers.\n\n"
                f"Paper A (Abstract): {sec_a['abstract'][:1000]}\n"
                f"Paper A (Method): {sec_a['method'][:1000]}\n\n"
                f"Paper B (Abstract): {sec_b['abstract'][:1000]}\n"
                f"Paper B (Method): {sec_b['method'][:1000]}"
            )

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                stream=True,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst. Focus on methodology differences and evaluation strategies."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.choices[0].delta.content})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/notes", response_model=list[schemas.NoteGlobalResponse])
def get_all_user_notes(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    # Perform a JOIN to get the workspace name alongside the note
    results = db.query(models.Note, models.Workspace.title.label('workspace_name'))\
        .join(models.Workspace, models.Note.workspace_id == models.Workspace.id)\
        .filter(models.Note.user_id == str(current_user.id))\
        .order_by(models.Note.updated_at.desc())\
        .all()
    
    # Map the tuple results to your response schema
    return [{"workspace_name": r.workspace_name, **r.Note.__dict__} for r in results]


@app.get("/notes", response_model=list[schemas.NoteGlobalResponse])
def get_all_user_notes(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    """
    Fetches all notes for the current user across all workspaces.
    Includes the workspace title using a SQL JOIN.
    """
    results = db.query(models.Note, models.Workspace.title.label('workspace_name'))\
        .join(models.Workspace, models.Note.workspace_id == models.Workspace.id)\
        .filter(models.Note.user_id == str(current_user.id))\
        .order_by(models.Note.updated_at.desc())\
        .all()
    
    # Unpack the SQLAlchemy Row object into a dictionary matching the schema
    return [
        {
            **note.Note.__dict__, 
            "workspace_name": note.workspace_name
        } 
        for note in results
    ]