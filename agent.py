import os
from database import SessionLocal
import models
from dotenv import load_dotenv

# Force Python to read your .env file
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
# The Heterogeneous Models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from tavily import TavilyClient
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

# 1. State Definition
class AgentState(TypedDict):
    workspace_id: str
    messages: Annotated[list, add_messages] 

# 2. Initialize Models & APIs
embed_model = HuggingFaceEmbedding(model_name="./local_models/bge-small-en-v1.5")

orchestrator_llm = ChatGroq(
    model_name="moonshotai/kimi-k2-instruct-0905", 
    temperature=0.1,
    max_tokens=4096,
    api_key=os.environ.get("GROQ_API_KEY_ORCH")
)

# --- THE SCRAPER THREAD ---
# Kimi stays as the high-speed web data extractor
kimi_llm = ChatGroq(
    model_name="moonshotai/kimi-k2-instruct-0905",
    temperature=0.1, 
    max_tokens=2048,
    api_key=os.environ.get("GROQ_API_KEY")
)

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# 3. The Tool Definitions

@tool
def list_workspace_documents(workspace_id: str) -> str:
    """
    MANDATORY TOOL: Use this when the user asks "what papers do I have", 
    "list my documents", or asks for a high-level overview of the workspace contents.
    """
    db = SessionLocal()
    try:
        papers = db.query(models.Paper).filter(models.Paper.workspace_id == workspace_id).all()
        if not papers:
            return "The workspace is currently empty. No documents have been uploaded."
        
        result = "Workspace Documents:\n"
        for paper in papers:
            date_str = paper.created_at.strftime("%Y-%m-%d")
            result += f"- Title: '{paper.title}' (Uploaded: {date_str})\n"
        return result
    finally:
        db.close()

@tool
def vector_search(query: str, workspace_id: str) -> str:
    """
    MANDATORY TOOL: You MUST use this tool to answer questions about the specific 
    content, metrics, or details inside the user's uploaded research papers.
    """
    vector_store = PGVectorStore.from_params(
        database="res_ai", host="localhost", password="24072006",
        port=5432, user="admin", table_name="workspace_embeddings", embed_dim=384
    )
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="workspace_id", value=workspace_id)]
    )
    
    retriever = index.as_retriever(similarity_top_k=4, filters=filters)
    nodes = retriever.retrieve(query)
    
    if not nodes:
        return "No relevant information found in the local documents."
    return "\n\n".join([n.get_content() for n in nodes])

@tool
def kimi_web_researcher(query: str) -> str:
    """
    PRIMARY TOOL for real-world facts, dates, and external technical documentation 
    not found in the user's uploaded PDF files.
    """
    try:
        # Phase 1: I/O Fetch
        search_result = tavily_client.search(query=query, search_depth="basic", max_results=3)
        raw_context = "\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in search_result.get("results", [])])
        
        # Phase 2: Kimi Worker Thread Extraction
        # We use Groq's LPU speed to instantly clean the noise out of the raw web data
        extraction_prompt = f"Extract a concise, factual answer to the query: '{query}' using ONLY the following web data:\n{raw_context}"
        
        kimi_response = kimi_llm.invoke([HumanMessage(content=extraction_prompt)])
        return f"Web Research Findings:\n{kimi_response.content}"
        
    except Exception as e:
        return f"Web search failed: {str(e)}"

# 4. Bind Tools to the Orchestrator
tools = [list_workspace_documents, vector_search, kimi_web_researcher]
orchestrator_with_tools = orchestrator_llm.bind_tools(tools)

# 5. The Reasoner Node
# 5. The Reasoner Node
def agent_node(state: AgentState):
    # Short, aggressive system prompt to prevent "I can only..." responses
    system_prompt = f"""ROLE: Autonomous Research Engine (ID: {state['workspace_id']})
    
    EXECUTION RULES:
    EXECUTION ARCHITECTURE & ROUTING RULES:
    1. CHITCHAT & DIRECT ANSWERS: If the user says "Hello", asks "Who are you", or asks a general knowledge/coding question that does not require live data, DO NOT CALL ANY TOOLS. Respond directly using your internal knowledge.
    2. LOCAL FIRST: If the user asks about "my documents", "the paper", or "the uploads", ALWAYS call `vector_search` or `list_workspace_documents` first.
    3. WEB SEARCH ESCALATION: Call `kimi_web_researcher` ONLY if:
       - The user explicitly requests a web search.
       - The data is time-sensitive (e.g., current prices, news, recent software updates).
       - You attempted `vector_search` and the data was missing.
    4. ATTACHED CONTEXT: If the user provides "Selected Context", analyze it directly without tools unless external verification is specifically requested.
    5. Provide the answers in a medium length way
    """
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Invoke the orchestrator
    response = orchestrator_with_tools.invoke(messages)
    return {"messages": [response]}


# 6. Build the DAG
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile()