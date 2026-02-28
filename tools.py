import requests
from bs4 import BeautifulSoup
from ddgs import DDGS  # <-- Updated import
import re
from playwright.sync_api import sync_playwright # <-- New Import
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- 1. LOAD THE UNIFIED EMBEDDING MODEL INTO RAM ONCE ---
print("[SYSTEM] Loading BAAI/bge-small-en-v1.5 into memory...")

# We use the exact BGE model you specified. It still outputs a 384-dimensional vector,
# so the Cosine Similarity math below it requires zero changes.
embedder = SentenceTransformer('./local_models/bge-small-en-v1.5')
# --- TOOL 1: THE SEARCHER ---
def search_web(query, max_results=3):
    """
    Simulates the 'Google Search' step.
    Returns a list of dictionaries containing title, href, and body.
    """
    print(f"[*] Searching for: {query}...")
    try:
        # The new DDGS API is simpler and directly returns a list
        results = DDGS().text(query, max_results=max_results)
        
        # Fallback check if results is None
        if not results:
            return []
            
        return results
    except Exception as e:
        print(f"[!] Search Error: {e}")
        return []

# --- TOOL 2: THE SCRAPER ---
def scrape_website(url, search_query):
    """
    Uses a headless Chromium browser to render JavaScript and bypass basic anti-bot protections.
    """
    print(f"[*] Scraping (Headless Browser): {url}...")
    
    try:
        # Boot up the Chromium engine
        with sync_playwright() as p:
            # Launch invisible browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            # Navigate and wait for the DOM to settle (max 15 seconds)
            # networkidle means it waits until network traffic stops (JS has finished loading)
            page.goto(url, timeout=15000, wait_until="networkidle")
            
            # Extract the fully rendered HTML
            html_content = page.content()
            browser.close()
            
        # 1. Parse the DOM Tree
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 2. Strip the noise
        for script in soup(["script", "style", "nav", "footer", "form", "svg"]):
            script.decompose() 

        text = soup.get_text(separator=' ')
        clean_text = ' '.join(text.split())
        
        # 3. Compress context via BM25
        print(f"[*] Compressing context for query: '{search_query}'...")
        filtered_text = filter_relevant_chunks(search_query, clean_text)
        
        if not filtered_text:
            return "[!] Scraped successfully, but found no text relevant to the query."
            
        return filtered_text
        
    except Exception as e:
        return f"[!] Error scraping {url}: {e}"

# --- NEW: THE BM25 FILTER (THE L1 CACHE) ---
def filter_relevant_chunks(query, raw_text, top_k=3):
    """
    Filters text by converting chunks into vectors and calculating Cosine Similarity
    against the query vector. Solves the vocabulary mismatch problem.
    """
    chunks = sliding_window_chunk(raw_text)
    if not chunks:
        return ""

    # Convert the query string into a 384-dimensional float array
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Convert all sliding window chunks into vectors
    # This runs as a highly optimized C++ batch operation under the hood
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)

    # Compute Cosine Similarity between the query and all chunks simultaneously
    # Returns a tensor of scores between -1.0 and 1.0
    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    # Convert tensors to CPU numpy arrays for sorting
    scores_np = cos_scores.cpu().numpy()
    
    # Priority Queue logic: Pair chunks with scores, sort descending
    ranked_chunks = sorted(zip(scores_np, chunks), key=lambda x: x[0], reverse=True)
    
    # Filter out garbage (anything below a 0.2 similarity threshold is usually unrelated)
    best_chunks = [chunk for score, chunk in ranked_chunks[:top_k] if score > 0.2]

    return "\n...\n".join(best_chunks)