import requests
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import time
import re

# Direct sentence-transformers
from sentence_transformers import SentenceTransformer

# Minimal LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.tools import tool

# Configuration - OPTIMIZED FOR SPEED
CHUNK_SIZE = 1000  # Larger chunks = fewer embeddings = faster
CHUNK_OVERLAP = 200   # Minimal overlap = faster chunking
EMBEDDINGS_PATH = os.path.join(os.getcwd(), "data", "embeddings.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_WORKERS = 8  # More parallel workers

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCmJN2FUOs0k7YFw0SgHBLII1hfBpkxO1s"

from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class LightningFastPDFPipeline:
    def __init__(self):
        print("🚀 Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME, device='cpu')
        self.embeddings = None
        self.chunks = None
        self.chunk_texts = None

        # Pre-compiled regex for table detection (much faster than string operations)
        self.table_patterns = [
            re.compile(r'^\s*[\d\$₹%,.\s]+\s*[\d\$₹%,.\s]+\s*[\d\$₹%,.\s]+', re.MULTILINE),
            re.compile(r'premium|age|amount|benefit|sum|rate|policy', re.IGNORECASE),
            re.compile(r'^\s*\w+\s+\d+\s+\d+', re.MULTILINE)
        ]

    def fetch_pdf_lightning(self, url: str) -> bytes:
        """Fastest possible PDF fetch"""
        print("⚡ Fetching PDF...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    def extract_text_only_fast(self, pdf_path: str, page_num: int) -> Dict:
        """FASTEST extraction - text only, no table detection overhead"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]

            # Get raw text - fastest method
            text = page.get_text()
            doc.close()

            # Quick table detection using regex (much faster than PyMuPDF tables)
            has_table_pattern = any(pattern.search(text) for pattern in self.table_patterns[:2])

            return {
                'page_num': page_num + 1,
                'text': text.strip(),
                'has_potential_table': has_table_pattern
            }
        except Exception as e:
            print(f"Page {page_num + 1}: {e}")
            return {'page_num': page_num + 1, 'text': '', 'has_potential_table': False}

    def extract_pdf_lightning(self, pdf_bytes: bytes) -> List[Document]:
        """Lightning-fast parallel extraction"""
        os.makedirs("data", exist_ok=True)
        temp_path = os.path.join("data", "temp_pdf.pdf")
        try:
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)

            doc = fitz.open(temp_path)
            num_pages = doc.page_count
            doc.close()

            print(f"⚡ Processing {num_pages} pages with {MAX_WORKERS} workers...")

            # Maximum parallelization
            results = {}
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_page = {
                    executor.submit(self.extract_text_only_fast, temp_path, i): i
                    for i in range(num_pages)
                }

                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    results[page_num] = future.result()

            # Convert to documents
            documents = []
            table_pages = 0
            for i in range(num_pages):
                if i in results and results[i]['text']:
                    if results[i]['has_potential_table']:
                        table_pages += 1

                    doc = Document(
                        page_content=results[i]['text'],
                        metadata={
                            'page': results[i]['page_num'],
                            'source': 'PDF',
                            'has_table_pattern': results[i]['has_potential_table']
                        }
                    )
                    documents.append(doc)

            print(f"⚡ Extracted {len(documents)} pages ({table_pages} with table patterns)")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return documents

    def lightning_chunking(self, documents: List[Document]) -> List[Document]:
        """Ultra-fast chunking with table pattern preservation"""
        if not documents:
            return []

        print("⚡ Lightning chunking...")

        # Single splitter for speed
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". "],  # Fewer separators = faster
            length_function=len
        )

        all_chunks = []
        for doc in documents:
            if not doc.page_content.strip():
                continue

            # For pages with table patterns, try to keep larger chunks
            if doc.metadata.get('has_table_pattern'):
                # Use larger chunks for potential tables
                large_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE * 2,  # Double size for tables
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n"],
                    length_function=len
                )
                chunks = large_splitter.split_documents([doc])

                # Mark as potential table chunks
                for chunk in chunks:
                    chunk.metadata['content_type'] = 'potential_table'

            else:
                chunks = splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata['content_type'] = 'text'

            all_chunks.extend(chunks)

        table_chunks = len([c for c in all_chunks if c.metadata.get('content_type') == 'potential_table'])
        text_chunks = len(all_chunks) - table_chunks

        print(f"⚡ Created {len(all_chunks)} chunks: {text_chunks} text, {table_chunks} table-pattern chunks")
        return all_chunks

    def embed_vectorized_batch(self, texts: List[str]) -> np.ndarray:
        """Maximum speed embedding with large batches"""
        print(f"⚡ Embedding {len(texts)} chunks in large batches...")

        # Use maximum batch size for speed
        batch_size = min(64, len(texts))  # Larger batches = faster

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Single embedding call per batch
            batch_embeddings = self.model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=(i == 0),  # Show progress only once
                convert_to_numpy=True,
                normalize_embeddings=True  # Pre-normalize for faster cosine similarity
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def create_lightning_vectorstore(self, chunks: List[Document]) -> None:
        """Fastest possible vectorstore creation"""
        if not chunks:
            return

        # Check cache first
        if os.path.exists(EMBEDDINGS_PATH):
            try:
                print("⚡ Loading cached embeddings...")
                with open(EMBEDDINGS_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.chunk_texts = data['chunk_texts']
                    self.chunks = data['chunks']

                    table_count = len([c for c in self.chunks if 'table' in c.metadata.get('content_type', '')])
                    print(f"⚡ Loaded {len(self.chunks)} cached chunks ({table_count} table-pattern)")
                    return
            except:
                print("Cache load failed, creating new embeddings...")

        # Create embeddings with maximum speed
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embed_vectorized_batch(chunk_texts)

        # Store everything
        self.embeddings = embeddings
        self.chunk_texts = chunk_texts
        self.chunks = chunks

        # Cache with compression for speed
        print("⚡ Caching embeddings...")
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'chunk_texts': chunk_texts,
                'chunks': chunks
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def search_lightning_fast(self, query: str, k: int = 3) -> List[Dict]:
        """Fastest possible similarity search"""
        if self.embeddings is None:
            return []

        # Single embedding call
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Vectorized cosine similarity (embeddings already normalized)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # Fast top-k using argpartition (faster than full sort)
        if len(similarities) > k:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        else:
            top_indices = np.argsort(similarities)[::-1]

        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'content': self.chunk_texts[idx],
                'score': float(similarities[idx]),
                'metadata': self.chunks[idx].metadata
            })

        return results

    def ingest_lightning_fast(self, pdf_url: str) -> None:
        """MAXIMUM SPEED ingestion pipeline with step-by-step timing"""
        import time

        total_start = time.time()
        print("⚡⚡⚡ LIGHTNING FAST INGESTION STARTING ⚡⚡⚡")

        # 1. Fetching PDF
        t0 = time.time()
        pdf_bytes = self.fetch_pdf_lightning(pdf_url)
        print(f"📥 Fetching PDF took {time.time() - t0:.2f} sec")

        # 2. Extract text from PDF
        t1 = time.time()
        documents = self.extract_pdf_lightning(pdf_bytes)
        print(f"📄 PDF extraction took {time.time() - t1:.2f} sec")

        # 3. Split into chunks
        t2 = time.time()
        chunks = self.lightning_chunking(documents)
        print(f"✂️  Chunking took {time.time() - t2:.2f} sec")

        # 4. Embed + save to FAISS
        t3 = time.time()
        self.create_lightning_vectorstore(chunks)
        print(f"🧠 Embedding + vectorstore took {time.time() - t3:.2f} sec")

        # Total time
        total_end = time.time()
        print(f"\n⚡ TOTAL PIPELINE TIME: {total_end - total_start:.2f} sec")
        print(f"⚡ READY TO SEARCH {len(self.chunks) if self.chunks else 0} chunks")


    def search(self, query: str, k: int = 3) -> str:
        """Lightning-fast search with smart formatting"""
        if self.embeddings is None:
            return "Error: No embeddings loaded."

        results = self.search_lightning_fast(query, k)
        if not results:
            return "No results found."

        # Quick formatting
        formatted = []
        for i, result in enumerate(results):
            page = result['metadata'].get('page', '?')
            score = f"{result['score']:.3f}"
            content_type = "📊" if 'table' in result['metadata'].get('content_type', '') else "📝"

            content = result['content']
            if len(content) > 500:
                content = content[:500] + "..."

            formatted.append(f"{i+1}. Page {page} {content_type} ({score}):\n{content}")

        return "\n\n" + ("-" * 50 + "\n").join(formatted)

# Global pipeline
lightning_pipeline = LightningFastPDFPipeline()

@tool
def document_retriever(query: str) -> str:
    """Lightning-fast document search"""
    print('Asked: ',query)
    a = lightning_pipeline.search(query)
    print('response', a)
    print('...............')
    return a

# Usage
# if __name__ == "__main__":
#     PDF_URL = "https://hackrx.in/policies/CHOTGDP23004V012223.pdf"

#     # Lightning ingestion
#     lightning_pipeline.ingest_lightning_fast(PDF_URL)

#     # Lightning searches
#     print("\n" + "=" * 60)
#     print("⚡ LIGHTNING SEARCH TEST:")
#     result = lightning_pipeline.search("premium amount age 25", k=3)
#     print(result)




from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

query_ans_prompt = """
You are a helpful AI assistant, you will recieve query regarding any doc which has been embedded and stored in a vector database and you can query it using 'document_retriever' tool
You have to understand the user query and intelligently answer them after querying and retrieving all relevant info using the tool (you can call the tool multiple times, simultaniously)
You only answer questions related to the doc.
"""
query_ans_prompt2 = """
You are a smart and structured data assistant. You will be given context from a PDF along with a user question about that content.

Your task is:
1. Read and analyze the provided context (from the PDF).
2. Understand the user's question.
3. Generate a clear, accurate, and concise answer based on the PDF and you should only use the data from the data provided.
4. Structure the answer in JSON format based on the type of question asked.

Only respond with a JSON object that includes:
- `answer`: A natural language response to the question.
- `structured_data`: A structured JSON containing relevant data extracted or inferred from the PDF, depending on the question type.

If the question cannot be answered from the provided context, set:
- `answer`: "The answer is not available in the provided data."
- `structured_data`: `null`
"""
query_ans_prompt3 = """
You are a smart, structured and data driven Q&A assistant. You will be given context from a PDF along with a user question about that content.
your task is:
1. Read and analyze the provided context (from the PDF).
2. Understand the user's question and its context in pdf.
3. Search for all the relevant information in the PDF.
4. Sort all the output relavant to the question.
5. Generate a clear, accurate, and concise answer based on the PDF and you should only use the data from the data provided.
6. Structure the answer in JSON format based on the type of question asked.

Only respond with a JSON object that includes:
- `answer`: A natural language response to the question.
- `structured_data`: A structured JSON containing relevant data extracted or inferred from the PDF, depending on the question type.

If the question cannot be answered from the provided context, set:
- `answer`: "The answer is not available in the provided data."
- `structured_data`: `null`
"""
Get_query_solved = create_react_agent(
    llm,
    tools = [document_retriever],
    prompt= query_ans_prompt
)
# query = input('what is your query? ')

# brief = Get_query_solved.invoke({"messages": [HumanMessage(content= f"User query: {query}" )]})

# print(brief['messages'][-1].content)
    
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: list

@app.post("/hackrx/run")
async def run_query(request: QueryRequest):
    try:
        # Step 1: Ingest the document
        lightning_pipeline.ingest_lightning_fast(request.documents)

        # Step 2: Answer each question
        answers = []
        for question in request.questions:
            result = Get_query_solved.invoke({
                "messages": [HumanMessage(content=f"User query: {question}")]
            })
            message = result['messages'][-1].content

            # Optional: Extract only the 'answer' field if the LLM returns JSON
            try:
                import json
                parsed = json.loads(message)
                answers.append(parsed.get('answer', message))
            except:
                answers.append(message)

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
