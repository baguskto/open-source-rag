from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from llama_cpp import Llama
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List
from datetime import datetime
import textwrap
import numpy as np
import PyPDF2
import json
import io

app = FastAPI()

CHUNK_SIZE = 500  # Hyperparameter for chunking documents
TOP_K = 2         # Number of top similar items to retrieve
MAX_TOKENS = 2048  # Model's maximum context window

llm = Llama.from_pretrained(
    repo_id="rubythalib33/llama3_1_8b_finetuned_bahasa_indonesia",
    filename="unsloth.Q4_K_M.gguf",
    n_gpu_layers=-1,
    # n_ctx=MAX_TOKENS
)

text_embedder = Llama.from_pretrained(
    repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
    embedding=True
)

alpaca_prompt = """Di bawah ini adalah instruksi yang menjelaskan tugas, dipasangkan dengan masukan yang memberikan konteks lebih lanjut. Tulis tanggapan yang melengkapi permintaan dengan tepat.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# Database credentials
DB_NAME = "rag_3"
DB_USER = "postgres"
DB_PASSWORD = "example"
DB_HOST = "localhost"
DB_PORT = "5432"

DATABASE_URL = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

# Define request and response models
class ChatRequest(BaseModel):
    instruction: str
    input_data: str = ""

class ChatResponse(BaseModel):
    response: str
    chat_history_id: int

class ReactionRequest(BaseModel):
    chat_history_id: int
    reaction: str  # "like" or "dislike"

class RegenerateRequest(BaseModel):
    chat_history_id: int

def get_cached_response(instruction: str, input_data: str):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        query = """
        SELECT ch.response, ch.id FROM chat_history ch
        JOIN cached_chat cc ON cc.chat_history_id = ch.id
        WHERE ch.instruction = %s AND ch.input_data = %s
        """
        cursor.execute(query, (instruction, input_data))
        result = cursor.fetchone()
        conn.close()
        return result  # result will have 'response' and 'id'
    except Exception as e:
        print(f"Error accessing database: {e}")
        return None
    
def cache_response(chat_history_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        query = "INSERT INTO cached_chat (chat_history_id) VALUES (%s)"
        cursor.execute(query, (chat_history_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error inserting into database: {e}")

def save_chat_history(instruction: str, input_data: str, response: str, input_vector, response_vector):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        query = """
        INSERT INTO chat_history (instruction, input_data, response, input_vector, response_vector)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """
        cursor.execute(query, (instruction, input_data, response, input_vector.tolist(), response_vector.tolist()))
        chat_history_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return chat_history_id
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return None
    
def add_reaction(chat_history_id: int, reaction: str):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        query = "INSERT INTO analytics (type, chat_history_id) VALUES (%s, %s)"
        cursor.execute(query, (reaction, chat_history_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving reaction: {e}")

def get_chat_history_by_id(chat_history_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        query = "SELECT instruction, input_data, response FROM chat_history WHERE id = %s"
        cursor.execute(query, (chat_history_id,))
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return None
    
def get_all_reactions(reaction_type: str, start_datetime: Optional[str], end_datetime: Optional[str]):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        query = "SELECT * FROM analytics WHERE type = %s"
        params = [reaction_type]
        
        # Filter by start and end datetime
        if start_datetime:
            query += " AND created_at >= %s"
            params.append(start_datetime)
        
        if end_datetime:
            query += " AND created_at <= %s"
            params.append(end_datetime)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Error retrieving reactions: {e}")
        return None
    
def embed_text(text:str):
    embeddings = text_embedder.embed(text)
    embeddings = np.array(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings

def search_knowledge_base(query_embedding, top_k=TOP_K):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        query = """
        SELECT chunk_text FROM knowledge_vector
        ORDER BY chunk_vector <-> (%s)::vector LIMIT %s;
        """

        cursor.execute(query, (query_embedding.tolist(), top_k))
        results = cursor.fetchall()
        conn.close()
        return [row['chunk_text'] for row in results]
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return []
    
def search_chat_history(query_embedding, top_k=TOP_K):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        query = """
        SELECT instruction, input_data, response FROM chat_history
        WHERE input_vector IS NOT NULL
        ORDER BY input_vector <-> (%s)::vector LIMIT %s;
        """
        cursor.execute(query, (query_embedding.tolist(), top_k))
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Error searching chat history: {e}")
        return []
    
@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    cached = get_cached_response(request.instruction, request.input_data)
    # if cached:
    #     return ChatResponse(response=cached['response'], chat_history_id=cached['id'])
    
    combined_input = request.instruction + " " + request.input_data
    input_embedding = embed_text(combined_input)

    relevant_chunks = search_knowledge_base(input_embedding)

    preprompt = "Kamu adalah seorang chatbot yang ditugaskan untuk menjadi customer service sebuah perusahaan bernama emerald. tolong hanya jawab sesuai konteks yang diberikan"

    knowledge_context = "\n".join(relevant_chunks)

    system_content = preprompt + "\n\n" + knowledge_context

    messages = [
        # {"role": "system", "content": },
        # {"role": "user", "content":system_content+"\n"+combined_input}
        {"role": "user", "content":alpaca_prompt.format(combined_input, system_content, '')}
    ]

    print("MESSAGES:\n"+json.dumps(messages, indent=4))

    result = llm.create_chat_completion(
        messages=messages,
        # stop = ["[/INST]"]
    )

    response_text = result['choices'][0]['message']['content'].replace('<<SYS>>','').strip()

    response_embedding = embed_text(response_text)

    chat_history_id = save_chat_history(request.instruction, request.input_data, response_text, input_embedding, response_embedding)

    cache_response(chat_history_id)

    return ChatResponse(response=response_text, chat_history_id=chat_history_id)

@app.post("/regenerate", response_model=ChatResponse)
async def regenerate_chat(request: RegenerateRequest):
    history = get_chat_history_by_id(request.chat_history_id)
    if not history:
        raise HTTPException(status_code=404, detail="Chat history not found")
    
    combined_input = history['instruction'] + " " + history['input_data']
    input_embedding = embed_text(combined_input)

    relevant_chunks = search_knowledge_base(input_embedding)

    preprompt = "Kamu adalah seorang chatbot yang ditugaskan untuk menjadi customer service sebuah perusahaan bernama emerald."

    knowledge_context = "\n".join(relevant_chunks)
    system_content = preprompt + "\n\n" + knowledge_context

    user_input = history['instruction'] + " " + history['input_data']

    # Construct messages
    messages = [
        # {"role": "system", "content": },
        # {"role": "user", "content":system_content+"\n"+combined_input}
        {"role": "user", "content":alpaca_prompt.format(combined_input, system_content, '')}
    ]

    result = llm.create_chat_completion(
        messages=messages,
        #  stop = ["[/INST]"]
    )

    response_text = result['choices'][0]['message']['content'].replace('<<SYS>>','').strip()

    response_embedding = embed_text(response_text)

    new_chat_history_id = save_chat_history(history['instruction'], history['input_data'], response_text, input_embedding, response_embedding)

    add_reaction(request.chat_history_id, "regenerate")

    return ChatResponse(response=response_text, chat_history_id=new_chat_history_id)

@app.post("/react")
async def react_to_chat(request: ReactionRequest):
    if request.reaction not in ["like", "dislike"]:
        raise HTTPException(status_code=400, detail="Invalid reaction. Use 'like' or 'dislike'.")

    # Add reaction to the chat history
    add_reaction(request.chat_history_id, request.reaction)

    return {"message": "Reaction saved successfully."}

@app.get("/all-reactions")
async def get_all_reaction(
    reaction_type: str = Query(..., description="Filter by reaction type: like, dislike, regenerate"),
    start_datetime: Optional[str] = Query(None, description="Start datetime in the format YYYY-MM-DD HH:MM:SS"),
    end_datetime: Optional[str] = Query(None, description="End datetime in the format YYYY-MM-DD HH:MM:SS")
):
    # Get all reactions with the filters
    reactions = get_all_reactions(reaction_type, start_datetime, end_datetime)
    
    if reactions is None:
        raise HTTPException(status_code=500, detail="Error retrieving reactions.")
    
    return {"reactions": reactions}

@app.post("/add_knowledge")
async def add_knowledge(document_id: str = Query(...), file: UploadFile = File(...)):
    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    try:
        if file.content_type == "text/plain":
            content = await file.read()
            document_text = content.decode('utf-8')
        elif file.content_type == "application/pdf":
            content = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            document_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    document_text += text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
    
    chunks = textwrap.wrap(document_text, width=CHUNK_SIZE)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        knowledge_query = """
        INSERT INTO knowledge (document_id)
        VALUES (%s) RETURNING id
        """

        cursor.execute(knowledge_query, (document_id,))
        knowledge_id = cursor.fetchone()[0]

        for chunk in chunks:
            chunk_embedding = embed_text(chunk)
            chunk_query = """
            INSERT INTO knowledge_vector (knowledge_id, chunk_text, chunk_vector)
            VALUES (%s, %s, %s)
            """
            cursor.execute(chunk_query, (knowledge_id, chunk, chunk_embedding.tolist()))

        conn.commit()
        conn.close()
        return {"message": "Knowledge added successfully."}
    except Exception as e:
        print(f"Error adding knowledge: {e}")
        raise HTTPException(status_code=500, detail="Error adding knowledge.")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)