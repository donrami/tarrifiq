import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Configuration
EXCEL_FILE = "Tariff.xlsx" # Replace with your actual file name
DB_PATH = "./chroma_hs_db"
COLLECTION_NAME = "saso_hs_codes_gemini"

# 2. Custom Gemini Embedding Function
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        
    def __call__(self, input: list) -> list:
        import time
        from google.genai.errors import ClientError
        embeddings = []
        batch_size = 100
        for i in range(0, len(input), batch_size):
            batch = input[i:i+batch_size]
            retries = 3
            while retries > 0:
                try:
                    res = self.client.models.embed_content(
                        model='gemini-embedding-001',
                        contents=batch
                    )
                    embeddings.extend([e.values for e in res.embeddings])
                    # Avoid hitting 3000 RPM (requests per minute) limits:
                    # 3000 requests per 60 secs = 50 requests/sec. We'll add a tiny sleep.
                    time.sleep(0.5) 
                    break
                except ClientError as e:
                    if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                        print(f"Rate limited. Sleeping for 60 seconds...")
                        time.sleep(60)
                    else:
                        print(f"GenAI Error embedding batch {i}: {e}")
                        time.sleep(10)
                        retries -= 1
                except Exception as e:
                    print(f"General Error embedding batch {i}: {e}")
                    time.sleep(10)
                    retries -= 1
        return embeddings

# 3. Initialize ChromaDB (Persistent local storage)
client = chromadb.PersistentClient(path=DB_PATH)

gemini_ef = GeminiEmbeddingFunction(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Create or load the collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=gemini_ef
)

def ingest_data():
    print(f"Loading Excel file: {EXCEL_FILE}...")
    # Adjust 'HS_CODE', 'DESC_EN', 'DESC_AR' to match your actual Excel column headers
    df = pd.read_excel(EXCEL_FILE, dtype=str) 
    
    # Rename columns based on actual dataset headers
    df = df.rename(columns={
        'رمز النظام المنسق \n Harmonized Code': 'HS_CODE',
        'الصنف باللغة الانجليزية \n Item English Name': 'DESC_EN',
        'الصنف باللغة العربية \n Item Arabic Name': 'DESC_AR'
    })
    
    # Clean data: drop rows where essential fields are missing
    df = df.dropna(subset=['HS_CODE', 'DESC_EN']) 
    
    documents = []
    metadatas = []
    ids = []
    
    current_count = collection.count()
    print(f"Collection already has {current_count} documents. Skipping those...")
    df = df.iloc[current_count:]
    print(f"Remaining to ingest: {len(df)}")
    
    print("Preparing rows for vectorization. This might take a few minutes...")
    
    for index, row in df.iterrows():
        # The 'document' is what Chroma reads to find semantic meaning.
        # We combine English and Arabic so it can be searched in either language.
        combined_text = f"English: {row['DESC_EN']} | Arabic: {row.get('DESC_AR', '')}"
        
        documents.append(combined_text)
        
        # Metadata is what we get back (the actual code and isolated text)
        metadatas.append({
            "hs_code": str(row['HS_CODE']),
            "desc_en": str(row['DESC_EN']),
            "desc_ar": str(row.get('DESC_AR', ''))
        })
        
        # ID must be unique. The HS code itself is a good ID if there are no duplicates.
        # Prefixing with index to avoid accidental identical HS Code collision overwriting
        ids.append(f"{index}_{row['HS_CODE']}")

    # Batch insert into ChromaDB (Max 5461 per batch usually, but Chroma handles chunking in newer versions)
    # We slice it into chunks of 5000 just to be safe and avoid memory spikes
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        print(f"Inserting batch {i} to {i+batch_size}...")
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    print("Ingestion complete! Database is ready.")

def search_hs_code(query_text, top_k=5):
    print(f"\nSearching for: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    for i in range(len(results['ids'][0])):
        code = results['metadatas'][0][i]['hs_code']
        desc_en = results['metadatas'][0][i]['desc_en']
        distance = results['distances'][0][i] # Lower distance = better match
        print(f"[Match {i+1}] HS Code: {code} | Distance: {distance:.4f} | Desc: {desc_en[:80]}...")

# --- Execution ---
if __name__ == "__main__":
    # ONLY RUN THIS ONCE. Once ingested, comment it out.
    ingest_data() 
    
    # Test your query. Use an English translation of a German part.
    search_hs_code("industrial ball bearing for machinery")
    search_hs_code("stainless steel centrifugal water pump")
