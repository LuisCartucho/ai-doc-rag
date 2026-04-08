from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_collection("docs")

def query(q):
    embedding = model.encode([q]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=3
    )

    return results["documents"]

if __name__ == "__main__":
    q = "What is the return policy?"
    results = query(q)

    for r in results[0]:
        print("\n---\n", r)