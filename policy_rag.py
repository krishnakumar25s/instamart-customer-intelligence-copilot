
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3


def load_policy_documents():
    """Read all policy text files and return their names and contents."""
    documents = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents[path.name] = text

    if not documents:
        raise FileNotFoundError(
            f"No non-empty .txt policy documents found in {POLICY_DIR}"
        )

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into word chunks with the requested overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    start = 0

    while start < len(words):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        # Stop when this chunk reaches or passes the final word.
        if start + chunk_size >= len(words):
            break

        start += step

    return chunks


def build_policy_index():
    """Chunk the policies and fit a TF-IDF index."""
    documents = load_policy_documents()
    chunks = []

    for source, text in documents.items():
        for chunk_number, chunk in enumerate(chunk_text(text), start=1):
            chunks.append({
                "source": source,
                "chunk_number": chunk_number,
                "text": chunk,
            })

    if not chunks:
        raise ValueError("No policy chunks were created.")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return documents, chunks, vectorizer, matrix


def retrieve(query, k=TOP_K):
    """Return the k most relevant policy chunks for a query."""
    if not query or not query.strip():
        return []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:min(k, len(chunks))]

    results = []

    for index in ranked_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"],
        })

    return results


# Build the index when this module is imported.
documents, chunks, vectorizer, matrix = build_policy_index()


if __name__ == "__main__":
    print("Policy RAG index created successfully.")
    print(f"Policy documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    print("\nChunks per policy:")
    counts = {}

    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")

    print("\nSample retrieval:")
    query = "What refund applies to an order delivered late?"
    print(f"Query: {query}")

    for result in retrieve(query, k=3):
        print(
            f"\nSource: {result['source']}"
            f"\nScore: {result['score']:.4f}"
            f"\nText: {result['text']}"
        )
