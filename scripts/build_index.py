"""Embed long_offering texts with sentence-transformers and build a FAISS index."""

import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import COMPANIES_CSV_PATH, COMPANIES_META_PATH, EMBEDDING_DIM, EMBEDDING_MODEL, FAISS_INDEX_PATH


def main() -> None:
    if not COMPANIES_CSV_PATH.exists():
        print(f"ERROR: {COMPANIES_CSV_PATH} not found. Run prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv(COMPANIES_CSV_PATH)
    print(f"Loaded {len(df)} companies from CSV")

    # Load embedding model (downloads ~130MB on first run)
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed all long_offering texts
    texts = df["long_offering"].tolist()
    print(f"Embedding {len(texts)} texts...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    vectors = np.array(vectors, dtype=np.float32)

    print(f"Vectors shape: {vectors.shape}")

    # Build FAISS index (Inner Product = cosine similarity for normalised vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    print(f"FAISS index built: {index.ntotal} vectors, dim={EMBEDDING_DIM}")

    # Save index
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Index saved to {FAISS_INDEX_PATH}")

    # Save metadata (parallel to FAISS row indices)
    meta = []
    for _, row in df.iterrows():
        meta.append({
            "id": str(row["id"]),
            "company_name": row["company_name"],
            "country": row["country"],
            "long_offering": row["long_offering"],
        })

    with open(COMPANIES_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"Metadata saved to {COMPANIES_META_PATH} ({len(meta)} entries)")


if __name__ == "__main__":
    main()
