import os
import uuid
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_pdfs(pdf_paths: List[str]):
    """Load PDFs into LangChain Documents."""
    docs = []
    for p in pdf_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"PDF not found: {p}")
        loader = PyPDFLoader(p)
        # Each element is a Document with metadata={"source": path, "page": n}
        docs.extend(loader.load())
    return docs


def _chunk_docs(docs, chunk_size=1000, chunk_overlap=150):
    """Chunk documents for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_db(
    pdf_paths: List[str],
    backend: str | None,
    storage_dir: str,
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Dict:
    """
    Ingest PDFs → chunk → embed (OpenAI) → persist FAISS/Chroma.
    Returns info dict with where the index was saved.
    """
    if not pdf_paths or not isinstance(pdf_paths, list):
        raise ValueError("pdf_paths must be a non-empty list of file paths")

    backend = (backend or "faiss").lower()
    if backend not in {"faiss", "chroma"}:
        raise ValueError("backend must be 'faiss' or 'chroma'")

    # Load + split
    docs = _load_pdfs(pdf_paths)
    chunks = _chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Embeddings
    embeddings = OpenAIEmbeddings(model=embed_model)

    # Run ID for a new collection/index
    run_id = uuid.uuid4().hex
    base_dir = os.path.join(storage_dir, backend, run_id)
    _ensure_dir(base_dir)

    # Persist
    if backend == "faiss":
        vs = FAISS.from_documents(chunks, embeddings)
        # FAISS saves two files in base_dir
        vs.save_local(base_dir)
    else:  # chroma
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=base_dir,
        )
        vs.persist()

    return {
        "backend": backend,
        "vector_db_path": base_dir,
        "num_chunks": len(chunks),
        "num_docs": len(docs),
        "run_id": run_id,
    }
