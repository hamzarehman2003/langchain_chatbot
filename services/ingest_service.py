import os, json, uuid
from typing import Any, Dict, List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def _collect_strings(obj: Any, out: List[str]) -> None:
    if isinstance(obj, str):
        s = obj.strip()
        if s: out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values(): _collect_strings(v, out)
    elif isinstance(obj, list):
        for v in obj: _collect_strings(v, out)
    # other types are ignored


def build_vector_db_from_json(
    pdf_paths: str,
    storage_dir: str,
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Dict[str, Any]:
    with open(pdf_paths, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

    # 1) Pull ALL text from the JSON (recursively)
    texts: List[str] = []
    _collect_strings(data, texts)

    # 2) Make a single Document (or split per item if you prefer)
    doc = Document(page_content="\n".join(texts), metadata={"source": pdf_paths})

    # 3) Chunk → embed → save FAISS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents([doc])

    embeddings = OpenAIEmbeddings(model=embed_model)

    run_id = uuid.uuid4().hex
    base_dir = os.path.join(storage_dir, "faiss", run_id)
    os.makedirs(base_dir, exist_ok=True)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(base_dir)

    return {
        "backend": "faiss",
        "vector_db_path": base_dir,
        "num_chunks": len(chunks),
        "num_docs": 1,
        "run_id": run_id,
    }
