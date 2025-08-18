import os
from typing import Dict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def _build_prompt() -> PromptTemplate:
    """
    RetrievalQA expects a prompt with {context} and {question}.
    Keep it short and grounded.
    """
    template = (
        "You are a helpful assistant. Use the context to answer the "
        "question.\n"
        "If the user asks a specific question that requires context and\n"
        "the answer is not found in the context, say I don't know.\n\n"
        "If the user asks a general question (e.g., greetings, about "
        "yourself, casual queries), answer it normally even if it is not "
        "in the context.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    return PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )


def answer_question(
    *,
    vector_db_path: str,
    question: str,
) -> Dict:
    """
    Load the FAISS vector DB at vector_db_path and answer the question using 
    RetrievalQA. Returns {answer}.
    """
    if not vector_db_path or not os.path.isdir(vector_db_path):
        raise ValueError("vector_db_path must be an existing directory")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    # Embeddings (must match ingest model)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Load FAISS vector store
    vectordb = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True  # required in recent LangChain
    )

    # Retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )

    # Prompt
    prompt = _build_prompt()

    # Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa.invoke({"query": question})
    answer: str = result.get("result", "")

    return {
        "answer": answer,
    }