from langchain.tools import BaseTool
from services.query_service import answer_question
from typing import Optional


class RetrievalQATool(BaseTool):
    """
    ReAct-friendly retrieval tool with default path support.
    You can pass `default_vector_db_path` in the constructor.

    INPUT FORMAT:
      Either:
        PATH=<vector_db_path>
        QUESTION=<your question>
      Or (if default path set at init):
        QUESTION=<your question>
    """
    name: str = "retrieval_qa"
    description: str = (
        "Use this tool for any informational/specific question that must be answered strictly "
        "from the provided vector database (RAG). . "
    )
    default_vector_db_path: Optional[str] = None

    def __init__(self, default_vector_db_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.default_vector_db_path = default_vector_db_path

    def _run(self, tool_input: str) -> str:
        text = (tool_input or "").strip()
        if not text:
            return ("Missing input. Provide QUESTION=... "
                    "(and PATH=... if no default path is set).")

        path = None
        question = None

        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("PATH="):
                path = line.split("=", 1)[1].strip()
            elif line.upper().startswith("QUESTION="):
                question = line.split("=", 1)[1].strip()

        # fallback to default
        path = path or self.default_vector_db_path

        if not path:
            return ("No vector DB path provided. "
                    "Please add PATH=... or configure default.")

        if not question:
            return "Invalid input. Provide QUESTION=..."

        result = answer_question(vector_db_path=path, question=question)
        return (result.get("answer", "") or
                "I don't know based on the provided documents.")
