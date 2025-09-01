from langchain.tools import BaseTool
from services.query_service import answer_question
import json


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
        "Use this tool to answer questions strictly from the provided vector database (RAG). "
        "the Input to this tool must be a JSON object in stringified form with two keys: 'path' and 'question'. "
        "- 'path': string, the file path to the vector database. "
        "- 'question': string, the user query to be answered. "
        "Example input: "
        "{'path': './storage/vectordb/weather/karachi_4/faiss/7adb2109d14d4581b742f307957b1c49', "
        "'question': 'weather in Karachi for the next 4 days'}"
        "if any of the key value is missing then pass the value exactly as "
        "'nan' for that key, for example if value for 'path' is missing"
        " then the input to the tool shoud be like {'path': 'nan', 'question': 'question string'} "
    )

    def _run(self, tool_input: str) -> str:
        print("siuuuuu tool triggered", tool_input)
        data = tool_input.strip("'")
        data = json.loads(data)
        if data['path'] == 'nan':
            return "path missing"
        if data['question'] == 'nan':
            return "question missing"
        print("\n\n\n\nthis si the path:", data['path'])
        print("\n\n\n\nthis si the question:", data['question'])

        path = data['path']
        question = data['question']

        if not path:
            return ("No vector DB path provided. "
                    "Please add PATH=... or configure default.")
        if not question:
            return "Invalid input. Provide QUESTION=..."

        # Optional: strip surrounding quotes if a tool added them
        if (question.startswith('"') and question.endswith('"')) or (question.startswith("'") and question.endswith("'")):
            question = question[1:-1].strip()
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1].strip()

        result = answer_question(vector_db_path=path, question=question)
        return (result.get("answer", "") or
                "I don't know based on the provided documents.")
