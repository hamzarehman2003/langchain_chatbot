import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from services.ingest_service import build_vector_db
from services.query_service import answer_question
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from services.agent.tools.age_tool import AgeCalculatorTool
from services.agent.tools.retrievalqa_tool import RetrievalQATool
from prompt import prompt_template


load_dotenv()

application = Flask(__name__)
API_KEY = os.getenv("API", None)


@application.before_request
def api_key():

    provided = request.headers.get("API-Key")
    if not provided or provided != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    return None


@application.get("/")
def home_route():
    return "Home Route", 200


@application.post("/create_embeddings")
def create_embeddings():
    """
    Body:
    {
        "pdf_paths": ["absolute/or/relative/path1.pdf", "path2.pdf", ...],
        "backend": "faiss" | "chroma" 
    }
    Returns:
    {
        "backend": "...",
        "vector_db_path": "...",
        "num_chunks": 123,
        "num_docs": 4,
        "run_id": "..."
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        pdf_paths = data.get("pdf_paths")
        backend = data.get("backend") or os.getenv("VECTOR_DB", "faiss")
        storage_dir = os.getenv("STORAGE_DIR", "./storage/vectordb")

        if not pdf_paths:
            return jsonify({"error": "pdf_paths (list) is required"}), 400

        # Ensure base storage dir exists
        os.makedirs(storage_dir, exist_ok=True)

        result = build_vector_db(
            pdf_paths=pdf_paths,
            backend=backend,
            storage_dir=storage_dir,
            embed_model="text-embedding-3-small",
            chunk_size=1000,
            chunk_overlap=150,
        )
        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Don't leak internals in prod; okay during dev
        return jsonify({
            "error": "Internal Server Error",
            "detail": str(e)
        }), 500


@application.post("/query")
def query():
    """
    Body:
    {
        "vector_db_path": "<path returned by /embeddings/create>",
        "question": "Your question"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        vector_db_path = data.get("vector_db_path")
        question = data.get("question")

        if not vector_db_path:
            return jsonify({"error": "vector_db_path is required"}), 400
        if not question:
            return jsonify({"error": "question is required"}), 400

        result = answer_question(
            vector_db_path=vector_db_path,
            question=question,
        )
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        # During dev it's useful to see detail; in prod you might hide it.
        return jsonify({
            "error": "Internal Server Error",
            "detail": str(e)
        }), 500


@application.post("/agent")
def agent_route():
    """
    Body:
    {
        "message": "user question here",
        "vector_db_path": "./storage/vectordb/faiss/<run_id>",
        "force_age": false
    }
    """
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message is required"}), 400

    default_path = (
        "./storage/vectordb/faiss/"
        "221c3582683c440d9998a8ca0e73aab4"
    )
    tools = [
        AgeCalculatorTool(),
        RetrievalQATool(default_vector_db_path=default_path),
    ]

    prompt = prompt_template
    print("this is the prompt:", prompt)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
    print("AGENT EXECUTOR\n\n\n\n\n")
    print(agent_executor.agent)
    print("Ended")

    result = agent_executor.invoke({"input": msg})
    return jsonify({
        "message": msg,
        "reply": result.get("output", ""),
    }), 200


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8000, debug=True)
