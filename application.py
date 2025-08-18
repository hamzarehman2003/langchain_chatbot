import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from services.ingest_service import build_vector_db
from services.query_service import answer_question


def create_app():
    load_dotenv()

    application = Flask(__name__)
    API_KEY = os.getenv("API", None)
    #OPEN_ENDPOINTS = {"/"}

    @application.before_request
    def api_key():
        # if request.path in OPEN_ENDPOINTS:
        #     return None

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
            # Don’t leak internals in prod; okay during dev
            return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500

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
            return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500

    return application


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=8000, debug=True)