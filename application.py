import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from services.ingest_service import build_vector_db_from_json
from services.query_service import answer_question
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from services.agent.tools.age_tool import AgeCalculatorTool
from services.agent.tools.weather_tool import weather_tool
from services.agent.tools.retrievalqa_tool import RetrievalQATool
from prompt import prompt_template
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools.base import ToolException
from langchain.callbacks.base import BaseCallbackHandler

from langchain import hub

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
        "pdf_paths": ["absolute/or/relative/path1.pdf", "path2.pdf", ...]
    }
    Returns:
    {
        "vector_db_path": "...",
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        pdf_paths = data.get("pdf_paths")
        storage_dir = "./storage/vectordb/weather"

        if not pdf_paths:
            return jsonify({"error": "pdf_paths (list) is required"}), 400

        # Ensure base storage dir exists
        os.makedirs(storage_dir, exist_ok=True)

        result = build_vector_db_from_json(
            pdf_paths=pdf_paths,
            storage_dir=storage_dir,
            embed_model="text-embedding-3-small",
            chunk_size=1000,
            chunk_overlap=150
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
    Accepts either:
      1) {"messages": [{"role":"user|assistant|system","content":"..."} , ...]}
      2) {"message": "single user message"}   # backward compatible

    Returns:
      {
        "reply": "<assistant reply>",
        "append_this_message": {
            "role": "assistant",
            "content": "<assistant reply>"
        }
      }
    """
    data = request.get_json(silent=True) or {}

    # ---- Accept transcript or single message
    messages = data.get("messages")
    # ---- Validate last turn is a user message
    last = messages[-1]
    if last.get("role") != "user" or not (last.get("content") or "").strip():
        return jsonify({
            "error": (
                "Last item in `messages` must be a user message with "
                "non-empty `content`"
            )
        }), 400

    history_msgs = messages[:-1]                 # seed memory from this
    current_user_input = last["content"].strip()  # run agent on this

    tools = [
        AgeCalculatorTool(handle_tool_error=False),
        RetrievalQATool(),
        weather_tool(),
    ]

    # ---- LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,
        output_key="output"
    )
    memory.chat_memory.clear()

    # ---- Preload memory from client transcript (except last user turn)
    role_map = {
        "user": HumanMessage,
        "assistant": AIMessage,
        "system": SystemMessage,
    }
    # Seed memory with paired user→assistant turns only
    buffered_user_input = None
    for m in history_msgs:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content or role not in role_map:
            continue

        if role == "user":
            buffered_user_input = content
        elif role == "assistant" and buffered_user_input is not None:
            # Save as one turn: user input → assistant output
            memory.save_context(
                {"input": buffered_user_input},
                {"output": content}
            )
            # Debug: confirm the pair was saved
            print("Seed pair -> input:", buffered_user_input)
            print("Seed pair -> output:", content)
            buffered_user_input = None
        else:
            # Ignore system or unmatched assistant messages
            continue
    print("BRRRRRRRRRRRRRRRRR\n\n\n", memory.load_memory_variables({}))

    # ---- Build agent with your custom prompt
    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    # print("\n\n\n\n\nPROMPT\n\n\n\n")
    # print(agent_executor.agent)

    try:
        # Only pass the current user input
        # Memory injects history into {chat_history}
        # Debug: show what the memory will inject
        _mem = memory.load_memory_variables({})
        _hist = (_mem.get("chat_history") or "")
        print("chat_history preview (first 500 chars):")
        print(_hist[:500])
        result = agent_executor.invoke({"input": current_user_input})
        reply = (result.get("output") or "").strip()
    except ToolException as e:
        reply = str(e)
    except Exception as e:
        # Surface unexpected errors
        return jsonify({"error": f"Agent error: {e}"}), 500

    return jsonify({
        "reply": reply,
        "append_this_message": {"role": "assistant", "content": reply},
    }), 200


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8000, debug=True)
