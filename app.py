from flask import Flask, render_template, request, jsonify
from agent import AIAgent
import os
from dotenv import load_dotenv
load_dotenv()  # this loads values from .env


app = Flask(__name__)
print("API KEY:", os.getenv("OPENROUTER_API_KEY"))
print("MODEL:", os.getenv("LLM_MODEL"))
print("BASE URL:", os.getenv("LLM_BASE_URL"))

# Check if the API key is in the expected format
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key:
    print("API Key length:", len(api_key))
    print("API Key starts with:", api_key[:10] + "...")
    
agent = AIAgent()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json(force=True)

        query_text = data.get("query", "").strip()
        language = data.get("language", "en")

        profile = {
            "age": int(data.get("age", 0)) if str(data.get("age")).isdigit() else 0,
            "income": int(data.get("income", 0)) if str(data.get("income")).isdigit() else 0,
            "landholding": float(data.get("landholding", 0)) if str(data.get("landholding")).replace('.','',1).isdigit() else 0,
            "occupation": data.get("occupation", ""),
            "caste": data.get("caste", ""),
            "residence": data.get("residence", "rural")
        }

        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        response = agent.process_query(query_text, profile, language)
        return jsonify(response)

    except Exception as e:
        print("Error in /query:", e)
        return jsonify({"error": "Error processing your query"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")
    language = data.get("language", "en")

    try:
        # Ask GenAI via OpenRouter
        answer = agent.ask_genai(user_query, language=language)
        return jsonify({"answer": answer})
    except Exception as e:
        print("Chatbot error:", str(e))
        return jsonify({"answer": "⚠️ Server error while processing your query."})


if __name__ == "__main__":
    # debug=True for hackathon development; remove for production
    app.run(host="0.0.0.0", port=5000, debug=True)
