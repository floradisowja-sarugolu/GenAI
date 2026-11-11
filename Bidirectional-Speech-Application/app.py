from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "success": False,
            "error": "Please enter or speak a question."
        }), 400

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful voice assistant. "
                            "Answer clearly and naturally using simple English. "
                            "Keep responses concise and easy to understand."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        result = response.json()

        ai_reply = result.get("message", {}).get(
            "content",
            "Sorry, I could not generate a response."
        )

        return jsonify({
            "success": True,
            "transcription": user_message,
            "response": ai_reply
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Ollama is not running. Please start Ollama and try again."
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "The AI response took too long. Please try again."
        }), 504

    except Exception as error:
        print("Error:", repr(error))
        return jsonify({
            "success": False,
            "error": "Something went wrong while processing your request."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)