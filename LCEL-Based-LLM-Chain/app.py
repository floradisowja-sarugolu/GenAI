
from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import time

app = Flask(__name__)

MODEL_NAME = "llama3.2:3b"

# Initialize the local Ollama model
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2
)

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Summarize the following text in a clear and concise way.

    Include the main ideas and important information.
    Do not add information that is not present in the text.

    Text:
    {text}
    """
)

# Convert the model response into plain text
output_parser = StrOutputParser()

# Build the LCEL chain
chain = prompt | llm | output_parser


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/summarize", methods=["POST"])
def summarize():

    data = request.get_json(silent=True) or {}
    user_text = data.get("text", "").strip()

    if not user_text:
        return jsonify({
            "success": False,
            "error": "Please enter some text to summarize."
        }), 400

    start_time = time.time()

    try:
        # Execute the complete LCEL chain
        result = chain.invoke({
            "text": user_text
        })

        elapsed_time = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "result": result,
            "model": MODEL_NAME,
            "elapsed_time": elapsed_time,
            "chain": "Prompt Template → ChatOllama → StrOutputParser"
        })

    except Exception as error:

        error_message = str(error)

        if "Connection refused" in error_message or "actively refused" in error_message:
            error_message = (
                "Cannot connect to Ollama. Please make sure Ollama is running."
            )

        return jsonify({
            "success": False,
            "error": error_message
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
