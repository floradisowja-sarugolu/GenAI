
from flask import Flask, render_template, request, jsonify # type: ignore
import requests # type: ignore
import time

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def ask_ollama(prompt):
    """Send a prompt to the locally running Ollama model."""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=300
    )

    response.raise_for_status()
    result = response.json()

    return result.get("response", "").strip()


def run_direct(problem):
    prompt = f"""
Solve the following problem directly.

Give a clear final answer with a short explanation.
Do not invent missing information.

Problem:
{problem}
"""
    return ask_ollama(prompt)


def run_cot(problem):
    prompt = f"""
Solve the following problem using a clear, step-by-step approach.

Show the important calculation or logical steps in a concise,
user-facing explanation. Then provide the final answer.

If information is missing, state your assumption.

Problem:
{problem}
"""
    return ask_ollama(prompt)


def run_decomposition(problem):
    # Stage 1: Break the problem into smaller questions.
    planning_prompt = f"""
You are a task decomposition assistant.

Break the following problem into 3 to 5 small, logical,
sequential sub-questions.

Rules:
- Each sub-question must help solve the original problem.
- Keep the questions specific and easy to answer.
- Do not solve the original problem yet.
- Return only a numbered list of sub-questions.

Original problem:
{problem}
"""

    subquestions = ask_ollama(planning_prompt)

    # Stage 2: Solve each sub-question and combine the results.
    solving_prompt = f"""
Solve the original problem using the following decomposition.

Original problem:
{problem}

Sub-questions:
{subquestions}

Instructions:
1. Answer each sub-question separately.
2. Label each answer clearly.
3. Use the intermediate answers to solve the original problem.
4. Give a final combined answer.
5. Keep the explanation clear and concise.
6. Do not invent missing information.

Use this format:

SUB-QUESTION RESULTS:
1. Question:
   Answer:

2. Question:
   Answer:

FINAL COMBINED ANSWER:
...
"""

    combined_result = ask_ollama(solving_prompt)

    return {
        "subquestions": subquestions,
        "combined_result": combined_result
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/solve", methods=["POST"])
def solve():
    data = request.get_json(silent=True) or {}
    problem = data.get("problem", "").strip()

    if not problem:
        return jsonify({
            "success": False,
            "error": "Please enter a problem first."
        }), 400

    start_time = time.time()

    try:
        # 1. Direct-answer baseline
        direct_result = run_direct(problem)

        # 2. Zero-shot Chain-of-Thought approach
        cot_result = run_cot(problem)

        # 3. Task decomposition approach
        decomposition_result = run_decomposition(problem)

        elapsed_time = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "model": MODEL_NAME,
            "elapsed_time": elapsed_time,
            "direct": direct_result,
            "cot": cot_result,
            "subquestions": decomposition_result["subquestions"],
            "decomposed": decomposition_result["combined_result"]
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Cannot connect to Ollama. Open Ollama and try again."
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Ollama took too long to respond. Try a shorter problem."
        }), 504

    except requests.exceptions.HTTPError as error:
        return jsonify({
            "success": False,
            "error": f"Ollama returned an HTTP error: {error}"
        }), 502

    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
