
import ast
import json
import operator
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

# ---------------- SAFE CALCULATOR ----------------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression):
    """Evaluate basic arithmetic without using Python eval()."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid mathematical expression.")

    def calculate(node):
        if isinstance(node, ast.Expression):
            return calculate(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left = calculate(node.left)
            right = calculate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = OPERATORS[type(node.op)](left, right)

            if abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](calculate(node.operand))

        raise ValueError("Unsupported or unsafe expression.")

    try:
        return calculate(tree)
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")


def calculator_tool(expression):
    return str(safe_eval(expression))


# ---------------- UNIT CONVERTER ----------------

UNIT_GROUPS = {
    "length": {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1,
        "km": 1000,
        "inch": 0.0254,
        "in": 0.0254,
        "ft": 0.3048,
        "mile": 1609.344,
    },
    "weight": {
        "mg": 0.001,
        "g": 1,
        "kg": 1000,
        "tonne": 1000000,
        "lb": 453.59237,
        "lbs": 453.59237,
    },
    "time": {
        "sec": 1,
        "s": 1,
        "min": 60,
        "hr": 3600,
        "hour": 3600,
        "day": 86400,
    },
}


def unit_converter_tool(query):
    pattern = (
        r"^\s*(-?\d+(?:\.\d+)?)\s*"
        r"([a-zA-Z]+)\s+(?:to|in)\s+([a-zA-Z]+)\s*$"
    )
    match = re.match(pattern, query, re.IGNORECASE)

    if not match:
        raise ValueError(
            "Use a format like '10 km to m' or '500 g to kg'."
        )

    value = float(match.group(1))
    from_unit = match.group(2).lower()
    to_unit = match.group(3).lower()

    for group in UNIT_GROUPS.values():
        if from_unit in group and to_unit in group:
            result = value * group[from_unit] / group[to_unit]
            return f"{value:g} {from_unit} = {result:g} {to_unit}"

    raise ValueError("These units are not supported or are incompatible.")


# ---------------- DATE AND TIME TOOL ----------------

def datetime_tool(query):
    now = datetime.now().astimezone()

    return (
        f"Current date: {now.strftime('%A, %d %B %Y')}\n"
        f"Current time: {now.strftime('%I:%M:%S %p %Z')}"
    )


# ---------------- TOOL REGISTRY ----------------

TOOLS = {
    "calculator": calculator_tool,
    "unit_converter": unit_converter_tool,
    "datetime": datetime_tool,
}


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ---------------- AGENT DECISION ----------------

def choose_tool(user_query):
    prompt = f"""
You are the tool-selection controller for an AI agent.

Choose exactly one tool only when it is needed.

Available tools:
1. calculator: arithmetic, percentages, and mathematical calculations.
2. unit_converter: convert length, weight, or time units.
3. datetime: current date or current time.
4. none: general questions that do not need these tools.

Return ONLY one valid JSON object in this format:
{{"tool": "calculator", "input": "25 * 4"}}

For a general question:
{{"tool": "none", "input": ""}}

Rules:
- Do not calculate the answer yourself.
- For calculator, put only a mathematical expression in input.
- For unit_converter, use a format like "10 km to m".
- For datetime, input can be the user's original question.
- Never select a tool that is not listed.

User question: {user_query}

JSON response:
"""

    raw = ask_ollama(prompt)

    # Extract JSON even if the model adds code fences or extra text.
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return {"tool": "none", "input": ""}

    try:
        decision = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"tool": "none", "input": ""}

    tool_name = decision.get("tool")
    tool_input = decision.get("input", "")

    if tool_name not in list(TOOLS.keys()) + ["none"]:
        return {"tool": "none", "input": ""}

    if not isinstance(tool_input, str):
        return {"tool": "none", "input": ""}

    return {"tool": tool_name, "input": tool_input}


# ---------------- AGENT EXECUTION ----------------

def run_agent(user_query):
    decision = choose_tool(user_query)

    tool_name = decision["tool"]
    tool_input = decision["input"]

    if tool_name == "none":
        answer = ask_ollama(
            "Answer the following user question clearly and accurately:\n"
            + user_query
        )
        return {
            "tool": "None",
            "tool_input": "",
            "tool_output": "",
            "answer": answer,
        }

    try:
        tool_output = TOOLS[tool_name](tool_input)

        answer = ask_ollama(
            f"""
The user asked: {user_query}

The tool used was: {tool_name}
The tool result was:
{tool_output}

Explain the result to the user in a clear, concise way.
Do not change the tool result or invent additional calculations.
"""
        )

        return {
            "tool": tool_name,
            "tool_input": tool_input,
            "tool_output": str(tool_output),
            "answer": answer,
        }

    except Exception as error:
        return {
            "tool": tool_name,
            "tool_input": tool_input,
            "tool_output": "",
            "answer": f"Tool execution failed: {error}",
        }


# ---------------- FLASK ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        result = run_agent(user_query)
        return jsonify(result)

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Cannot connect to Ollama. Start Ollama and try again."
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Ollama took too long to respond. Please try again."
        }), 504

    except requests.exceptions.RequestException as error:
        return jsonify({"error": f"Ollama request failed: {error}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
