
from flask import Flask, render_template, request, jsonify # type: ignore
import requests # pyright: ignore[reportMissingModuleSource]
import json
import yaml # type: ignore
import re
import time

app = Flask(__name__)

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

MAX_RETRIES = 2


def call_ollama(prompt):
    """Generate text using the local Ollama model."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()
    return result.get("response", "").strip()


def clean_output(text):
    """Remove Markdown code fences if the model includes them."""

    text = text.strip()

    # Remove opening code fence, including optional language label
    text = re.sub(
        r"^```(?:json|yaml|yml)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove closing code fence
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def validate_data(text, output_format):
    """Validate generated JSON or YAML using the appropriate parser."""

    try:
        if output_format == "json":
            parsed_data = json.loads(text)
            formatted_data = json.dumps(
                parsed_data,
                indent=4,
                ensure_ascii=False
            )

        else:
            parsed_data = yaml.safe_load(text)

            if parsed_data is None:
                raise ValueError("The YAML document is empty.")

            formatted_data = yaml.safe_dump(
                parsed_data,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False
            )

        if not isinstance(parsed_data, (dict, list)):
            raise ValueError(
                "The output must contain a JSON/YAML object or list."
            )

        return {
            "valid": True,
            "parsed_data": parsed_data,
            "formatted_data": formatted_data,
            "error": None
        }

    except (json.JSONDecodeError, yaml.YAMLError, ValueError) as error:
        return {
            "valid": False,
            "parsed_data": None,
            "formatted_data": text,
            "error": str(error)
        }


def build_prompt(description, output_format):
    """Build a format-specific structured data prompt."""

    format_name = output_format.upper()

    return f"""
You are a structured data generation assistant.

Convert the following dataset description into valid {format_name}.

DATASET DESCRIPTION:
{description}

STRICT REQUIREMENTS:
1. Return only the {format_name} data.
2. Do not include explanations or introductory sentences.
3. Use meaningful field names.
4. Preserve all details provided in the description.
5. Do not invent unnecessary records or facts.
6. Ensure all strings, numbers, lists, and objects use valid syntax.
7. The result must be parseable by a standard {format_name} parser.
"""


def build_refinement_prompt(description, output_format, invalid_output, error):
    """Ask Ollama to correct invalid structured output."""

    return f"""
You are an expert {output_format.upper()} syntax correction assistant.

The following generated output is invalid.

ORIGINAL DATASET DESCRIPTION:
{description}

REQUIRED FORMAT:
{output_format.upper()}

INVALID OUTPUT:
{invalid_output}

PARSER ERROR:
{error}

TASK:
Correct the syntax errors and produce valid {output_format.upper()}.

RULES:
- Preserve the original dataset information.
- Do not add explanations.
- Do not use Markdown code fences.
- Return only the corrected {output_format.upper()} data.
- Ensure it can be parsed by a standard parser.
"""


def generate_structured_data(description, output_format):
    """Generate, validate, and refine structured output."""

    prompt = build_prompt(description, output_format)

    attempts = []
    current_prompt = prompt
    final_result = None

    for attempt_number in range(1, MAX_RETRIES + 2):

        raw_output = call_ollama(current_prompt)
        cleaned_output = clean_output(raw_output)

        validation = validate_data(
            cleaned_output,
            output_format
        )

        attempts.append({
            "attempt": attempt_number,
            "valid": validation["valid"],
            "error": validation["error"],
            "output": cleaned_output
        })

        final_result = validation

        if validation["valid"]:
            break

        if attempt_number <= MAX_RETRIES:
            current_prompt = build_refinement_prompt(
                description,
                output_format,
                cleaned_output,
                validation["error"]
            )

    return {
        "success": final_result["valid"],
        "format": output_format.upper(),
        "raw_output": attempts[-1]["output"],
        "formatted_output": final_result["formatted_data"],
        "parsed_data": final_result["parsed_data"],
        "error": final_result["error"],
        "attempts": attempts,
        "refined": len(attempts) > 1 and final_result["valid"]
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():

    data = request.get_json(silent=True) or {}

    description = data.get("description", "").strip()
    output_format = data.get("format", "json").lower()

    if len(description) < 5:
        return jsonify({
            "error": "Please enter a dataset description."
        }), 400

    if output_format not in ["json", "yaml"]:
        return jsonify({
            "error": "Select either JSON or YAML."
        }), 400

    try:
        start_time = time.time()

        result = generate_structured_data(
            description,
            output_format
        )

        elapsed = round(time.time() - start_time, 2)

        result["description"] = description
        result["model"] = MODEL_NAME
        result["time"] = elapsed

        return jsonify(result)

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": (
                "Cannot connect to Ollama. "
                "Please make sure Ollama is running."
            )
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "error": (
                "Ollama took too long to respond. "
                "Please try again."
            )
        }), 504

    except requests.exceptions.HTTPError as error:
        return jsonify({
            "error": f"Ollama HTTP error: {str(error)}"
        }), 502

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    print("======================================")
    print(" EXPERIMENT 9 - JSON/YAML GENERATION")
    print("======================================")
    print("Model:", MODEL_NAME)
    print("Open: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
