
import os
import time

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from google import genai

# ------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please check your .env file."
    )

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.6-flash"


# ------------------------------------------
# 2. GEMINI RESPONSE GENERATION
# ------------------------------------------

def generate_response(prompt, max_tokens=600):

    models_to_try = [MODEL_NAME, FALLBACK_MODEL]
    last_error = None

    for model in models_to_try:

        for attempt in range(3):

            try:
                print(
                    f"Trying {model}, attempt {attempt + 1}"
                )

                result = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "max_output_tokens": max_tokens
                    }
                )

                if result.text:
                    return result.text

                raise Exception("Gemini returned an empty response.")

            except Exception as error:

                last_error = error
                error_text = str(error).lower()

                transient_errors = [
                    "503",
                    "unavailable",
                    "429",
                    "500",
                    "502",
                    "504",
                    "timeout",
                    "timed out",
                    "server disconnected",
                    "connection reset",
                    "connection error"
                ]

                is_transient = any(
                    message in error_text
                    for message in transient_errors
                )

                if is_transient:

                    wait_time = 2 ** (attempt + 1)

                    print(
                        f"Temporary API error: {error}"
                    )

                    if attempt < 2:
                        print(
                            f"Retrying in {wait_time} seconds..."
                        )
                        time.sleep(wait_time)

                else:
                    raise

    raise Exception(
        f"Gemini could not generate a response. "
        f"Last error: {last_error}"
    )


# ------------------------------------------
# 3. RESPONSE COMPARISON AND EVALUATION
# ------------------------------------------

def compare_responses(
    baseline_prompt,
    baseline_response,
    enhanced_prompt,
    enhanced_response
):

    evaluation_prompt = f"""
You are an expert evaluator of Generative AI prompt engineering.

Compare the following two AI-generated responses.

BASELINE PROMPT:
{baseline_prompt}

BASELINE RESPONSE:
{baseline_response}

ENHANCED PROMPT:
{enhanced_prompt}

ENHANCED RESPONSE:
{enhanced_response}

Evaluate both responses using these criteria:

1. Relevance to the topic.
2. Clarity and readability.
3. Completeness of information.
4. Organization and structure.
5. Following the instructions in the prompt.

Provide your evaluation in the following format:

BASELINE PROMPT EVALUATION:
Explain the strengths and weaknesses.

ENHANCED PROMPT EVALUATION:
Explain the improvements and any remaining weaknesses.

KEY DIFFERENCES:
Explain the major differences between the two responses.

WINNER:
Identify which response better follows its prompt and explain why.

CONCLUSION:
Explain how improved prompt instructions affected the output.

Use simple, clear English.
Do not invent facts about the subject.
"""

    evaluation = generate_response(
        evaluation_prompt,
        max_tokens=600
    )

    return evaluation


# ------------------------------------------
# 4. HOME PAGE
# ------------------------------------------

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


# ------------------------------------------
# 5. COMPARE PROMPTS ROUTE
# ------------------------------------------

@app.route("/compare", methods=["POST"])
def compare():

    start_time = time.time()

    try:

        data = request.get_json(silent=True) or {}

        topic = data.get("topic", "").strip()

        if not topic:
            return jsonify({
                "error": "Please enter a topic."
            }), 400

        # Baseline prompt
        baseline_prompt = (
            f"Write a one-paragraph bio of {topic}."
        )

        # Enhanced prompt
        enhanced_prompt = f"""
You are an expert biography writer and history researcher.

Write a one-paragraph biography of {topic}.

Requirements:
- Include background and important achievements.
- Mention major contributions and historical importance.
- Use clear, accurate and simple English.
- Keep the paragraph between 100 and 150 words.
- Maintain a professional and informative tone.
- Do not use bullet points.
- Return only one well-structured paragraph.
"""

        # Generate baseline response
        print("\nGenerating baseline response...")

        baseline_response = generate_response(
            baseline_prompt,
            max_tokens=400
        )

        print("Baseline response generated successfully.")

        # Generate enhanced response
        print("\nGenerating enhanced response...")

        enhanced_response = generate_response(
            enhanced_prompt,
            max_tokens=500
        )

        print("Enhanced response generated successfully.")

        # Evaluate both responses
        print("\nStarting evaluation...")

        evaluation = compare_responses(
            baseline_prompt,
            baseline_response,
            enhanced_prompt,
            enhanced_response
        )

        print("Evaluation completed successfully.")

        # Calculate execution time
        elapsed_time = round(
            time.time() - start_time, 2
        )

        print(
            f"Comparison completed in {elapsed_time} seconds."
        )

        # Send results to frontend
        return jsonify({

            "baseline_prompt": baseline_prompt,
            "baseline_response": baseline_response,

            "enhanced_prompt": enhanced_prompt,
            "enhanced_response": enhanced_response,

            "evaluation": evaluation,

            "time_taken": elapsed_time

        })

    except Exception as error:

        print("\nGemini API error:", str(error))

        return jsonify({
            "error": str(error)
        }), 500


# ------------------------------------------
# 6. FRONTEND HTML, CSS AND JAVASCRIPT
# ------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Experiment 2 | Prompt Comparison</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    background: #f3f5f9;
    color: #202938;
    padding: 30px 16px;
}

.container {
    max-width: 1050px;
    margin: auto;
}

header {
    text-align: center;
    margin-bottom: 28px;
}

header h1 {
    font-size: 29px;
    margin-bottom: 9px;
}

header p {
    color: #657187;
    line-height: 1.5;
}

.input-panel, .result-card {
    background: white;
    border: 1px solid #e1e6ee;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
}

label {
    display: block;
    font-weight: bold;
    margin-bottom: 10px;
}

input {
    width: 100%;
    padding: 13px;
    border: 1px solid #cbd3df;
    border-radius: 8px;
    font-size: 15px;
    outline: none;
}

input:focus {
    border-color: #526fe5;
}

button {
    margin-top: 15px;
    padding: 13px 22px;
    background: #405bd5;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    cursor: pointer;
}

button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.status {
    margin-top: 14px;
    color: #526fe5;
    font-size: 14px;
}

.results {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
}

.result-card h2 {
    font-size: 19px;
    margin-bottom: 14px;
}

.result-card h3 {
    font-size: 14px;
    margin: 15px 0 7px;
    color: #526fe5;
}

.prompt {
    background: #f2f4fa;
    padding: 12px;
    border-radius: 7px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.response {
    line-height: 1.7;
    font-size: 14px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.evaluation {
    line-height: 1.8;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.full-width {
    grid-column: 1 / -1;
}

.hidden {
    display: none;
}

.error {
    color: #c0392b;
    margin-top: 12px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

@media (max-width: 700px) {

    .results {
        grid-template-columns: 1fr;
    }

    .full-width {
        grid-column: auto;
    }

}

</style>

</head>

<body>

<div class="container">

    <header>

        <h1>Prompt Comparison</h1>


        <p>
            Compare a simple prompt with an improved prompt.
        </p>

    </header>


    <section class="input-panel">

        <label for="topic">
            Enter a person or topic
        </label>

        <input
            type="text"
            id="topic"
            placeholder="Example: Dr. B. R. Ambedkar"
        >

        <button
            id="compareBtn"
            onclick="comparePrompts()"
        >
            Generate Comparison
        </button>

        <p class="status" id="status"></p>

        <p class="error" id="error"></p>

    </section>


    <section class="results hidden" id="results">

        <div class="result-card">

            <h2>1. Baseline Prompt</h2>

            <h3>Prompt</h3>

            <div class="prompt" id="baselinePrompt"></div>

            <h3>AI Response</h3>

            <div class="response" id="baselineResponse"></div>

        </div>


        <div class="result-card">

            <h2>2. Enhanced Prompt</h2>

            <h3>Prompt</h3>

            <div class="prompt" id="enhancedPrompt"></div>

            <h3>AI Response</h3>

            <div class="response" id="enhancedResponse"></div>

        </div>


        <div class="result-card full-width">

            <h2>3. Evaluation and Comparison</h2>

            <div class="evaluation" id="evaluation"></div>

        </div>

    </section>

</div>


<script>

async function comparePrompts() {

    const topic =
        document.getElementById("topic").value.trim();

    const button =
        document.getElementById("compareBtn");

    const status =
        document.getElementById("status");

    const error =
        document.getElementById("error");

    const results =
        document.getElementById("results");


    error.textContent = "";

    results.classList.add("hidden");


    if (!topic) {

        error.textContent =
            "Please enter a topic first.";

        return;

    }


    button.disabled = true;

    button.textContent = "Generating...";

    status.textContent =
        "Generating baseline response...";


    try {

        const response = await fetch("/compare", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                topic: topic
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Something went wrong."
            );

        }


        // Display baseline prompt and response

        document.getElementById(
            "baselinePrompt"
        ).textContent = data.baseline_prompt;


        document.getElementById(
            "baselineResponse"
        ).textContent = data.baseline_response;


        // Display enhanced prompt and response

        document.getElementById(
            "enhancedPrompt"
        ).textContent = data.enhanced_prompt;


        document.getElementById(
            "enhancedResponse"
        ).textContent = data.enhanced_response;


        // Display evaluation

        document.getElementById(
            "evaluation"
        ).textContent = data.evaluation;


        results.classList.remove("hidden");


        status.textContent =
            "Comparison completed successfully! Time taken: " +
            data.time_taken + " seconds.";


    }

    catch (err) {

        error.textContent =
            "Error: " + err.message;

        status.textContent = "";

    }

    finally {

        button.disabled = false;

        button.textContent = "Generate Comparison";

    }

}

</script>

</body>

</html>
"""


# ------------------------------------------
# 7. RUN FLASK APPLICATION
# ------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000,
        threaded=True
    )
