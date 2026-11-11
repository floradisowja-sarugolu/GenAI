
import os
import re
import json
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from google import genai



load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

app = Flask(__name__)
client = genai.Client(api_key=API_KEY) if API_KEY else None


# =========================================================
# DEFAULT FEW-SHOT EXEMPLARS
# These examples are stored in the backend.
# They are not displayed in the GUI.
# =========================================================

DEFAULT_EXAMPLES = [
    {
        "text": "These headphones are amazing! The sound quality is excellent and the battery lasts all day.",
        "sentiment": "Positive"
    },
    {
        "text": "My new phone stopped working after two days. Customer service was completely unhelpful.",
        "sentiment": "Negative"
    },
    {
        "text": "The laptop has 16GB RAM, a 512GB SSD, and a 15-inch display.",
        "sentiment": "Neutral"
    }
]


# =========================================================
# PROMPT BUILDERS
# =========================================================

def build_zero_shot_prompt(question):
    return f"""
You are an expert customer review sentiment classifier.

Classify the sentiment of the given customer review into exactly
one of these categories:

- Positive
- Negative
- Neutral

Instructions:
1. Understand the complete meaning of the review.
2. Consider mixed emotions and sarcastic statements.
3. Do not classify based on individual keywords alone.
4. Give a confidence score between 0 and 1.
5. Return only valid JSON.

Required JSON format:
{{
    "sentiment": "Positive",
    "confidence": 0.95,
    "explanation": "Brief explanation of the classification."
}}

Customer review:
{question}
"""


def build_few_shot_prompt(question, examples):
    examples_text = ""

    for index, example in enumerate(examples, start=1):
        examples_text += f"""
Example {index}:
Review: {example["text"]}
Sentiment: {example["sentiment"]}
"""

    return f"""
You are an expert customer review sentiment classifier.

Learn the classification pattern from the following examples.

{examples_text}

Now classify the new customer review.

Categories:
- Positive
- Negative
- Neutral

Instructions:
1. Follow the classification pattern demonstrated by the examples.
2. Understand the complete meaning of the review.
3. Consider mixed emotions and sarcastic statements.
4. Do not classify based on individual keywords alone.
5. Give a confidence score between 0 and 1.
6. Return only valid JSON.

Required JSON format:
{{
    "sentiment": "Positive",
    "confidence": 0.95,
    "explanation": "Brief explanation of the classification."
}}

New customer review:
{question}
"""


# =========================================================
# RESPONSE PARSING
# =========================================================

def parse_model_response(response_text):
    try:
        cleaned = response_text.strip()

        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        result = json.loads(cleaned)

        sentiment = str(result.get("sentiment", "")).capitalize()

        if sentiment not in ["Positive", "Negative", "Neutral"]:
            raise ValueError("Invalid sentiment category")

        confidence = float(result.get("confidence", 0))

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        explanation = str(
            result.get("explanation", "No explanation provided.")
        )

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "explanation": explanation,
            "format_valid": True
        }

    except Exception:
        return {
            "sentiment": "Unable to parse",
            "confidence": 0,
            "explanation": response_text[:500],
            "format_valid": False
        }


# =========================================================
# TOKEN USAGE
# =========================================================

def get_token_usage(response):
    usage = getattr(response, "usage_metadata", None)

    if not usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    return {
        "prompt_tokens": getattr(
            usage, "prompt_token_count", 0
        ) or 0,

        "completion_tokens": getattr(
            usage, "candidates_token_count", 0
        ) or 0,

        "total_tokens": getattr(
            usage, "total_token_count", 0
        ) or 0
    }


# =========================================================
# GEMINI GENERATION
# =========================================================

def generate_with_gemini(prompt):
    if not client:
        raise RuntimeError(
            "Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    response_text = response.text or ""

    result = parse_model_response(response_text)
    result["tokens"] = get_token_usage(response)

    return result


# =========================================================
# DEMO MODE
# Illustrative local output; not a real ICL experiment.
# =========================================================

def generate_demo(question, strategy):
    text = question.lower()

    positive_words = [
        "amazing", "excellent", "great", "good", "love",
        "wonderful", "perfect", "happy", "best", "fantastic"
    ]

    negative_words = [
        "bad", "terrible", "worst", "poor", "hate",
        "broken", "disappointed", "unhelpful", "waste",
        "problem", "stopped working"
    ]

    positive_score = sum(
        word in text for word in positive_words
    )

    negative_score = sum(
        word in text for word in negative_words
    )

    if positive_score > negative_score:
        sentiment = "Positive"
        confidence = 0.80
        explanation = "The review contains more positive expressions."

    elif negative_score > positive_score:
        sentiment = "Negative"
        confidence = 0.80
        explanation = "The review contains more negative expressions."

    else:
        sentiment = "Neutral"
        confidence = 0.60
        explanation = (
            "The simple demo classifier did not find a clear "
            "positive or negative signal."
        )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "explanation": explanation,
        "format_valid": True,
        "tokens": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "mode": "Demo"
    }


# =========================================================
# COMPARISON ROUTE
# =========================================================

@app.route("/api/fewshot_compare", methods=["POST"])
def fewshot_compare():

    try:
        data = request.get_json(silent=True) or {}

        question = str(data.get("text", "")).strip()
        mode = str(data.get("mode", "demo")).lower()

        if not question:
            return jsonify({
                "success": False,
                "error": "Please enter a customer review."
            }), 400

        # Use backend examples automatically.
        examples = DEFAULT_EXAMPLES

        zero_prompt = build_zero_shot_prompt(question)

        few_prompt = build_few_shot_prompt(
            question,
            examples
        )

        if mode == "live":

            zero_result = generate_with_gemini(zero_prompt)

            few_result = generate_with_gemini(few_prompt)

        else:

            zero_result = generate_demo(
                question,
                "zero_shot"
            )

            few_result = generate_demo(
                question,
                "few_shot"
            )

        zero_result["mode"] = mode.capitalize()
        few_result["mode"] = mode.capitalize()

        zero_sentiment = zero_result["sentiment"]
        few_sentiment = few_result["sentiment"]

        return jsonify({
            "success": True,

            "question": question,

            "zero_shot": zero_result,

            "few_shot": few_result,

            "match": (
                zero_sentiment == few_sentiment
                and zero_result["format_valid"]
                and few_result["format_valid"]
            ),

            "exemplar_count": len(examples),

            "exemplar_distribution": {
                "Positive": 1,
                "Negative": 1,
                "Neutral": 1
            },

            "mode": mode.capitalize()
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# =========================================================
# MAIN GUI
# =========================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Sentiment Studio | Experiment 5</title>

<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: "Segoe UI", Arial, sans-serif;
    background: #0b1020;
    color: #e5e7eb;
    min-height: 100vh;
    padding: 35px 20px;
}

.container {
    max-width: 1150px;
    margin: auto;
}

.header {
    text-align: center;
    margin-bottom: 35px;
}

.header h1 {
    font-size: 34px;
    color: #ffffff;
    margin-bottom: 10px;
}

.header p {
    color: #9ca3af;
    font-size: 15px;
}

.tag {
    display: inline-block;
    background: #25264b;
    color: #a5b4fc;
    padding: 7px 15px;
    border-radius: 20px;
    font-size: 12px;
    margin-bottom: 15px;
}

.panel {
    background: #151c30;
    border: 1px solid #29334c;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 25px;
}

.panel h2 {
    font-size: 19px;
    margin-bottom: 18px;
    color: #f3f4f6;
}

textarea {
    width: 100%;
    min-height: 130px;
    resize: vertical;
    background: #0d1425;
    border: 1px solid #35415e;
    border-radius: 10px;
    padding: 16px;
    color: white;
    font-size: 15px;
    outline: none;
    line-height: 1.6;
}

textarea:focus {
    border-color: #818cf8;
}

.samples {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 15px;
}

.sample-btn {
    background: #202b43;
    color: #cbd5e1;
    border: 1px solid #35415e;
    padding: 9px 13px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 12px;
}

.sample-btn:hover {
    background: #303d5c;
}

.controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    flex-wrap: wrap;
    margin-top: 22px;
}

.mode-control {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #9ca3af;
    font-size: 14px;
}

select {
    background: #0d1425;
    color: white;
    border: 1px solid #35415e;
    border-radius: 8px;
    padding: 10px 14px;
    outline: none;
}

.actions {
    display: flex;
    gap: 10px;
}

button {
    font-family: inherit;
}

.primary-btn {
    background: #6366f1;
    border: none;
    color: white;
    padding: 12px 22px;
    border-radius: 9px;
    cursor: pointer;
    font-weight: 600;
}

.primary-btn:hover {
    background: #4f46e5;
}

.primary-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.clear-btn {
    background: #263249;
    color: #d1d5db;
    border: 1px solid #3a4964;
    padding: 12px 20px;
    border-radius: 9px;
    cursor: pointer;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}

.result-card {
    background: #151c30;
    border: 1px solid #29334c;
    border-radius: 15px;
    padding: 23px;
    min-width: 0;
}

.result-card h2 {
    font-size: 18px;
    margin-bottom: 8px;
}

.subtitle {
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 22px;
}

.sentiment {
    font-size: 27px;
    font-weight: 700;
    margin-bottom: 15px;
}

.positive {
    color: #4ade80;
}

.negative {
    color: #fb7185;
}

.neutral {
    color: #fbbf24;
}

.unknown {
    color: #9ca3af;
}

.metric {
    margin-top: 17px;
}

.metric-label {
    display: flex;
    justify-content: space-between;
    color: #9ca3af;
    font-size: 12px;
    margin-bottom: 8px;
}

.progress {
    height: 8px;
    background: #29334c;
    border-radius: 10px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: #818cf8;
    border-radius: 10px;
    width: 0%;
    transition: width 0.4s;
}

.explanation {
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 20px;
    min-height: 45px;
}

.token-info {
    border-top: 1px solid #29334c;
    margin-top: 20px;
    padding-top: 15px;
    color: #9ca3af;
    font-size: 12px;
    line-height: 1.8;
}

.status {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 20px;
    font-size: 11px;
    margin-top: 15px;
}

.valid {
    background: #123b2c;
    color: #4ade80;
}

.invalid {
    background: #4b1d2a;
    color: #fb7185;
}

.overview {
    background: #151c30;
    border: 1px solid #29334c;
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
}

.overview h2 {
    margin-bottom: 20px;
    font-size: 19px;
}

.overview-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}

.overview-item {
    background: #0d1425;
    padding: 17px;
    border-radius: 10px;
}

.overview-item span {
    color: #9ca3af;
    font-size: 12px;
    display: block;
    margin-bottom: 8px;
}

.overview-item strong {
    font-size: 20px;
    color: #a5b4fc;
}

.message {
    display: none;
    margin-top: 18px;
    padding: 13px;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.5;
}

.error {
    display: block;
    background: #451e2a;
    color: #fda4af;
}

.loading {
    display: none;
    text-align: center;
    color: #a5b4fc;
    margin-top: 22px;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 35px;
}

@media(max-width: 750px) {

    .results-grid {
        grid-template-columns: 1fr;
    }

    .overview-grid {
        grid-template-columns: 1fr;
    }

    .header h1 {
        font-size: 27px;
    }

    .controls {
        align-items: stretch;
        flex-direction: column;
    }

    .actions {
        width: 100%;
    }

    .actions button {
        flex: 1;
    }
}

</style>
</head>


<body>

<div class="container">

    <header class="header">

        <div class="tag">EXPERIMENT 05 · PROMPT ENGINEERING</div>

        <h1>Sentiment Studio</h1>

        <p>
            Compare Zero-Shot and Few-Shot prompting
            using customer review sentiment analysis.
        </p>

    </header>


    <!-- REVIEW INPUT -->

    <section class="panel">

        <h2>Customer Review Analyzer</h2>

        <textarea
            id="question"
            placeholder="Enter a customer review to analyze its sentiment..."
        ></textarea>


        <div class="samples">

            <button class="sample-btn" onclick="loadSample(0)">
                Sample 1 · Positive
            </button>

            <button class="sample-btn" onclick="loadSample(1)">
                Sample 2 · Negative
            </button>

            <button class="sample-btn" onclick="loadSample(2)">
                Sample 3 · Neutral
            </button>

            <button class="sample-btn" onclick="loadSample(3)">
                Sample 4 · Mixed
            </button>

            <button class="sample-btn" onclick="loadSample(4)">
                Sample 5 · Sarcastic
            </button>

        </div>


        <div class="controls">

            <div class="mode-control">

                <label for="mode">Execution Mode:</label>

                <select id="mode">
                    <option value="demo">Demo Mode</option>
                    <option value="live">Live Gemini AI</option>
                </select>

            </div>


            <div class="actions">

                <button
                    class="clear-btn"
                    onclick="clearAll()"
                >
                    Clear
                </button>

                <button
                    class="primary-btn"
                    id="compareBtn"
                    onclick="compareStrategies()"
                >
                    Compare Strategies
                </button>

            </div>

        </div>


        <div class="loading" id="loading">
            Analyzing review using both prompting strategies...
        </div>

        <div class="message" id="errorMessage"></div>

    </section>


    <!-- COMPARISON RESULTS -->

    <section class="results-grid">

        <!-- ZERO SHOT -->

        <div class="result-card">

            <h2>Zero-Shot Prompting</h2>

            <p class="subtitle">
                Classification without providing labeled examples.
            </p>

            <div class="sentiment unknown" id="zeroSentiment">
                Waiting for input
            </div>

            <div class="metric">

                <div class="metric-label">
                    <span>Classification Confidence</span>
                    <span id="zeroConfidence">0%</span>
                </div>

                <div class="progress">
                    <div
                        class="progress-fill"
                        id="zeroProgress"
                    ></div>
                </div>

            </div>

            <p class="explanation" id="zeroExplanation">
                The Zero-Shot result will appear here.
            </p>

            <div class="status" id="zeroStatus">
                Awaiting result
            </div>

            <div class="token-info" id="zeroTokens">
                Token usage: Not available
            </div>

        </div>


        <!-- FEW SHOT -->

        <div class="result-card">

            <h2>Few-Shot Prompting</h2>

            <p class="subtitle">
                Classification using three balanced labeled exemplars.
            </p>

            <div class="sentiment unknown" id="fewSentiment">
                Waiting for input
            </div>

            <div class="metric">

                <div class="metric-label">
                    <span>Classification Confidence</span>
                    <span id="fewConfidence">0%</span>
                </div>

                <div class="progress">
                    <div
                        class="progress-fill"
                        id="fewProgress"
                    ></div>
                </div>

            </div>

            <p class="explanation" id="fewExplanation">
                The Few-Shot result will appear here.
            </p>

            <div class="status" id="fewStatus">
                Awaiting result
            </div>

            <div class="token-info" id="fewTokens">
                Token usage: Not available
            </div>

        </div>

    </section>


    <!-- COMPARISON OVERVIEW -->

    <section class="overview">

        <h2>Comparison Overview</h2>

        <div class="overview-grid">

            <div class="overview-item">
                <span>Strategies Agreement</span>
                <strong id="agreement">--</strong>
            </div>

            <div class="overview-item">
                <span>Few-Shot Exemplars</span>
                <strong id="exemplarCount">3</strong>
            </div>

            <div class="overview-item">
                <span>Exemplar Distribution</span>
                <strong id="distribution">1 : 1 : 1</strong>
            </div>

        </div>

        <p
            class="explanation"
            id="comparisonMessage"
        >
            Submit a customer review to see the comparison.
        </p>

    </section>


    <footer class="footer">
        Experiment 05 · In-Context Learning · Zero-Shot vs Few-Shot
    </footer>

</div>


<script>

// =========================================================
// SAMPLE REVIEWS
// =========================================================

const samples = [

    "These headphones are amazing! The sound quality is excellent and the battery lasts all day.",

    "My new phone stopped working after two days. Customer service was completely unhelpful.",

    "The laptop has 16GB RAM, a 512GB SSD, and a 15-inch display.",

    "The product looks beautiful, but the battery drains very quickly.",

    "Amazing! My phone stopped working again after the latest update. Exactly what I wanted!"
];


// =========================================================
// LOAD SAMPLE
// =========================================================

function loadSample(index) {

    document.getElementById("question").value = samples[index];

    hideError();
}


// =========================================================
// ERROR MESSAGE
// =========================================================

function showError(message) {

    const element = document.getElementById("errorMessage");

    element.textContent = message;

    element.className = "message error";
}


function hideError() {

    const element = document.getElementById("errorMessage");

    element.textContent = "";

    element.className = "message";
}


// =========================================================
// SENTIMENT COLOR
// =========================================================

function getSentimentClass(sentiment) {

    if (sentiment === "Positive") return "positive";

    if (sentiment === "Negative") return "negative";

    if (sentiment === "Neutral") return "neutral";

    return "unknown";
}


// =========================================================
// DISPLAY INDIVIDUAL RESULT
// =========================================================

function displayResult(prefix, result) {

    const sentimentElement =
        document.getElementById(prefix + "Sentiment");

    sentimentElement.textContent = result.sentiment;

    sentimentElement.className =
        "sentiment " + getSentimentClass(result.sentiment);


    const confidence =
        Math.round((result.confidence || 0) * 100);

    document.getElementById(prefix + "Confidence")
        .textContent = confidence + "%";

    document.getElementById(prefix + "Progress")
        .style.width = confidence + "%";


    document.getElementById(prefix + "Explanation")
        .textContent = result.explanation;


    const status = document.getElementById(prefix + "Status");

    if (result.format_valid) {

        status.textContent = "Valid JSON Format";

        status.className = "status valid";

    } else {

        status.textContent = "Invalid JSON Format";

        status.className = "status invalid";
    }


    const tokens = result.tokens || {};

    document.getElementById(prefix + "Tokens").innerHTML =

        "Prompt tokens: " + (tokens.prompt_tokens || 0) +

        "<br>Completion tokens: " +
        (tokens.completion_tokens || 0) +

        "<br>Total tokens: " +
        (tokens.total_tokens || 0);
}


// =========================================================
// COMPARE STRATEGIES
// =========================================================

async function compareStrategies() {

    hideError();

    const question =
        document.getElementById("question").value.trim();

    const mode =
        document.getElementById("mode").value;


    if (!question) {

        showError("Please enter a customer review first.");

        return;
    }


    const button =
        document.getElementById("compareBtn");

    const loading =
        document.getElementById("loading");


    button.disabled = true;

    button.textContent = "Analyzing...";

    loading.style.display = "block";


    try {

        const response = await fetch("/api/fewshot_compare", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                text: question,

                mode: mode

            })

        });


        const data = await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.error || "Something went wrong."
            );
        }


        // Display both results

        displayResult("zero", data.zero_shot);

        displayResult("few", data.few_shot);


        // Comparison agreement

        document.getElementById("agreement").textContent =

            data.match ? "Agree" : "Differ";


        // Backend exemplar information

        document.getElementById("exemplarCount")
            .textContent = data.exemplar_count;


        document.getElementById("distribution")
            .textContent = "1 : 1 : 1";


        // Comparison explanation

        let comparisonText = "";

        if (data.match) {

            comparisonText =
                "Both strategies produced the same sentiment classification. " +
                "Their confidence scores and token usage may still differ.";

        } else {

            comparisonText =
                "The strategies produced different classifications. " +
                "Review the explanations and confidence scores to understand the difference.";
        }


        if (mode === "demo") {

            comparisonText +=
                " Demo Mode uses a simple local keyword-based classifier. " +
                "It does not demonstrate actual Gemini In-Context Learning behavior.";

        } else {

            comparisonText +=
                " Live Gemini Mode uses the model for both prompting strategies.";
        }


        document.getElementById("comparisonMessage")
            .textContent = comparisonText;


    } catch (error) {

        showError(error.message);

    } finally {

        button.disabled = false;

        button.textContent = "Compare Strategies";

        loading.style.display = "none";
    }

}


// =========================================================
// CLEAR GUI
// =========================================================

function clearAll() {

    document.getElementById("question").value = "";

    hideError();


    const resultIds = [

        "zeroSentiment",
        "fewSentiment"

    ];

    resultIds.forEach(id => {

        const element = document.getElementById(id);

        element.textContent = "Waiting for input";

        element.className = "sentiment unknown";

    });


    document.getElementById("zeroConfidence").textContent = "0%";

    document.getElementById("fewConfidence").textContent = "0%";


    document.getElementById("zeroProgress").style.width = "0%";

    document.getElementById("fewProgress").style.width = "0%";


    document.getElementById("zeroExplanation").textContent =
        "The Zero-Shot result will appear here.";

    document.getElementById("fewExplanation").textContent =
        "The Few-Shot result will appear here.";


    document.getElementById("zeroTokens").textContent =
        "Token usage: Not available";

    document.getElementById("fewTokens").textContent =
        "Token usage: Not available";


    document.getElementById("zeroStatus").textContent =
        "Awaiting result";

    document.getElementById("fewStatus").textContent =
        "Awaiting result";


    document.getElementById("zeroStatus").className = "status";

    document.getElementById("fewStatus").className = "status";


    document.getElementById("agreement").textContent = "--";

    document.getElementById("exemplarCount").textContent = "3";

    document.getElementById("distribution").textContent = "1 : 1 : 1";


    document.getElementById("comparisonMessage").textContent =
        "Submit a customer review to see the comparison.";

}


// =========================================================
// ENTER KEY SUPPORT
// =========================================================

document.getElementById("question").addEventListener(
    "keydown",
    function(event) {

        if (event.ctrlKey && event.key === "Enter") {

            compareStrategies();

        }

    }
);

</script>

</body>
</html>
"""


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
