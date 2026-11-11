
import os
from pydoc import html
import re
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from google import genai

# ---------------- CONFIGURATION ----------------

load_dotenv()

app = Flask(__name__)

MODEL_NAME = "gemini-3.8-flash"
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

client = genai.Client(api_key=API_KEY) if API_KEY else None

REQUIRED_TERMS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Applications",
    "Challenges"
]


# ---------------- PROMPT GENERATION ----------------

def build_prompts(article):
    return [
        {
            "title": "Basic Prompt",
            "description": "Initial summary without explicit constraints.",
            "prompt": f"""Summarize the following article.

Article:
{article}"""
        },
        {
            "title": "Refinement Cycle 1",
            "description": "Adds word-count and language constraints.",
            "prompt": f"""Summarize the following article in 80-100 words.
Use simple English.

Article:
{article}"""
        },
        {
            "title": "Refinement Cycle 2",
            "description": "Adds word-count, bullet-format and content constraints.",
            "prompt": f"""Summarize the following article with all these constraints:

- Exactly 5 bullet points.
- The complete summary must contain 80-100 words.
- Mention all these terms: Artificial Intelligence, Machine Learning,
  Deep Learning, Applications, Challenges.
- Use simple English.
- Do not add an introduction or conclusion outside the bullet points.

Article:
{article}"""
        }
    ]


# ---------------- OUTPUT ANALYSIS ----------------

def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def analyze_output(text, stage):
    word_count = count_words(text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    bullet_lines = [
        line for line in lines
        if re.match(r"^[-*•]\s+", line)
    ]

    required_results = {
        term: bool(
            re.search(re.escape(term), text, re.IGNORECASE)
        )
        for term in REQUIRED_TERMS
    }

    if stage == 0:
        checks = {
            "Response generated": bool(text.strip())
        }

    elif stage == 1:
        checks = {
            "Word count (80-100)": 80 <= word_count <= 100,
            "Simple English instruction": True
        }

    else:
        checks = {
            "Word count (80-100)": 80 <= word_count <= 100,
            "Exactly 5 bullet points": len(bullet_lines) == 5,
            "All required terms included": all(
                required_results.values()
            )
        }

    passed = sum(checks.values())
    score = round((passed / len(checks)) * 100)

    return {
        "word_count": word_count,
        "bullet_count": len(bullet_lines),
        "required_terms": required_results,
        "checks": checks,
        "score": score
    }


# ---------------- DEMO RESPONSES ----------------

def generate_demo(article, stage):
    """
    Illustrative responses.
    Demo Mode does not call Gemini.
    """

    if stage == 0:
        return (
            "The article explains how artificial intelligence is changing "
            "modern technology. It discusses how computers can learn from "
            "data, identify patterns, and perform tasks that normally require "
            "human intelligence. It also describes the role of machine learning "
            "and deep learning in areas such as healthcare, education, business, "
            "and transportation. The article highlights useful applications "
            "while also mentioning challenges such as privacy, bias, accuracy, "
            "and responsible use."
        )

    if stage == 1:
        return (
            "Artificial intelligence helps computers perform tasks that "
            "normally require human thinking. Machine learning allows systems "
            "to learn patterns from data, while deep learning uses layered "
            "networks to solve complex problems. These technologies support "
            "applications in healthcare, education, transport, and business. "
            "They can improve speed, support decisions, and automate repeated "
            "work. However, organizations must consider data privacy, bias, "
            "security, and accuracy. Careful development and human supervision "
            "help make these systems more useful and responsible."
        )

    return """- Artificial Intelligence enables computers to perform tasks that usually require human intelligence.
- Machine Learning helps systems learn patterns from data and improve their predictions over time.
- Deep Learning uses multiple layers of neural networks to process complex information and recognize patterns.
- Applications include healthcare, education, transportation, business, and many other important fields.
- Challenges include privacy, bias, security, and accuracy, so responsible development and human supervision remain essential."""


# ---------------- FLASK ROUTES ----------------

@app.route("/")
def home():
    return render_template_string(html)


@app.route("/api/refine", methods=["POST"])
def refine():
    data = request.get_json(silent=True) or {}

    article = data.get("article", "").strip()
    mode = data.get("mode", "demo").lower()

    if not article:
        return jsonify({
            "error": "Please enter an article or technical text."
        }), 400

    if len(article) < 40:
        return jsonify({
            "error": "Please enter a longer article (at least 40 characters)."
        }), 400

    if mode not in ["demo", "live"]:
        return jsonify({
            "error": "Invalid mode selected."
        }), 400

    if mode == "live" and client is None:
        return jsonify({
            "error": "GEMINI_API_KEY is missing. Add it to your .env file."
        }), 400

    prompts = build_prompts(article)
    results = []

    try:
        for index, item in enumerate(prompts):

            if mode == "demo":
                output = generate_demo(article, index)

            else:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=item["prompt"]
                )

                output = response.text or ""

                if not output.strip():
                    raise ValueError(
                        f"Gemini returned an empty response in {item['title']}."
                    )

            audit = analyze_output(output, index)

            results.append({
                "title": item["title"],
                "description": item["description"],
                "prompt": item["prompt"],
                "output": output,
                "audit": audit
            })

        return jsonify({
            "results": results,
            "mode": mode,
            "message": "All three refinement stages completed."
        })

    except Exception as error:
        app.logger.exception("Experiment generation failed")

        error_message = str(error)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            return jsonify({
                "error": (
                    "Gemini API daily quota exceeded. "
                    "Please switch to Demo Mode to view the experiment outputs, "
                    "or retry when your quota resets."
                ),
                "quota_exceeded": True
            }), 429

        return jsonify({
            "error": error_message
        }), 500

# ---------------- FRONTEND ----------------

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Constraint Refinement Studio</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, Helvetica, sans-serif;
    background: #f3f6fb;
    color: #1e293b;
    line-height: 1.6;
}

/* HEADER */

header {
    background: #172554;
    color: white;
    padding: 24px 7%;
}

header h1 {
    font-size: 25px;
    font-weight: 700;
}

/* MAIN */

main {
    width: 90%;
    max-width: 1100px;
    margin: 30px auto;
}

/* INPUT PANEL */

.panel {
    background: white;
    border: 1px solid #e0e6ef;
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

.panel h2 {
    font-size: 21px;
    margin-bottom: 8px;
    color: #172554;
}

.subtext {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 18px;
}

textarea {
    width: 100%;
    min-height: 180px;
    resize: vertical;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    padding: 16px;
    font-family: inherit;
    font-size: 14px;
    line-height: 1.7;
    outline: none;
}

textarea:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10);
}

/* CONTROLS */

.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 18px;
}

select,
button {
    padding: 11px 18px;
    border-radius: 8px;
    font-size: 14px;
}

select {
    border: 1px solid #cbd5e1;
    background: white;
    min-width: 160px;
}

button {
    border: none;
    cursor: pointer;
    font-weight: 600;
}

button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.primary {
    background: #2563eb;
    color: white;
}

.primary:hover {
    background: #1d4ed8;
}

.secondary {
    background: #e2e8f0;
    color: #334155;
}

.secondary:hover {
    background: #cbd5e1;
}

/* STATUS */

.status {
    margin-top: 16px;
    color: #475569;
    font-size: 13px;
}

.error {
    display: none;
    color: #b91c1c;
    background: #fef2f2;
    border: 1px solid #fecaca;
    padding: 12px;
    border-radius: 8px;
    margin-top: 14px;
    font-size: 14px;
}

/* RESULTS CONTAINER */

.results {
    display: none;
    width: 100%;
}

/* VERTICAL CYCLE SECTIONS */

.stage {
    display: block;
    width: 100%;
    background: white;
    border: 1px solid #dce3ed;
    border-radius: 12px;
    margin-bottom: 28px;
    overflow: hidden;
    page-break-inside: avoid;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

.stage-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    padding: 20px 24px;
    background: #eef3ff;
    border-bottom: 1px solid #dce3ed;
}

.stage-header h3 {
    font-size: 19px;
    color: #172554;
    margin-bottom: 4px;
}

.stage-header p {
    color: #64748b;
    font-size: 13px;
}

.score {
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 13px;
    font-weight: 700;
    padding: 7px 12px;
    border-radius: 20px;
    white-space: nowrap;
}

.stage-body {
    display: block;
    width: 100%;
    padding: 24px;
}

.block-label {
    font-size: 14px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 9px;
}

.prompt-box {
    display: block;
    width: 100%;
    padding: 16px;
    background: #f7f8fb;
    border: 1px solid #e1e6ef;
    border-radius: 8px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 22px;
    color: #475569;
}

.output {
    display: block;
    width: 100%;
    padding: 20px;
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.9;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    margin-bottom: 22px;
    color: #1e293b;
}

/* METRICS */

.metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    width: 100%;
    margin-bottom: 18px;
}

.metric {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 13px;
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.metric span {
    color: #64748b;
    font-size: 12px;
}

.metric strong {
    color: #172554;
    font-size: 19px;
}

/* CONSTRAINT CHECKS */

.checks {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
}

.check {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}

.pass {
    background: #dcfce7;
    color: #166534;
}

.fail {
    background: #fee2e2;
    color: #991b1b;
}

/* REFINEMENT OVERVIEW */

.summary {
    margin-top: 30px;
    padding-top: 22px;
    border-top: 1px solid #e2e8f0;
}

.summary h3 {
    color: #172554;
    font-size: 18px;
    margin-bottom: 15px;
}

.summary-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
}

.summary-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 13px 16px;
    font-size: 14px;
}

.summary-item strong {
    color: #2563eb;
    font-size: 16px;
}

/* RESPONSIVE */

@media (max-width: 650px) {
    main {
        width: 94%;
        margin-top: 18px;
    }

    .panel {
        padding: 18px;
    }

    .stage-body {
        padding: 17px;
    }

    .metrics {
        grid-template-columns: 1fr;
    }

    .controls {
        flex-direction: column;
    }

    .controls select,
    .controls button {
        width: 100%;
    }
}

/* PRINT */

@media print {
    body {
        background: white;
    }

    header,
    .input-panel,
    .summary {
        display: none !important;
    }

    main {
        width: 100%;
        max-width: 100%;
        margin: 0;
    }

    .results {
        display: block !important;
    }

    .stage {
        display: block !important;
        width: 100% !important;
        margin-bottom: 20px;
        box-shadow: none;
        break-inside: avoid;
    }

    .stage-header {
        background: #eef3ff !important;
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }

    .pass {
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }

    .fail {
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }
}

</style>
</head>

<body>

<header>
    <h1>Constraint Refinement Studio</h1>
</header>

<main>

    <!-- ARTICLE INPUT -->

    <section class="panel input-panel">

        <h2>Article Input</h2>

        <p class="subtext">
            Enter a technical article or passage to compare three
            successive prompt refinement stages.
        </p>

        <textarea
            id="article"
            placeholder="Paste your technical article here..."
        ></textarea>

        <div class="controls">

            <select id="mode">
                <option value="demo">Demo Mode</option>
                <option value="live">Live Gemini AI</option>
            </select>

            <button
                class="primary"
                id="generateBtn"
                onclick="runExperiment()"
            >
                Run Experiment
            </button>

            <button
                class="secondary"
                onclick="clearAll()"
            >
                Clear
            </button>

        </div>

        <div class="status" id="status">
            Ready. Select a mode and enter your article.
        </div>

        <div class="error" id="error"></div>

    </section>


    <!-- RESULTS -->

    <section class="results" id="results">

        <div class="panel">

            <h2>Experiment Output</h2>

            <p class="subtext">
                Each refinement stage is displayed separately below.
            </p>

            <!-- All three cycles are stacked vertically -->

            <div id="stageResults"></div>

            <div class="summary">

                <h3>Refinement Overview</h3>

                <div
                    class="summary-grid"
                    id="summaryGrid"
                ></div>

            </div>

        </div>

    </section>

</main>


<script>

/* ---------------- RUN EXPERIMENT ---------------- */

async function runExperiment() {

    const article = document.getElementById("article").value.trim();
    const mode = document.getElementById("mode").value;

    const button = document.getElementById("generateBtn");
    const status = document.getElementById("status");
    const error = document.getElementById("error");
    const results = document.getElementById("results");

    error.style.display = "none";
    error.textContent = "";

    results.style.display = "none";

    if (!article) {
        showError("Please enter an article before running the experiment.");
        return;
    }

    if (article.length < 40) {
        showError("Please enter a longer article (at least 40 characters).");
        return;
    }

    button.disabled = true;
    button.textContent = "Generating...";

    status.textContent = mode === "demo"
        ? "Running demonstration outputs..."
        : "Generating three responses using Gemini AI...";

    try {

        const response = await fetch("/api/refine", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                article: article,
                mode: mode
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Something went wrong."
            );
        }

        displayResults(data.results);

        results.style.display = "block";

        status.textContent =
            data.message + " Mode: " + mode.toUpperCase();

    } catch (err) {

        showError(err.message);

        status.textContent =
            "Experiment could not be completed.";

    } finally {

        button.disabled = false;
        button.textContent = "Run Experiment";

    }
}


/* ---------------- DISPLAY RESULTS ---------------- */

function displayResults(items) {

    const container = document.getElementById("stageResults");
    const summary = document.getElementById("summaryGrid");

    container.replaceChildren();
    summary.replaceChildren();

    items.forEach((item) => {

        const stage = document.createElement("section");
        stage.className = "stage";

        /* STAGE HEADER */

        const header = document.createElement("div");
        header.className = "stage-header";

        const heading = document.createElement("div");

        const title = document.createElement("h3");
        title.textContent = item.title;

        const description = document.createElement("p");
        description.textContent = item.description;

        heading.append(title, description);

        const score = document.createElement("span");
        score.className = "score";
        score.textContent =
            "Audit Score: " + item.audit.score + "%";

        header.append(heading, score);


        /* STAGE BODY */

        const body = document.createElement("div");
        body.className = "stage-body";


        /* PROMPT */

        const promptLabel = document.createElement("div");
        promptLabel.className = "block-label";
        promptLabel.textContent = "Prompt Used";

        const prompt = document.createElement("div");
        prompt.className = "prompt-box";
        prompt.textContent = item.prompt;


        /* GENERATED OUTPUT */

        const outputLabel = document.createElement("div");
        outputLabel.className = "block-label";
        outputLabel.textContent = "Generated Response";

        const output = document.createElement("div");
        output.className = "output";
        output.textContent = item.output;


        /* METRICS */

        const metrics = document.createElement("div");
        metrics.className = "metrics";

        addMetric(
            metrics,
            "Word Count",
            item.audit.word_count
        );

        addMetric(
            metrics,
            "Bullet Points",
            item.audit.bullet_count
        );

        addMetric(
            metrics,
            "Constraint Score",
            item.audit.score + "%"
        );


        /* CONSTRAINT CHECKS */

        const checks = document.createElement("div");
        checks.className = "checks";

        Object.entries(item.audit.checks).forEach(
            ([name, passed]) => {

                const chip = document.createElement("span");

                chip.className =
                    "check " + (passed ? "pass" : "fail");

                chip.textContent =
                    (passed ? "✓ " : "✕ ") + name;

                checks.appendChild(chip);

            }
        );


        /* REQUIRED TERM CHECKS */

        if (item.audit.required_terms) {

            Object.entries(item.audit.required_terms).forEach(
                ([term, found]) => {

                    const chip = document.createElement("span");

                    chip.className =
                        "check " + (found ? "pass" : "fail");

                    chip.textContent =
                        (found ? "✓ " : "✕ ") + term;

                    checks.appendChild(chip);

                }
            );

        }


        /* ADD CONTENT */

        body.append(
            promptLabel,
            prompt,
            outputLabel,
            output,
            metrics,
            checks
        );

        stage.append(header, body);

        // Append each complete cycle as a separate vertical section.
        container.appendChild(stage);


        /* OVERVIEW */

        const summaryItem = document.createElement("div");
        summaryItem.className = "summary-item";

        const label = document.createElement("span");
        label.textContent = item.title;

        const value = document.createElement("strong");
        value.textContent = item.audit.score + "%";

        summaryItem.append(label, value);

        summary.appendChild(summaryItem);

    });

}


/* ---------------- METRIC CARD ---------------- */

function addMetric(container, label, value) {

    const box = document.createElement("div");
    box.className = "metric";

    const name = document.createElement("span");
    name.textContent = label;

    const number = document.createElement("strong");
    number.textContent = value;

    box.append(name, number);

    container.appendChild(box);

}


/* ---------------- ERROR MESSAGE ---------------- */

function showError(message) {

    const error = document.getElementById("error");

    error.textContent = message;
    error.style.display = "block";

}


/* ---------------- CLEAR ---------------- */

function clearAll() {

    document.getElementById("article").value = "";

    document.getElementById("results").style.display = "none";

    document.getElementById("stageResults").replaceChildren();

    document.getElementById("summaryGrid").replaceChildren();

    document.getElementById("error").style.display = "none";

    document.getElementById("error").textContent = "";

    document.getElementById("status").textContent =
        "Ready. Select a mode and enter your article.";

}

</script>

</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
