
import os
import re
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from google import genai

# ---------------- CONFIGURATION ----------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-3.8-flash"

app = Flask(__name__)

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ---------------- PROMPT ANALYZER ----------------

def analyze_prompt(prompt):
    """Detect common prompt problems using simple rule-based checks."""

    issues = []

    words = prompt.split()
    lower_prompt = prompt.lower()

    # 1. Ambiguity detection
    vague_terms = [
        "good", "better", "nice", "interesting",
        "appropriate", "proper", "some", "something"
    ]

    found_vague = [word for word in vague_terms if word in lower_prompt]

    if found_vague:
        issues.append({
            "type": "Ambiguity",
            "severity": "Medium",
            "description": (
                "The prompt contains vague terms: "
                + ", ".join(found_vague)
                + ". Their intended meaning is unclear."
            ),
            "suggestion": "Replace vague words with specific requirements."
        })

    # 2. Missing context
    context_indicators = [
        "for students", "for beginners", "for professionals",
        "for children", "for engineers", "for a general audience"
    ]

    if not any(term in lower_prompt for term in context_indicators):
        issues.append({
            "type": "Missing Context",
            "severity": "Medium",
            "description": "The target audience or background context is not clearly specified.",
            "suggestion": "Mention the target audience and relevant background."
        })

    # 3. Contradictory length instructions
    has_short = bool(re.search(r"\b(short|brief|concise)\b", lower_prompt))
    has_long = bool(re.search(r"\b(detailed|extensive|long)\b", lower_prompt))

    if has_short and has_long:
        issues.append({
            "type": "Contradictory Instructions",
            "severity": "High",
            "description": "The prompt requests both a short response and a detailed or long response.",
            "suggestion": "Choose one clear length requirement."
        })

    # 4. Numeric word-count conflicts
    word_limits = re.findall(r"\b(\d+)\s*words?\b", lower_prompt)

    if len(set(word_limits)) > 1:
        issues.append({
            "type": "Contradictory Instructions",
            "severity": "High",
            "description": "Multiple different word-count requirements were detected.",
            "suggestion": "Specify one final word-count range or target."
        })

    # 5. Format checking
    format_terms = [
        "bullet points", "numbered list", "table",
        "paragraph", "json", "yaml"
    ]

    if not any(term in lower_prompt for term in format_terms):
        issues.append({
            "type": "Missing Format",
            "severity": "Low",
            "description": "The prompt does not specify an output format.",
            "suggestion": "Specify whether the response should be a paragraph, list, table, or structured data."
        })

    # 6. Very short prompt
    if len(words) < 6:
        issues.append({
            "type": "Insufficient Instruction",
            "severity": "High",
            "description": "The prompt is very short and provides limited task information.",
            "suggestion": "Add the task, context, expected content, and output requirements."
        })

    return issues


# ---------------- PROMPT SCORING ----------------

def calculate_score(prompt, issues):
    """A simple heuristic score, not an AI quality measurement."""

    score = 100

    for issue in issues:
        if issue["severity"] == "High":
            score -= 25
        elif issue["severity"] == "Medium":
            score -= 15
        else:
            score -= 10

    return max(0, score)


# ---------------- GEMINI FUNCTIONS ----------------

def generate_text(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return (response.text or "").strip()


def create_refined_prompt(original_prompt, issues):
    """Ask Gemini to rewrite the prompt clearly."""

    issue_details = "\n".join(
        f"- {item['type']}: {item['description']}\n"
        f"  Suggested fix: {item['suggestion']}"
        for item in issues
    )

    refinement_request = f"""
You are an expert prompt engineer.

Improve the following user prompt.

ORIGINAL PROMPT:
{original_prompt}

DETECTED ISSUES:
{issue_details if issue_details else "No obvious issues detected."}

INSTRUCTIONS:
1. Preserve the user's original goal.
2. Remove ambiguity.
3. Add suitable context only when it can be reasonably inferred.
4. Resolve contradictory requirements by choosing a practical,
   clearly stated interpretation.
5. Specify a clear output format.
6. Do not invent facts or change the main task.
7. If important information is missing and cannot be inferred,
   insert a short placeholder such as [specify audience].
8. Return only the refined prompt, without explanations.
"""

    return generate_text(refinement_request)


# ---------------- FLASK API ----------------

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/api/diagnose", methods=["POST"])
def diagnose():

    data = request.get_json(silent=True) or {}
    original_prompt = data.get("prompt", "").strip()

    if not original_prompt:
        return jsonify({"error": "Please enter a prompt."}), 400

    if not client:
        return jsonify({
            "error": "Gemini API key is missing. Add GEMINI_API_KEY to your .env file."
        }), 500

    try:
        # STEP 1: Analyze the prompt
        issues = analyze_prompt(original_prompt)

        # STEP 2: Calculate original score
        original_score = calculate_score(original_prompt, issues)

        # STEP 3: Generate baseline response
        baseline_response = generate_text(original_prompt)

        # STEP 4: Create refined prompt
        refined_prompt = create_refined_prompt(
            original_prompt,
            issues
        )

        if not refined_prompt:
            raise ValueError("The model returned an empty refined prompt.")

        # STEP 5: Analyze the refined prompt
        refined_issues = analyze_prompt(refined_prompt)
        refined_score = calculate_score(refined_prompt, refined_issues)

        # STEP 6: Generate improved response
        improved_response = generate_text(refined_prompt)

        return jsonify({
            "original_prompt": original_prompt,
            "issues": issues,
            "original_score": original_score,
            "baseline_response": baseline_response,
            "refined_prompt": refined_prompt,
            "refined_issues": refined_issues,
            "refined_score": refined_score,
            "improved_response": improved_response,
            "score_difference": refined_score - original_score
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500


# ---------------- FRONTEND ----------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Prompt Debugger</title>

<style>
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    background: #f3f5fa;
    color: #20283b;
    min-height: 100vh;
}

header {
    background: #18243d;
    color: white;
    padding: 25px 7%;
}

header h1 {
    font-size: 25px;
    margin-bottom: 7px;
}

header p {
    color: #cbd5e1;
    font-size: 14px;
}

.container {
    max-width: 1150px;
    margin: 30px auto;
    padding: 0 20px;
}

.panel {
    background: white;
    border: 1px solid #e1e6ef;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 22px;
    box-shadow: 0 3px 12px rgba(20, 35, 60, 0.04);
}

h2 {
    font-size: 19px;
    margin-bottom: 15px;
}

.subtitle {
    color: #68748a;
    font-size: 13px;
    margin-bottom: 15px;
}

textarea {
    width: 100%;
    min-height: 125px;
    border: 1px solid #cbd3e1;
    border-radius: 8px;
    padding: 14px;
    resize: vertical;
    font-family: inherit;
    font-size: 14px;
    line-height: 1.6;
    outline: none;
}

textarea:focus {
    border-color: #4268d5;
}

.actions {
    display: flex;
    gap: 10px;
    margin-top: 15px;
    flex-wrap: wrap;
}

button {
    background: #3157c8;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 7px;
    cursor: pointer;
    font-size: 14px;
    font-weight: bold;
}

button:hover {
    background: #2445a8;
}

button.secondary {
    background: #e9edf5;
    color: #26334b;
}

button:disabled {
    opacity: 0.6;
    cursor: wait;
}

.status {
    margin-top: 15px;
    color: #3157c8;
    font-size: 14px;
}

.hidden {
    display: none;
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}

.result-card {
    border: 1px solid #e2e7ef;
    border-radius: 10px;
    padding: 18px;
    min-width: 0;
}

.result-card h3 {
    font-size: 16px;
    margin-bottom: 12px;
}

.result-text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.65;
    font-size: 14px;
    color: #47536a;
}

.score {
    display: inline-block;
    margin: 5px 0 14px;
    padding: 6px 10px;
    border-radius: 20px;
    background: #e9efff;
    color: #294cae;
    font-size: 12px;
    font-weight: bold;
}

.issue {
    padding: 13px;
    border-radius: 8px;
    background: #fff7e8;
    border: 1px solid #f3dfb6;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.5;
}

.issue.high {
    background: #fff0f0;
    border-color: #f0cccc;
}

.issue h4 {
    margin-bottom: 5px;
    color: #8b4b12;
}

.issue.high h4 {
    color: #a22d2d;
}

.success {
    color: #167447;
    background: #e9f8ef;
    border-radius: 8px;
    padding: 13px;
    font-size: 14px;
}

.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 12px;
}

.metric {
    flex: 1;
    min-width: 130px;
    padding: 15px;
    border-radius: 8px;
    background: #f3f6fc;
}

.metric strong {
    display: block;
    font-size: 23px;
    margin-bottom: 5px;
}

.metric span {
    color: #68748a;
    font-size: 12px;
}

.steps {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.step {
    padding: 9px 13px;
    background: #e9edf5;
    color: #46536b;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}

.arrow {
    color: #8290a8;
}

footer {
    text-align: center;
    color: #7c8799;
    padding: 25px;
    font-size: 12px;
}

@media(max-width: 750px) {
    .grid {
        grid-template-columns: 1fr;
    }

    header {
        padding: 22px 20px;
    }
}
</style>
</head>

<body>

<header>
    <h1>Prompt Debugger</h1>
    <p>Diagnosing Prompt Failures & Edge Cases</p>
</header>

<main class="container">

    <section class="panel">
        <h2>Enter Your Original Prompt</h2>

        <p class="subtitle">
            Enter a vague, incomplete, or contradictory prompt
            and observe how prompt debugging improves it.
        </p>

        <textarea id="promptInput"
        placeholder="Example: Explain photosynthesis in a detailed and very short way in 500 words."></textarea>

        <div class="actions">
            <button id="analyzeBtn" onclick="runExperiment()">
                Diagnose & Refine
            </button>

            <button class="secondary" onclick="loadExample()">
                Load Example
            </button>

            <button class="secondary" onclick="clearAll()">
                Clear
            </button>
        </div>

        <div id="status" class="status"></div>
    </section>

    <section id="results" class="hidden">

        <section class="panel">
            <h2>Prompt Debugging Pipeline</h2>

            <div class="steps">
                <span class="step">01. Original Prompt</span>
                <span class="arrow">→</span>
                <span class="step">02. Diagnose</span>
                <span class="arrow">→</span>
                <span class="step">03. Refine</span>
                <span class="arrow">→</span>
                <span class="step">04. Compare</span>
            </div>

            <div class="metric-row">
                <div class="metric">
                    <strong id="originalScore">0%</strong>
                    <span>Original prompt score</span>
                </div>

                <div class="metric">
                    <strong id="refinedScore">0%</strong>
                    <span>Refined prompt score</span>
                </div>

                <div class="metric">
                    <strong id="scoreDifference">0%</strong>
                    <span>Score improvement</span>
                </div>
            </div>

            <p class="subtitle" style="margin-top:12px">
                Scores are rule-based indicators of detected prompt issues,
                not measurements of factual accuracy.
            </p>
        </section>

        <section class="panel">
            <h2>1. Failure Analysis</h2>
            <p class="subtitle">
                The system checks for ambiguity, missing context,
                contradictions, and missing format instructions.
            </p>

            <div id="issues"></div>
        </section>

        <section class="panel">
            <h2>2. Original vs Refined Prompt</h2>

            <div class="grid">
                <div class="result-card">
                    <h3>Original Prompt</h3>
                    <div class="result-text" id="originalPrompt"></div>
                </div>

                <div class="result-card">
                    <h3>Refined Prompt</h3>
                    <div class="result-text" id="refinedPrompt"></div>
                </div>
            </div>
        </section>

        <section class="panel">
            <h2>3. Response Comparison</h2>

            <div class="grid">
                <div class="result-card">
                    <h3>Baseline Response</h3>
                    <div class="result-text" id="baselineResponse"></div>
                </div>

                <div class="result-card">
                    <h3>Improved Response</h3>
                    <div class="result-text" id="improvedResponse"></div>
                </div>
            </div>
        </section>

        <section class="panel">
            <h2>4. Refined Prompt Validation</h2>
            <div id="refinedIssues"></div>
        </section>

        <section class="panel">
            <h2>5. Experiment Result</h2>
            <div id="resultSummary" class="success"></div>
        </section>

    </section>

</main>

<footer>
    Generative AI Lab | Experiment 04 | Prompt Engineering
</footer>

<script>
function loadExample() {
    document.getElementById("promptInput").value =
        "Explain photosynthesis in a detailed and very short way in 500 words. Make it simple but use advanced scientific language.";
}

function clearAll() {
    document.getElementById("promptInput").value = "";
    document.getElementById("results").classList.add("hidden");
    document.getElementById("status").textContent = "";
}

function addIssueCards(containerId, issues) {
    const container = document.getElementById(containerId);
    container.replaceChildren();

    if (!issues || issues.length === 0) {
        const message = document.createElement("div");
        message.className = "success";
        message.textContent = "No issues were detected by the basic rule-based analyzer.";
        container.appendChild(message);
        return;
    }

    issues.forEach(issue => {
        const card = document.createElement("div");
        card.className = "issue" +
            (issue.severity === "High" ? " high" : "");

        const heading = document.createElement("h4");
        heading.textContent =
            issue.type + " | " + issue.severity + " severity";

        const description = document.createElement("p");
        description.textContent = issue.description;

        const suggestion = document.createElement("p");
        suggestion.textContent = "Suggested fix: " + issue.suggestion;

        card.append(heading, description, suggestion);
        container.appendChild(card);
    });
}

function setText(id, value) {
    document.getElementById(id).textContent = value || "";
}

async function runExperiment() {
    const prompt = document.getElementById("promptInput").value.trim();
    const button = document.getElementById("analyzeBtn");
    const status = document.getElementById("status");

    if (!prompt) {
        status.textContent = "Please enter a prompt first.";
        return;
    }

    button.disabled = true;
    status.textContent =
        "Analyzing prompt and generating responses. Please wait...";

    document.getElementById("results").classList.add("hidden");

    try {
        const response = await fetch("/api/diagnose", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ prompt: prompt })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Something went wrong.");
        }

        setText("originalPrompt", data.original_prompt);
        setText("refinedPrompt", data.refined_prompt);
        setText("baselineResponse", data.baseline_response);
        setText("improvedResponse", data.improved_response);

        setText("originalScore", data.original_score + "%");
        setText("refinedScore", data.refined_score + "%");

        const difference = data.score_difference;
        setText(
            "scoreDifference",
            (difference > 0 ? "+" : "") + difference + "%"
        );

        addIssueCards("issues", data.issues);
        addIssueCards("refinedIssues", data.refined_issues);

        setText(
            "resultSummary",
            "Prompt diagnosis completed. The system identified " +
            data.issues.length +
            " possible issue(s) in the original prompt. " +
            "The refined prompt score changed by " +
            (difference > 0 ? "+" : "") + difference +
            " percentage points. Compare both responses to observe " +
            "how clarification and additional instructions affect output."
        );

        document.getElementById("results").classList.remove("hidden");

        status.textContent = "Experiment completed successfully.";

    } catch (error) {
        status.textContent = "Error: " + error.message;
    } finally {
        button.disabled = false;
    }
}
</script>

</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)
