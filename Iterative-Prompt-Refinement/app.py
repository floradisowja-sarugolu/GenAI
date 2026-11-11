
import os
import re
import time

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=API_KEY)

app = Flask(__name__)


# --------------------------------------------------
# 2. GEMINI RESPONSE GENERATION
# --------------------------------------------------

def generate_response(prompt, max_tokens=600):

    for attempt in range(2):

        try:
            print(f"Generating response using {MODEL_NAME}...")

            result = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "max_output_tokens": max_tokens
                }
            )

            response = (result.text or "").strip()

            if not response:
                raise Exception("Gemini returned an empty response.")

            return response

        except Exception as error:

            error_text = str(error).lower()

            # Quota errors should not be retried.
            if "429" in error_text or "resource_exhausted" in error_text:
                raise Exception(
                    "Gemini API quota exhausted. Your selected model "
                    "has no available free-tier request quota. "
                    "Check your Google AI Studio usage and select a "
                    "model with available quota, or enable billing."
                )

            # Retry temporary server/network errors.
            temporary_error = any(term in error_text for term in [
                "503",
                "unavailable",
                "500",
                "502",
                "504",
                "timeout",
                "server disconnected",
                "connection reset",
                "connection error"
            ])

            if temporary_error and attempt == 0:
                print("Temporary connection error. Retrying...")
                time.sleep(3)
                continue

            raise Exception(str(error))

    raise Exception("Unable to generate a response.")


# --------------------------------------------------
# 3. VALIDATION ENGINE
# --------------------------------------------------

REQUIRED_KEYWORDS = [
    "24-hr cold",
    "ocean plastic",
    "$29"
]

ENERGETIC_WORDS = [
    "discover", "amazing", "exciting", "powerful",
    "bold", "adventure", "refresh", "enjoy",
    "transform", "ultimate", "revolutionary",
    "energize", "ready", "freedom", "explore"
]


def count_words(text):
    return len(re.findall(r"\b[\w$-]+\b", text))


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def check_energetic_tone(text):

    lower_text = text.lower()

    return any(
        re.search(r"\b" + re.escape(word) + r"\b", lower_text)
        for word in ENERGETIC_WORDS
    )


def validate_cycle(cycle_number, text):

    word_count = count_words(text)
    sentence_count = len(split_sentences(text))

    if cycle_number == 1:

        checks = [
            {
                "name": "Response is not empty",
                "passed": bool(text.strip())
            },
            {
                "name": "Product is mentioned",
                "passed": "bottle" in text.lower()
            }
        ]

    elif cycle_number == 2:

        checks = [
            {
                "name": "Exactly 2 sentences",
                "passed": sentence_count == 2
            },
            {
                "name": "Energetic tone",
                "passed": check_energetic_tone(text)
            }
        ]

    else:

        checks = [
            {
                "name": "Exactly 2 sentences",
                "passed": sentence_count == 2
            },
            {
                "name": "Energetic tone",
                "passed": check_energetic_tone(text)
            }
        ]

        for keyword in REQUIRED_KEYWORDS:
            checks.append({
                "name": f'Contains "{keyword}"',
                "passed": keyword.lower() in text.lower()
            })

        checks.append({
            "name": "Word count between 45 and 60",
            "passed": 45 <= word_count <= 60
        })

    passed_count = sum(
        1 for check in checks if check["passed"]
    )

    total_checks = len(checks)

    score = round(
        (passed_count / total_checks) * 100
    ) if total_checks else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "checks": checks,
        "passed": passed_count,
        "total": total_checks,
        "score": score
    }


# --------------------------------------------------
# 4. THREE ITERATIVE PROMPT CYCLES
# --------------------------------------------------

def run_refinement_experiment(product):

    cycles = []

    # CYCLE 1: BASELINE

    prompt1 = f"""
Write a promotional product pitch for this product:

{product}

Make it interesting and explain why someone might want to buy it.
"""

    print("Cycle 1: Baseline generation...")

    response1 = generate_response(prompt1)

    validation1 = validate_cycle(1, response1)

    cycles.append({
        "number": 1,
        "title": "Baseline",
        "description": "General instructions without strict constraints",
        "prompt": prompt1.strip(),
        "response": response1,
        "validation": validation1
    })

    # CYCLE 2: STRUCTURAL CONSTRAINTS

    prompt2 = f"""
Improve the following product pitch.

Product:
{product}

Previous response:
{response1}

Instructions:
- Write exactly 2 sentences.
- Use an energetic and exciting tone.
- Make the pitch attractive to potential customers.
- Keep the main product idea from the previous response.
- Return only the improved pitch.
"""

    print("Cycle 2: Structural refinement...")

    response2 = generate_response(prompt2)

    validation2 = validate_cycle(2, response2)

    cycles.append({
        "number": 2,
        "title": "Structural Constraints",
        "description": "Two sentences with an energetic tone",
        "prompt": prompt2.strip(),
        "response": response2,
        "validation": validation2
    })

    # CYCLE 3: FACTUAL AND ENTITY GROUNDING

    prompt3 = f"""
Further improve the following product pitch.

Product:
{product}

Previous response:
{response2}

Follow ALL the instructions below:

1. Write exactly 2 sentences.
2. Maintain an energetic and exciting tone.
3. Include the exact keyword "24-hr cold".
4. Include the exact keyword "ocean plastic".
5. Include the exact price "$29".
6. The complete pitch must contain between 45 and 60 words.
7. Keep the product information clear and persuasive.
8. Do not remove any required keywords.
9. Return only the final product pitch.

Make sure every condition is satisfied.
"""

    print("Cycle 3: Factual refinement...")

    response3 = generate_response(prompt3)

    validation3 = validate_cycle(3, response3)

    cycles.append({
        "number": 3,
        "title": "Factual & Entity Grounding",
        "description": "Exact keywords, price and word-count requirements",
        "prompt": prompt3.strip(),
        "response": response3,
        "validation": validation3
    })

    # WORD COUNT DELTA

    for index, cycle in enumerate(cycles):

        if index == 0:
            cycle["word_delta"] = 0

        else:
            previous_count = cycles[index - 1]["validation"]["word_count"]
            current_count = cycle["validation"]["word_count"]

            cycle["word_delta"] = current_count - previous_count

    return cycles


# --------------------------------------------------
# 5. FLASK ROUTES
# --------------------------------------------------

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/api/refine", methods=["POST"])
def refine():

    try:

        data = request.get_json(silent=True) or {}

        product = data.get("product", "").strip()

        if not product:
            return jsonify({
                "error": "Please enter a product description."
            }), 400

        cycles = run_refinement_experiment(product)

        return jsonify({
            "success": True,
            "cycles": cycles
        })

    except Exception as error:

        print("Gemini API error:", str(error))

        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# 6. FRONTEND
# --------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Iterative Prompt Refinement</title>

<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    background: #f4f6fb;
    color: #202938;
    padding: 35px 22px;
}

.container {
    width: 100%;
    max-width: 1350px;
    margin: auto;
}

header {
    text-align: center;
    margin-bottom: 32px;
}

header h1 {
    font-size: 32px;
    margin-bottom: 10px;
    color: #202b43;
}

header p {
    color: #758096;
    line-height: 1.7;
    font-size: 15px;
}

.input-panel {
    background: white;
    padding: 27px;
    border: 1px solid #e3e8f1;
    border-radius: 14px;
    box-shadow: 0 4px 18px rgba(25, 40, 70, 0.04);
    margin-bottom: 30px;
}

label {
    font-weight: bold;
    display: block;
    margin-bottom: 12px;
    font-size: 16px;
}

textarea {
    width: 100%;
    min-height: 100px;
    padding: 15px;
    border: 1px solid #d5dce8;
    border-radius: 9px;
    font-size: 15px;
    resize: vertical;
    font-family: Arial, sans-serif;
    line-height: 1.6;
    outline: none;
}

textarea:focus {
    border-color: #526fe5;
    box-shadow: 0 0 0 3px rgba(82, 111, 229, 0.10);
}

button {
    background: #405bd5;
    color: white;
    border: none;
    padding: 14px 25px;
    border-radius: 9px;
    margin-top: 17px;
    cursor: pointer;
    font-size: 15px;
    font-weight: bold;
}

button:hover {
    background: #3048b8;
}

button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.status {
    color: #405bd5;
    margin-top: 15px;
    font-size: 14px;
}

.error {
    color: #c0392b;
    background: #fff3f1;
    border-radius: 8px;
    padding: 12px;
    margin-top: 14px;
    font-size: 14px;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.error:empty {
    display: none;
}

/* Progress timeline */

.timeline {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 38px 0;
    gap: 15px;
}

.step {
    text-align: center;
}

.circle {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: #dce2ed;
    color: #536176;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto auto 9px;
    font-weight: bold;
    font-size: 17px;
}

.step.completed .circle {
    background: #16864a;
    color: white;
}

.step span {
    font-size: 13px;
    color: #69758a;
}

.line {
    height: 3px;
    width: 100px;
    background: #dce2ed;
    margin-bottom: 23px;
}

/* Result cards */

.cycles {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 20px;
    align-items: stretch;
}

.cycle-card {
    background: white;
    border: 1px solid #e2e7f0;
    border-radius: 14px;
    padding: 23px;
    min-width: 0;
    box-shadow: 0 4px 18px rgba(25, 40, 70, 0.04);
}

.card-top {
    border-bottom: 1px solid #edf0f5;
    padding-bottom: 15px;
    margin-bottom: 18px;
}

.cycle-number {
    color: #526fe5;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.cycle-card h2 {
    font-size: 19px;
    margin-bottom: 8px;
}

.description {
    color: #7a8497;
    font-size: 13px;
    line-height: 1.5;
}

.section-label {
    font-size: 12px;
    color: #526fe5;
    font-weight: bold;
    margin: 17px 0 9px;
    letter-spacing: 0.5px;
}

.prompt-box {
    background: #f3f5fa;
    padding: 13px;
    border-radius: 8px;
    font-size: 12px;
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-height: 210px;
    overflow-y: auto;
}

.output-box {
    background: #f8f9fc;
    border: 1px solid #edf0f5;
    padding: 15px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.8;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    min-height: 145px;
}

.badges {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 15px 0;
}

.badge {
    font-size: 11px;
    padding: 7px 10px;
    border-radius: 20px;
    background: #edf0f7;
    color: #39465b;
}

.score {
    font-size: 24px;
    font-weight: bold;
    margin: 15px 0;
    color: #202b43;
}

.score-caption {
    color: #7a8497;
    font-size: 12px;
    font-weight: normal;
}

.check-item {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    font-size: 12px;
    line-height: 1.5;
    padding: 9px 0;
    border-bottom: 1px solid #edf0f5;
}

.pass {
    color: #16864a;
    font-weight: bold;
}

.fail {
    color: #c0392b;
    font-weight: bold;
}

/* Delta table */

.section {
    background: white;
    padding: 25px;
    border: 1px solid #e1e6ee;
    border-radius: 14px;
    margin-top: 28px;
    box-shadow: 0 4px 18px rgba(25, 40, 70, 0.04);
}

.section h2 {
    margin-bottom: 18px;
    font-size: 20px;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    text-align: left;
    border-bottom: 1px solid #e7eaf0;
    padding: 15px 12px;
    font-size: 14px;
}

th {
    background: #f3f5f9;
    color: #39465b;
}

.hidden {
    display: none;
}

@media (max-width: 1050px) {
    .cycles {
        grid-template-columns: 1fr 1fr;
    }
}

@media (max-width: 700px) {
    body {
        padding: 20px 12px;
    }

    header h1 {
        font-size: 25px;
    }

    .cycles {
        grid-template-columns: 1fr;
    }

    .timeline {
        gap: 6px;
    }

    .line {
        width: 35px;
    }

    .step span {
        font-size: 10px;
    }

    .circle {
        width: 40px;
        height: 40px;
    }

    th, td {
        padding: 9px 6px;
        font-size: 11px;
    }
}

</style>
</head>

<body>

<div class="container">

<header>
    <h1>Iterative Prompt Refinement</h1>
    <p>Observe how instructions improve an AI-generated product pitch.</p>
</header>

<section class="input-panel">

    <label for="product">Product Description</label>

    <textarea id="product">Reusable insulated stainless steel water bottle designed for everyday use.</textarea>

    <button id="runButton" onclick="runExperiment()">
        Start Refinement
    </button>

    <p class="status" id="status"></p>
    <p class="error" id="error"></p>

</section>

<div class="timeline hidden" id="timeline">

    <div class="step" id="step1">
        <div class="circle">1</div>
        <span>Baseline</span>
    </div>

    <div class="line"></div>

    <div class="step" id="step2">
        <div class="circle">2</div>
        <span>Structural</span>
    </div>

    <div class="line"></div>

    <div class="step" id="step3">
        <div class="circle">3</div>
        <span>Grounding</span>
    </div>

</div>

<section class="cycles hidden" id="cycles"></section>

<section class="section hidden" id="deltaSection">

    <h2>Delta Analysis Across Three Rounds</h2>

    <div style="overflow-x:auto;">

        <table>

            <thead>
                <tr>
                    <th>Cycle</th>
                    <th>Words</th>
                    <th>Word Delta</th>
                    <th>Sentences</th>
                    <th>Compliance</th>
                </tr>
            </thead>

            <tbody id="deltaBody"></tbody>

        </table>

    </div>

</section>

</div>

<script>

async function runExperiment() {

    const product =
        document.getElementById("product").value.trim();

    const button =
        document.getElementById("runButton");

    const status =
        document.getElementById("status");

    const error =
        document.getElementById("error");

    const cyclesContainer =
        document.getElementById("cycles");

    const timeline =
        document.getElementById("timeline");

    const deltaSection =
        document.getElementById("deltaSection");


    error.textContent = "";
    cyclesContainer.innerHTML = "";

    deltaSection.classList.add("hidden");
    timeline.classList.add("hidden");
    cyclesContainer.classList.add("hidden");

    document.querySelectorAll(".step").forEach(step => {
        step.classList.remove("completed");
    });


    if (!product) {
        error.textContent =
            "Please enter a product description.";
        return;
    }


    button.disabled = true;
    button.textContent = "Generating...";

    status.textContent =
        "Cycle 1: Generating baseline response...";


    try {

        const response = await fetch("/api/refine", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                product: product
            })

        });


        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Something went wrong."
            );
        }


        status.textContent =
            "Displaying the three refinement cycles...";

        renderCycles(data.cycles);
        renderDelta(data.cycles);

        timeline.classList.remove("hidden");
        cyclesContainer.classList.remove("hidden");
        deltaSection.classList.remove("hidden");

        status.textContent =
            "All three refinement cycles completed successfully.";

    }

    catch (err) {

        error.textContent = "Error: " + err.message;
        status.textContent = "";

    }

    finally {

        button.disabled = false;
        button.textContent = "Start Refinement";

    }

}


function renderCycles(cycles) {

    const container =
        document.getElementById("cycles");

    container.innerHTML = "";


    cycles.forEach(cycle => {

        const validation = cycle.validation;

        const card = document.createElement("div");
        card.className = "cycle-card";


        const top = document.createElement("div");
        top.className = "card-top";

        const number = document.createElement("div");
        number.className = "cycle-number";
        number.textContent = "REFINEMENT ROUND " + cycle.number;

        const title = document.createElement("h2");
        title.textContent = cycle.title;

        const description = document.createElement("p");
        description.className = "description";
        description.textContent = cycle.description;

        top.appendChild(number);
        top.appendChild(title);
        top.appendChild(description);

        card.appendChild(top);


        addSection(card, "PROMPT", cycle.prompt, "prompt-box");

        addSection(
            card,
            "GENERATED OUTPUT",
            cycle.response,
            "output-box"
        );


        const badges = document.createElement("div");
        badges.className = "badges";

        badges.appendChild(
            makeBadge(validation.word_count + " words")
        );

        badges.appendChild(
            makeBadge(validation.sentence_count + " sentences")
        );

        badges.appendChild(
            makeBadge(
                "Word delta: " +
                (cycle.word_delta > 0 ? "+" : "") +
                cycle.word_delta
            )
        );

        card.appendChild(badges);


        const score = document.createElement("div");
        score.className = "score";
        score.textContent = validation.score + "%";

        const caption = document.createElement("span");
        caption.className = "score-caption";
        caption.textContent =
            " Compliance (" +
            validation.passed + "/" +
            validation.total + " checks)";

        score.appendChild(caption);
        card.appendChild(score);


        const heading = document.createElement("div");
        heading.className = "section-label";
        heading.textContent = "VALIDATION CHECKLIST";

        card.appendChild(heading);


        validation.checks.forEach(check => {

            const item = document.createElement("div");
            item.className = "check-item";

            const icon = document.createElement("span");
            icon.className = check.passed ? "pass" : "fail";
            icon.textContent = check.passed ? "✓" : "✗";

            const label = document.createElement("span");
            label.textContent = check.name;

            item.appendChild(icon);
            item.appendChild(label);

            card.appendChild(item);

        });


        container.appendChild(card);

        document.getElementById(
            "step" + cycle.number
        ).classList.add("completed");

    });

}


function addSection(card, headingText, content, className) {

    const heading = document.createElement("div");
    heading.className = "section-label";
    heading.textContent = headingText;

    const box = document.createElement("div");
    box.className = className;
    box.textContent = content;

    card.appendChild(heading);
    card.appendChild(box);

}


function makeBadge(text) {

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = text;

    return badge;

}


function renderDelta(cycles) {

    const body =
        document.getElementById("deltaBody");

    body.innerHTML = "";


    cycles.forEach(cycle => {

        const row = document.createElement("tr");

        const values = [
            "Cycle " + cycle.number,
            cycle.validation.word_count,
            (cycle.word_delta > 0 ? "+" : "") + cycle.word_delta,
            cycle.validation.sentence_count,
            cycle.validation.score + "%"
        ];

        values.forEach(value => {

            const cell = document.createElement("td");
            cell.textContent = value;

            row.appendChild(cell);

        });

        body.appendChild(row);

    });

}

</script>

</body>
</html>
"""


# --------------------------------------------------
# 7. START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
