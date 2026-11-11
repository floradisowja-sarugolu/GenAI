
import os
import logging
import random
import time

from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
from google import genai

# ---------------- CONFIGURATION ----------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ---------------- GEMINI ----------------


def generate_with_gemini(prompt, max_retries=4):

    if client is None:
        raise ValueError(
            "Gemini API key is missing. Check your .env file."
        )

    for attempt in range(max_retries):

        try:
            logger.info(
                "Gemini request attempt %s/%s",
                attempt + 1,
                max_retries
            )

            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )

            answer = interaction.output_text

            if not answer or not answer.strip():
                raise ValueError("Gemini returned an empty response.")

            logger.info("Gemini response received successfully.")

            return answer.strip()

        except Exception as error:

            error_message = str(error).upper()

            temporary_error = any(
                code in error_message
                for code in [
                    "503",
                    "UNAVAILABLE",
                    "SERVICE_UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "502",
                    "504"
                ]
            )

            if not temporary_error:
                raise

            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Gemini service is temporarily unavailable "
                    f"after {max_retries} attempts. "
                    f"Please try again in a minute. "
                    f"Original error: {error}"
                ) from error

            delay = (2 ** attempt) + random.uniform(0, 1)

            logger.warning(
                "Temporary Gemini error: %s. Retrying in %.1f seconds.",
                error,
                delay
            )

            time.sleep(delay)

    raise RuntimeError("Gemini request failed.")



def build_role_prompt(role, expertise, personality, tone, question):
    return f"""
You are an AI assistant acting as the following persona.

ROLE:
{role}

EXPERTISE:
{expertise or "Use relevant knowledge for this role."}

PERSONALITY:
{personality or "Helpful, clear and professional."}

COMMUNICATION TONE:
{tone}

INSTRUCTIONS:
1. Answer from the perspective of the assigned role.
2. Adapt your knowledge and vocabulary to the role.
3. Maintain the assigned personality and tone.
4. Answer the user's actual question.
5. Do not invent facts or qualifications.
6. Make the answer useful and understandable.

USER QUESTION:
{question}

Provide your response now.
"""


def build_combined_prompt(role_prompt, negative):
    return f"""
{role_prompt}

ADDITIONAL NEGATIVE PROMPTING INSTRUCTIONS:

Avoid the following content or behaviour:

{negative}

STRICT REQUIREMENTS:
1. Follow the original assigned persona.
2. Follow all applicable negative instructions.
3. Do not include restricted content.
4. If a restriction prevents part of the answer,
   explain briefly and provide a suitable alternative.

Answer the same question while following BOTH
the role and negative prompting instructions.
"""


# ---------------- API ROUTES ----------------

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "running",
        "api_key_configured": bool(API_KEY),
        "model": MODEL_NAME
    })


@app.route("/api/generate", methods=["POST"])
def generate():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request data."}), 400

        role = str(data.get("role", "")).strip()
        expertise = str(data.get("expertise", "")).strip()
        personality = str(data.get("personality", "")).strip()
        tone = str(data.get("tone", "Professional")).strip()
        negative = str(data.get("negative", "")).strip()
        question = str(data.get("question", "")).strip()

        if not role:
            return jsonify({"error": "Please enter a persona role."}), 400

        if not negative:
            return jsonify({
                "error": "Please enter negative instructions."
            }), 400

        if not question:
            return jsonify({"error": "Please enter your question."}), 400

        role_prompt = build_role_prompt(
            role, expertise, personality, tone, question
        )

        combined_prompt = build_combined_prompt(
            role_prompt, negative
        )

        # Generate responses sequentially for easier debugging.
        logger.info("Generating role-based response.")

        role_response = generate_with_gemini(role_prompt)

        logger.info("Generating role + negative response.")

        combined_response = generate_with_gemini(combined_prompt)

        logger.info("Both responses generated successfully.")

        return jsonify({
            "success": True,
            "role_response": role_response,
            "combined_response": combined_response,
            "role_prompt": role_prompt,
            "combined_prompt": combined_prompt,
            "role": role,
            "model": MODEL_NAME
        })

    except Exception as error:
        logger.exception("Generation failed")
        return jsonify({
            "error": str(error)
        }), 500


# ---------------- FRONTEND ----------------

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Persona Lab | Experiment 06</title>

<script src="https://cdn.tailwindcss.com"></script>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #faf8f4;
    color: #29251f;
    font-family: Inter, Arial, sans-serif;
}

.sidebar {
    background: #211e1a;
    color: white;
}

.card {
    background: white;
    border: 1px solid #eee8df;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(40,30,20,.035);
}

.input-field {
    width: 100%;
    padding: 12px 13px;
    background: #fcfaf7;
    border: 1px solid #e9e2d8;
    border-radius: 9px;
    outline: none;
    font-size: 13px;
}

.input-field:focus {
    border-color: #c58b35;
    box-shadow: 0 0 0 3px rgba(197,139,53,.10);
}

textarea { resize: vertical; }

.primary-btn {
    background: #b77924;
    color: white;
    border-radius: 10px;
    padding: 11px 18px;
    font-size: 13px;
    font-weight: 700;
}

.primary-btn:hover { background: #996019; }

.primary-btn:disabled {
    opacity: .55;
    cursor: not-allowed;
}

.secondary-btn {
    border: 1px solid #e8dfd2;
    background: white;
    color: #554b3f;
    border-radius: 9px;
    padding: 9px 13px;
    font-size: 12px;
    font-weight: 600;
}

.step-number {
    width: 27px;
    height: 27px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #f5ead8;
    color: #98621d;
    font-size: 12px;
    font-weight: 800;
    flex-shrink: 0;
}

.hidden-section { display: none; }

.output-box {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.75;
    font-size: 13px;
    color: #403b35;
    min-height: 170px;
    max-height: 520px;
    overflow-y: auto;
}

.prompt-preview {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.6;
    font-size: 11px;
    color: #65594b;
    max-height: 230px;
    overflow-y: auto;
}

.role-top { border-top: 4px solid #d49a42; }
.combined-top { border-top: 4px solid #8872bb; }

.history-item {
    display: block;
    width: 100%;
    text-align: left;
    padding: 10px;
    border-radius: 9px;
    color: #d6d0c8;
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.history-item:hover { background: #34302a; }

.spinner {
    width: 17px;
    height: 17px;
    border: 2px solid #e8d7b9;
    border-top-color: #a36a20;
    border-radius: 50%;
    animation: spin .8s linear infinite;
    display: inline-block;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

@media(max-width:850px) {
    .sidebar { display: none; }
}
</style>
</head>

<body class="min-h-screen flex">

<!-- SIDEBAR -->
<aside class="sidebar w-64 min-h-screen flex flex-col p-5 flex-shrink-0">

    <div class="mb-8">
        <div class="flex items-center gap-2">
            <div class="w-9 h-9 rounded-xl bg-amber-600 flex items-center justify-center font-bold">P</div>
            <div>
                <h1 class="font-bold text-sm">Persona Lab</h1>
                <p class="text-[10px] text-stone-400">AI Prompt Experiments</p>
            </div>
        </div>
    </div>

    <button type="button" id="newPromptBtn"
        class="primary-btn w-full mb-8">+ New Prompt</button>

    <div class="flex-1">
        <p class="text-[10px] uppercase tracking-widest text-stone-500 font-bold mb-3">
            Recent Experiments
        </p>
        <div id="historyList" class="space-y-1">
            <p class="text-xs text-stone-500 px-2">
                Your submitted questions will appear here.
            </p>
        </div>
    </div>

    <div class="border-t border-stone-700 pt-4 mt-5">
        <p class="text-[10px] text-stone-500 text-center">
            Experiment 06<br>Role-Based & Negative Prompting
        </p>
    </div>
</aside>


<!-- MAIN AREA -->
<div class="flex-1 min-w-0">

<header class="bg-white border-b border-stone-200 px-5 md:px-9 py-5">
    <div class="max-w-7xl mx-auto flex justify-between items-center gap-4">
        <div>
            <p class="text-[10px] text-amber-700 uppercase tracking-widest font-bold mb-1">
                Prompt Engineering / Experiment 06
            </p>
            <h2 class="text-lg md:text-xl font-bold">
                Role-Based & Negative Prompting
            </h2>
            <p class="text-xs text-stone-500 mt-1">
                Create an AI persona and control its response using negative instructions.
            </p>
        </div>
        <button type="button" id="resetBtn" class="secondary-btn">Reset</button>
    </div>
</header>


<main class="max-w-7xl mx-auto px-4 md:px-9 py-7 space-y-6">

    <section class="rounded-2xl bg-[#f4eadb] border border-[#ead9bd] p-5 md:p-6">
        <h3 class="font-bold text-sm">✦ Welcome to your AI Persona Experiment</h3>
        <p class="text-xs leading-6 text-stone-600 mt-2">
            Choose any role, describe its behaviour, add restrictions and ask a question.
            Gemini generates two responses so you can observe the effect of negative prompting.
        </p>
    </section>


    <!-- STEP 1 -->
    <section class="card p-5 md:p-6 space-y-5">
        <div class="flex items-center gap-3">
            <span class="step-number">1</span>
            <div>
                <h3 class="font-bold text-sm">Configure Your AI Persona</h3>
                <p class="text-xs text-stone-500 mt-1">Define the role and behaviour of your AI.</p>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
                <label class="block text-xs font-semibold mb-2">Persona Role *</label>
                <input id="inputRole" class="input-field"
                    placeholder="Example: Experienced Python teacher">
            </div>

            <div>
                <label class="block text-xs font-semibold mb-2">Area of Expertise</label>
                <input id="inputExpertise" class="input-field"
                    placeholder="Example: Python and programming">
            </div>

            <div>
                <label class="block text-xs font-semibold mb-2">Personality</label>
                <input id="inputPersonality" class="input-field"
                    placeholder="Example: Friendly, patient and supportive">
            </div>

            <div>
                <label class="block text-xs font-semibold mb-2">Communication Tone</label>
                <select id="inputTone" class="input-field">
                    <option>Professional</option>
                    <option>Friendly</option>
                    <option>Simple and Beginner Friendly</option>
                    <option>Formal</option>
                    <option>Technical</option>
                    <option>Conversational</option>
                    <option>Academic</option>
                </select>
            </div>
        </div>

        <p class="text-xs text-stone-500">
            Example roles: Teacher, Doctor, Software Engineer, Career Counselor,
            Chef, Researcher or any custom persona.
        </p>
    </section>


    <!-- STEP 2 -->
    <section class="card p-5 md:p-6 space-y-4">
        <div class="flex items-center gap-3">
            <span class="step-number">2</span>
            <div>
                <h3 class="font-bold text-sm">Negative Prompt Configuration</h3>
                <p class="text-xs text-stone-500 mt-1">Specify what the AI should avoid.</p>
            </div>
        </div>

        <label class="block text-xs font-semibold mb-2">Negative Instructions *</label>
        <textarea id="inputNegative" rows="4" class="input-field"
            placeholder="Example:&#10;- Do not use complicated language.&#10;- Do not mention brand names.&#10;- Do not provide unnecessary information."></textarea>

        <p class="text-[11px] text-stone-500">
            You can provide multiple restrictions.
        </p>
    </section>


    <!-- STEP 3 -->
    <section class="card p-5 md:p-6 space-y-4">
        <div class="flex items-center gap-3">
            <span class="step-number">3</span>
            <div>
                <h3 class="font-bold text-sm">Ask Your Question</h3>
                <p class="text-xs text-stone-500 mt-1">
                    Both responses will use the same question.
                </p>
            </div>
        </div>

        <label class="block text-xs font-semibold mb-2">Enter Your Question *</label>
        <textarea id="inputQuery" rows="3" class="input-field"
            placeholder="Ask anything related to your selected persona..."></textarea>

        <div class="flex flex-wrap justify-between items-center gap-3">
            <p class="text-[11px] text-stone-500">Generate two responses for comparison.</p>
            <button type="button" id="generateBtn" class="primary-btn">
                Generate & Compare →
            </button>
        </div>

        <div id="errorBox" role="alert"
            class="hidden text-xs text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
        </div>

        <div id="loadingBox"
            class="hidden text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">
            <span class="spinner align-middle mr-2"></span>
            Gemini is generating your responses. Please wait...
        </div>
    </section>


    <!-- STEP 4 -->
    <section id="promptSection" class="hidden-section space-y-4">
        <div class="flex items-center gap-3">
            <span class="step-number">4</span>
            <div>
                <h3 class="font-bold text-sm">Generated Prompt Structures</h3>
                <p class="text-xs text-stone-500 mt-1">Inspect the prompts sent to Gemini.</p>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="card p-5 role-top">
                <h4 class="font-bold text-sm mb-3 text-amber-800">Role-Based Prompt</h4>
                <div id="rolePromptPreview" class="prompt-preview bg-amber-50 rounded-xl p-4"></div>
            </div>

            <div class="card p-5 combined-top">
                <h4 class="font-bold text-sm mb-3 text-purple-800">Role + Negative Prompt</h4>
                <div id="combinedPromptPreview" class="prompt-preview bg-purple-50 rounded-xl p-4"></div>
            </div>
        </div>
    </section>


    <!-- STEP 5 -->
    <section id="resultsSection" class="hidden-section space-y-4">
        <div class="flex items-center gap-3">
            <span class="step-number">5</span>
            <div>
                <h3 class="font-bold text-sm">AI Response Comparison</h3>
                <p class="text-xs text-stone-500 mt-1">Compare both generated responses.</p>
            </div>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-2 gap-5">

            <div class="card p-5 role-top">
                <span class="text-[10px] uppercase tracking-widest font-bold text-amber-700">
                    Response A
                </span>
                <h3 class="font-bold text-base mt-1">Role-Based Response</h3>
                <p id="roleLabel" class="text-[11px] text-stone-500 mt-1"></p>

                <div id="roleOutput" class="output-box bg-stone-50 rounded-xl border border-stone-100 p-4 mt-4"></div>

                <button type="button" id="copyRoleBtn" class="secondary-btn mt-4">
                    Copy Response
                </button>
            </div>

            <div class="card p-5 combined-top">
                <span class="text-[10px] uppercase tracking-widest font-bold text-purple-700">
                    Response B
                </span>
                <h3 class="font-bold text-base mt-1">Role + Negative Response</h3>
                <p class="text-[11px] text-stone-500 mt-1">Persona with restrictions applied</p>

                <div id="combinedOutput" class="output-box bg-stone-50 rounded-xl border border-stone-100 p-4 mt-4"></div>

                <button type="button" id="copyCombinedBtn" class="secondary-btn mt-4">
                    Copy Response
                </button>
            </div>

        </div>
    </section>


    <!-- STEP 6 -->
    <section id="evaluationSection" class="hidden-section card p-5 md:p-6 space-y-4">
        <div class="flex items-center gap-3">
            <span class="step-number">6</span>
            <div>
                <h3 class="font-bold text-sm">Experiment Evaluation</h3>
                <p class="text-xs text-stone-500 mt-1">Review the differences between both responses.</p>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="bg-amber-50 rounded-xl p-4">
                <h4 class="text-xs font-bold text-amber-900 mb-2">Role-Based Prompting</h4>
                <p class="text-xs text-stone-700 leading-6">
                    Observe whether the answer reflects the selected role,
                    expertise and communication style.
                </p>
            </div>

            <div class="bg-purple-50 rounded-xl p-4">
                <h4 class="text-xs font-bold text-purple-900 mb-2">Negative Prompting</h4>
                <p class="text-xs text-stone-700 leading-6">
                    Check whether the second answer avoids the content specified
                    in your negative instructions.
                </p>
            </div>
        </div>

        <div class="border border-stone-200 rounded-xl p-4">
            <h4 class="text-xs font-bold mb-2">What to observe</h4>
            <ul class="text-xs text-stone-600 leading-7 list-disc pl-5">
                <li>Did the AI adapt to the selected role?</li>
                <li>Did the communication style match the persona?</li>
                <li>Did the second response follow negative instructions?</li>
                <li>Did the restrictions change the answer's detail or usefulness?</li>
                <li>Were any restrictions missed?</li>
            </ul>
        </div>
    </section>


    <footer class="text-center text-[10px] text-stone-400 py-5">
        Persona Lab · Experiment 06 · Powered by Google Gemini
    </footer>

</main>
</div>


<script>
document.addEventListener("DOMContentLoaded", function () {

    const $ = (id) => document.getElementById(id);

    const roleInput = $("inputRole");
    const expertiseInput = $("inputExpertise");
    const personalityInput = $("inputPersonality");
    const toneInput = $("inputTone");
    const negativeInput = $("inputNegative");
    const queryInput = $("inputQuery");

    const generateBtn = $("generateBtn");
    const errorBox = $("errorBox");
    const loadingBox = $("loadingBox");

    let experimentHistory = [];

    // ---------------- GENERATE ----------------

    async function generateOutputs() {

        const role = roleInput.value.trim();
        const expertise = expertiseInput.value.trim();
        const personality = personalityInput.value.trim();
        const tone = toneInput.value;
        const negative = negativeInput.value.trim();
        const question = queryInput.value.trim();

        errorBox.classList.add("hidden");
        errorBox.textContent = "";

        if (!role || !negative || !question) {
            errorBox.textContent =
                "Please complete Persona Role, Negative Instructions and Question.";
            errorBox.classList.remove("hidden");
            return;
        }

        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";
        loadingBox.classList.remove("hidden");

        $("resultsSection").classList.add("hidden-section");
        $("promptSection").classList.add("hidden-section");
        $("evaluationSection").classList.add("hidden-section");

        try {

            console.log("Sending request to Flask...");

            const response = await fetch("/api/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    role,
                    expertise,
                    personality,
                    tone,
                    negative,
                    question
                })
            });

            const data = await response.json();

            console.log("Flask response:", data);

            if (!response.ok) {
                throw new Error(data.error || "Server request failed.");
            }

            if (!data.success) {
                throw new Error("Gemini did not return successful results.");
            }

            $("rolePromptPreview").textContent = data.role_prompt;
            $("combinedPromptPreview").textContent = data.combined_prompt;

            $("roleOutput").textContent = data.role_response;
            $("combinedOutput").textContent = data.combined_response;

            $("roleLabel").textContent = "Persona: " + data.role;

            $("promptSection").classList.remove("hidden-section");
            $("resultsSection").classList.remove("hidden-section");
            $("evaluationSection").classList.remove("hidden-section");

            addHistory(question, role);

            $("resultsSection").scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        } catch (error) {

            console.error("Generation error:", error);

            errorBox.textContent = "Error: " + error.message;
            errorBox.classList.remove("hidden");

        } finally {

            generateBtn.disabled = false;
            generateBtn.textContent = "Generate & Compare →";
            loadingBox.classList.add("hidden");

        }
    }


    // ---------------- RESET ----------------

    function resetForm() {

        roleInput.value = "";
        expertiseInput.value = "";
        personalityInput.value = "";
        toneInput.value = "Professional";
        negativeInput.value = "";
        queryInput.value = "";

        [
            "roleOutput",
            "combinedOutput",
            "rolePromptPreview",
            "combinedPromptPreview",
            "roleLabel"
        ].forEach(id => $(id).textContent = "");

        [
            "resultsSection",
            "promptSection",
            "evaluationSection"
        ].forEach(id => $(id).classList.add("hidden-section"));

        errorBox.classList.add("hidden");
        errorBox.textContent = "";

        window.scrollTo({ top: 0, behavior: "smooth" });
    }


    // ---------------- HISTORY ----------------

    function addHistory(question, role) {

        experimentHistory.unshift({
            question,
            role,
            expertise: expertiseInput.value,
            personality: personalityInput.value,
            tone: toneInput.value,
            negative: negativeInput.value
        });

        experimentHistory = experimentHistory.slice(0, 8);

        const historyList = $("historyList");
        historyList.innerHTML = "";

        experimentHistory.forEach(item => {

            const button = document.createElement("button");

            button.type = "button";
            button.className = "history-item";
            button.textContent = item.question;
            button.title = item.question;

            button.addEventListener("click", function () {
                roleInput.value = item.role;
                expertiseInput.value = item.expertise;
                personalityInput.value = item.personality;
                toneInput.value = item.tone;
                negativeInput.value = item.negative;
                queryInput.value = item.question;

                window.scrollTo({ top: 0, behavior: "smooth" });
            });

            historyList.appendChild(button);
        });
    }


    // ---------------- COPY ----------------

    async function copyText(elementId, buttonId) {

        const text = $(elementId).textContent;

        if (!text.trim()) return;

        try {

            await navigator.clipboard.writeText(text);

            const button = $(buttonId);
            const original = button.textContent;

            button.textContent = "Copied!";

            setTimeout(() => {
                button.textContent = original;
            }, 1500);

        } catch (error) {
            alert("Copy failed. Please select and copy the response manually.");
        }
    }


    // ---------------- BUTTON EVENTS ----------------

    generateBtn.addEventListener("click", generateOutputs);

    $("newPromptBtn").addEventListener("click", resetForm);
    $("resetBtn").addEventListener("click", resetForm);

    $("copyRoleBtn").addEventListener("click", function () {
        copyText("roleOutput", "copyRoleBtn");
    });

    $("copyCombinedBtn").addEventListener("click", function () {
        copyText("combinedOutput", "copyCombinedBtn");
    });

    console.log("Persona Lab frontend initialized.");

});
</script>

</body>
</html>
"""


# ---------------- START SERVER ----------------

if __name__ == "__main__":

    if not API_KEY:
        print("WARNING: GEMINI_API_KEY is missing in .env")

    print("------------------------------------")
    print("PERSONA LAB - EXPERIMENT 06")
    print("Model:", MODEL_NAME)
    print("API key configured:", bool(API_KEY))
    print("Open: http://127.0.0.1:5000")
    print("------------------------------------")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
