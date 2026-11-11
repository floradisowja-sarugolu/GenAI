
from flask import Flask, request, jsonify, render_template_string
import requests
import re
import time

app = Flask(__name__)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_with_ollama(prompt):
    """Send a prompt to the locally running Ollama model."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
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


def build_prompts(topic):
    """Create two prompts that request different output structures."""

    bullet_prompt = f"""
You are a helpful educational assistant.

Explain the following topic using a bullet list.

Topic: {topic}

Instructions:
- Give exactly 3 bullet points.
- Each bullet must explain one important point.
- Use simple and clear English.
- Do not write an introduction or conclusion.
- Return only the bullet list.
"""

    table_prompt = f"""
You are a helpful educational assistant.

Explain the following topic using a Markdown table.

Topic: {topic}

Instructions:
- Create a Markdown table.
- Use exactly these columns: Benefit and Description.
- Include exactly 3 data rows.
- Use simple and clear English.
- Do not write an introduction or conclusion.
- Return only the Markdown table.
"""

    return bullet_prompt, table_prompt


def analyze_output(text, output_format):
    """Check whether the generated response follows the requested format."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if output_format == "bullet":
        bullet_lines = [
            line for line in lines
            if re.match(r"^[-*•]\s+", line)
        ]

        checks = {
            "Bullet list detected": len(bullet_lines) > 0,
            "Exactly 3 bullet points": len(bullet_lines) == 3,
            "Non-empty response": len(text.strip()) > 0
        }

    else:
        table_lines = [
            line for line in lines
            if "|" in line
        ]

        header_detected = any(
            "benefit" in line.lower()
            and "description" in line.lower()
            for line in table_lines
        )

        separator_detected = any(
            re.match(r"^\|?\s*:?-{3,}", line)
            for line in table_lines
        )

        data_rows = []

        if header_detected and separator_detected:
            separator_index = next(
                (
                    i for i, line in enumerate(table_lines)
                    if re.match(r"^\|?\s*:?-{3,}", line)
                ),
                -1
            )

            if separator_index >= 0:
                data_rows = table_lines[separator_index + 1:]

        checks = {
            "Markdown table detected": len(table_lines) >= 3,
            "Required columns detected": header_detected,
            "Table separator detected": separator_detected,
            "Exactly 3 data rows": len(data_rows) == 3,
            "Non-empty response": len(text.strip()) > 0
        }

    passed = sum(1 for value in checks.values() if value)
    total = len(checks)

    score = round((passed / total) * 100)

    return {
        "checks": checks,
        "score": score,
        "passed": passed,
        "total": total
    }


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/structured", methods=["POST"])
def structured():
    data = request.get_json(silent=True) or {}

    topic = data.get("topic", "").strip()

    if len(topic) < 3:
        return jsonify({
            "error": "Please enter a topic with at least 3 characters."
        }), 400

    bullet_prompt, table_prompt = build_prompts(topic)

    try:
        start_time = time.time()

        # Generate bullet-list response
        bullet_response = generate_with_ollama(bullet_prompt)

        # Generate Markdown-table response
        table_response = generate_with_ollama(table_prompt)

        if not bullet_response or not table_response:
            return jsonify({
                "error": "Ollama returned an empty response. Please try again."
            }), 502

        bullet_analysis = analyze_output(
            bullet_response, "bullet"
        )

        table_analysis = analyze_output(
            table_response, "table"
        )

        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            "topic": topic,
            "model": MODEL_NAME,
            "bullet": {
                "prompt": bullet_prompt,
                "response": bullet_response,
                "analysis": bullet_analysis
            },
            "table": {
                "prompt": table_prompt,
                "response": table_response,
                "analysis": table_analysis
            },
            "time": elapsed
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": (
                "Cannot connect to Ollama. Make sure Ollama is installed "
                "and running, then try again."
            )
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "error": (
                "The model is taking too long to respond. "
                "Try again or use a smaller model."
            )
        }), 504

    except requests.exceptions.HTTPError as e:
        return jsonify({
            "error": f"Ollama HTTP error: {str(e)}"
        }), 502

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Structured Format Studio</title>

<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    background: #f3f5f9;
    color: #1e293b;
    height: 100vh;
}

.header {
    height: 65px;
    background: #111c35;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 25px;
}

.logo {
    font-size: 19px;
    font-weight: bold;
}

.header-right {
    font-size: 12px;
    color: #a9c7d9;
}

.app {
    display: grid;
    grid-template-columns: 330px 1fr;
    height: calc(100vh - 65px);
    gap: 16px;
    padding: 16px;
}

.sidebar {
    background: white;
    border: 1px solid #e1e6ee;
    border-radius: 12px;
    padding: 22px;
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.sidebar h2 {
    font-size: 19px;
}

.subheading {
    color: #778398;
    font-size: 13px;
    line-height: 1.5;
}

label {
    font-size: 13px;
    font-weight: bold;
    display: block;
    margin-bottom: 9px;
}

textarea {
    width: 100%;
    height: 155px;
    resize: vertical;
    border: 1px solid #d8deea;
    border-radius: 8px;
    padding: 13px;
    font-family: inherit;
    font-size: 14px;
    outline: none;
    line-height: 1.5;
}

textarea:focus {
    border-color: #4979dc;
}

select {
    width: 100%;
    padding: 12px;
    border: 1px solid #d8deea;
    border-radius: 8px;
    background: white;
    font-size: 14px;
}

button {
    border: none;
    border-radius: 8px;
    padding: 13px;
    font-size: 14px;
    font-weight: bold;
    cursor: pointer;
}

.generate-btn {
    background: #315fce;
    color: white;
}

.generate-btn:hover {
    background: #244eaf;
}

.clear-btn {
    background: #eef1f6;
    color: #44516a;
}

button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.sidebar-footer {
    margin-top: auto;
    padding-top: 15px;
    border-top: 1px solid #edf0f4;
    font-size: 12px;
    color: #8791a2;
    line-height: 1.6;
}

.chat-panel {
    background: white;
    border: 1px solid #e1e6ee;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    min-width: 0;
    overflow: hidden;
}

.chat-header {
    height: 65px;
    border-bottom: 1px solid #edf0f4;
    padding: 0 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-header h2 {
    font-size: 17px;
}

.chat-status {
    color: #15966a;
    font-size: 12px;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 25px;
}

.empty-state {
    height: 100%;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: #8993a4;
    gap: 12px;
}

.empty-icon {
    width: 58px;
    height: 58px;
    background: #edf2ff;
    color: #315fce;
    border-radius: 18px;
    display: grid;
    place-items: center;
    font-size: 25px;
}

.empty-state h3 {
    color: #34415a;
    font-size: 18px;
}

.empty-state p {
    max-width: 340px;
    font-size: 13px;
    line-height: 1.6;
}

.conversation {
    margin-bottom: 32px;
}

.user-message {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 22px;
}

.user-bubble {
    max-width: 75%;
    padding: 13px 17px;
    background: #315fce;
    color: white;
    border-radius: 14px 14px 3px 14px;
    font-size: 14px;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.ai-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 13px;
}

.ai-avatar {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: #e8f7f1;
    color: #15966a;
    display: grid;
    place-items: center;
    font-weight: bold;
}

.ai-heading strong {
    font-size: 14px;
}

.ai-heading span {
    color: #8993a4;
    font-size: 11px;
    margin-left: 5px;
}

.results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-left: 42px;
}

.result-card {
    border: 1px solid #e4e9f0;
    border-radius: 10px;
    overflow: hidden;
    min-width: 0;
}

.result-title {
    padding: 13px 15px;
    background: #f7f9fc;
    border-bottom: 1px solid #e4e9f0;
    font-size: 13px;
    font-weight: bold;
}

.result-content {
    padding: 16px;
    font-size: 13px;
    line-height: 1.7;
    overflow-wrap: anywhere;
    white-space: normal;
}

.result-content ul {
    padding-left: 20px;
}

.result-content li {
    margin-bottom: 9px;
}

.result-content table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.result-content th,
.result-content td {
    border: 1px solid #dce2eb;
    padding: 9px;
    text-align: left;
}

.result-content th {
    background: #edf2ff;
}

.audit {
    padding: 12px 15px;
    border-top: 1px solid #e9edf2;
    background: #fbfcfe;
}

.audit-top {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 9px;
}

.score {
    color: #15966a;
}

.score.warning {
    color: #d97706;
}

.check-row {
    font-size: 11px;
    color: #677387;
    margin-top: 5px;
}

.check-pass {
    color: #15966a;
}

.check-fail {
    color: #dc5252;
}

.time-label {
    margin: 12px 0 0 42px;
    font-size: 11px;
    color: #98a1af;
}

.loading {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #6c7890;
    font-size: 13px;
    padding: 15px 0;
}

.spinner {
    width: 17px;
    height: 17px;
    border: 2px solid #dce5f7;
    border-top-color: #315fce;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.error {
    color: #c43f3f;
    background: #fff0f0;
    border: 1px solid #ffdada;
    border-radius: 8px;
    padding: 11px;
    font-size: 12px;
    line-height: 1.5;
    display: none;
}

@media(max-width: 950px) {
    .app {
        grid-template-columns: 270px 1fr;
    }

    .results-grid {
        grid-template-columns: 1fr;
    }
}

@media(max-width: 700px) {
    body {
        height: auto;
    }

    .app {
        display: flex;
        flex-direction: column;
        height: auto;
    }

    .sidebar {
        min-height: 400px;
    }

    .chat-panel {
        min-height: 650px;
    }

    .results-grid {
        margin-left: 0;
    }

    .user-bubble {
        max-width: 90%;
    }
}
</style>
</head>

<body>

<header class="header">
    <div class="logo">Structured Format Studio</div>
    <div class="header-right">LOCAL AI · OLLAMA</div>
</header>

<main class="app">

    <aside class="sidebar">

        <div>
            <h2>Ask a Question</h2>
            <p class="subheading">
                Generate and compare structured AI responses.
            </p>
        </div>

        <div>
            <label for="topic">Enter your topic or question</label>
            <textarea
                id="topic"
                placeholder="Example: Explain three benefits of daily exercise..."
            ></textarea>
        </div>

        <div>
            <label for="model">AI Model</label>
            <select id="model">
                <option value="llama3.2:3b">Llama 3.2 (3B)</option>
            </select>
        </div>

        <button class="generate-btn" id="generateBtn" onclick="generateAnswer()">
            Generate Answer
        </button>

        <button class="clear-btn" onclick="clearConversation()">
            Clear Conversation
        </button>

        <div class="error" id="errorBox"></div>

        <div class="sidebar-footer">
            <strong>Connected to Ollama</strong><br>
            Model: llama3.2:3b<br>
            Local generation through your computer.
        </div>

    </aside>

    <section class="chat-panel">

        <div class="chat-header">
            <h2>AI Conversation</h2>
            <div class="chat-status">● Local model</div>
        </div>

        <div class="chat-messages" id="chatMessages">

            <div class="empty-state" id="emptyState">
                <div class="empty-icon">✦</div>
                <h3>Start a conversation</h3>
                <p>
                    Enter a question on the left. The AI will generate
                    a bullet list and a Markdown table for comparison.
                </p>
            </div>

        </div>

    </section>

</main>

<script>
const topicInput = document.getElementById("topic");
const generateBtn = document.getElementById("generateBtn");
const chatMessages = document.getElementById("chatMessages");
const errorBox = document.getElementById("errorBox");

function addText(parent, tag, text, className) {
    const element = document.createElement(tag);
    element.textContent = text;

    if (className) {
        element.className = className;
    }

    parent.appendChild(element);
    return element;
}

function createAudit(analysis) {
    const audit = document.createElement("div");
    audit.className = "audit";

    const top = document.createElement("div");
    top.className = "audit-top";

    addText(top, "span", "Format verification");

    const score = document.createElement("span");
    score.className = analysis.score === 100
        ? "score"
        : "score warning";

    score.textContent = analysis.score + "%";
    top.appendChild(score);

    audit.appendChild(top);

    Object.entries(analysis.checks).forEach(([name, passed]) => {
        const row = document.createElement("div");

        row.className = passed
            ? "check-row check-pass"
            : "check-row check-fail";

        row.textContent = (passed ? "✓ " : "✕ ") + name;
        audit.appendChild(row);
    });

    return audit;
}

function renderBulletList(text, container) {
    const lines = text.split("\n").filter(line => line.trim());

    const list = document.createElement("ul");

    let foundBullet = false;

    lines.forEach(line => {
        const match = line.match(/^[-*•]\s+(.+)$/);

        if (match) {
            foundBullet = true;
            addText(list, "li", match[1]);
        }
    });

    if (foundBullet) {
        container.appendChild(list);
    } else {
        addText(container, "p", text);
    }
}

function renderMarkdownTable(text, container) {
    const lines = text
        .split("\n")
        .map(line => line.trim())
        .filter(line => line.includes("|"));

    if (lines.length < 2) {
        addText(container, "p", text);
        return;
    }

    function parseCells(line) {
        return line
            .replace(/^\|/, "")
            .replace(/\|$/, "")
            .split("|")
            .map(cell => cell.trim());
    }

    const table = document.createElement("table");
    const header = document.createElement("thead");
    const headerRow = document.createElement("tr");

    parseCells(lines[0]).forEach(cell => {
        addText(headerRow, "th", cell);
    });

    header.appendChild(headerRow);
    table.appendChild(header);

    const body = document.createElement("tbody");

    lines.slice(2).forEach(line => {
        const cells = parseCells(line);

        if (!cells.length) return;

        const row = document.createElement("tr");

        cells.forEach(cell => {
            addText(row, "td", cell);
        });

        body.appendChild(row);
    });

    table.appendChild(body);
    container.appendChild(table);
}

function createResultCard(title, text, analysis, type) {
    const card = document.createElement("div");
    card.className = "result-card";

    addText(card, "div", title, "result-title");

    const content = document.createElement("div");
    content.className = "result-content";

    if (type === "bullet") {
        renderBulletList(text, content);
    } else {
        renderMarkdownTable(text, content);
    }

    card.appendChild(content);
    card.appendChild(createAudit(analysis));

    return card;
}

function addConversation(topic, data) {
    const conversation = document.createElement("div");
    conversation.className = "conversation";

    const userMessage = document.createElement("div");
    userMessage.className = "user-message";

    addText(userMessage, "div", topic, "user-bubble");
    conversation.appendChild(userMessage);

    const aiHeading = document.createElement("div");
    aiHeading.className = "ai-heading";

    addText(aiHeading, "div", "AI", "ai-avatar");

    const headingText = document.createElement("div");
    addText(headingText, "strong", "Structured AI Response");

    const modelText = document.createElement("span");
    modelText.textContent = data.model;
    headingText.appendChild(modelText);

    aiHeading.appendChild(headingText);
    conversation.appendChild(aiHeading);

    const resultsGrid = document.createElement("div");
    resultsGrid.className = "results-grid";

    resultsGrid.appendChild(
        createResultCard(
            "Bullet List Format",
            data.bullet.response,
            data.bullet.analysis,
            "bullet"
        )
    );

    resultsGrid.appendChild(
        createResultCard(
            "Markdown Table Format",
            data.table.response,
            data.table.analysis,
            "table"
        )
    );

    conversation.appendChild(resultsGrid);

    addText(
        conversation,
        "div",
        "Generated in " + data.time + " seconds",
        "time-label"
    );

    document.getElementById("emptyState")?.remove();

    chatMessages.appendChild(conversation);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function generateAnswer() {
    const topic = topicInput.value.trim();

    errorBox.style.display = "none";
    errorBox.textContent = "";

    if (topic.length < 3) {
        errorBox.textContent =
            "Please enter a topic with at least 3 characters.";
        errorBox.style.display = "block";
        return;
    }

    generateBtn.disabled = true;
    generateBtn.textContent = "Generating...";

    const loading = document.createElement("div");
    loading.className = "loading";

    const spinner = document.createElement("div");
    spinner.className = "spinner";

    loading.appendChild(spinner);
    addText(loading, "span", "Ollama is generating two responses...");

    chatMessages.appendChild(loading);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch("/api/structured", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                topic: topic
            })
        });

        const data = await response.json();

        loading.remove();

        if (!response.ok) {
            throw new Error(data.error || "Something went wrong.");
        }

        addConversation(topic, data);

        topicInput.value = "";

    } catch (error) {
        loading.remove();

        errorBox.textContent = error.message;
        errorBox.style.display = "block";

    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate Answer";
    }
}

function clearConversation() {
    chatMessages.innerHTML = `
        <div class="empty-state" id="emptyState">
            <div class="empty-icon">✦</div>
            <h3>Start a conversation</h3>
            <p>
                Enter a question on the left. The AI will generate
                a bullet list and a Markdown table for comparison.
            </p>
        </div>
    `;

    topicInput.value = "";
    errorBox.style.display = "none";
}

topicInput.addEventListener("keydown", function(event) {
    if (event.key === "Enter" && event.ctrlKey) {
        generateAnswer();
    }
});
</script>

</body>
</html>
"""


if __name__ == "__main__":
    print("Starting Experiment 8 - Structured Format Prompting")
    print("Ollama model:", MODEL_NAME)
    print("Open http://127.0.0.1:5000 in your browser")

    app.run(debug=True, host="127.0.0.1", port=5000)
