
import streamlit as st
import requests
import pandas as pd
import altair as alt
import re
import time
from datetime import datetime


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="NexusPrompt IQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


# ==========================================================
# 2. LIGHT THEME AND CSS
# ==========================================================

st.markdown("""
<style>

.stApp {
    background: #f6f8fc;
    color: #263247;
}

.main .block-container {
    max-width: 1550px;
    padding: 1.8rem 2rem 3rem 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

h1, h2, h3 {
    color: #17243a !important;
    font-weight: 650 !important;
}

h1 {
    font-size: 2.1rem !important;
}

h2 {
    font-size: 1.4rem !important;
}

h3 {
    font-size: 1.1rem !important;
}

p, label {
    color: #45536a;
}

.app-subtitle {
    color: #7b879b;
    font-size: 14px;
    margin-top: -12px;
    margin-bottom: 20px;
}

.section-heading {
    color: #24344d;
    font-size: 18px;
    font-weight: 650;
    margin-bottom: 12px;
}

.panel {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 12px;
    padding: 18px;
}

.panel-caption {
    font-size: 12px;
    color: #8995a8;
    margin-top: -8px;
    margin-bottom: 14px;
}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: #ffffff !important;
    color: #263247 !important;
    border: 1px solid #d7deea !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus {
    border-color: #3b82f6 !important;
}

div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d7deea !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    color: #263247 !important;
}

div.stButton > button {
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
    border: 1px solid #d7deea;
    background: #ffffff;
    color: #34445d;
}

div.stButton > button[kind="primary"] {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #2563eb;
}

div.stButton > button[kind="primary"]:hover {
    background: #1d4ed8;
    color: #ffffff;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    color: #2563eb !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 18px;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #2563eb !important;
}

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 9px;
}

.output-card {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 12px;
}

.small-muted {
    color: #8995a8;
    font-size: 12px;
}

hr {
    border-color: #e5eaf2 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. SESSION STATE
# ==========================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

if "run_errors" not in st.session_state:
    st.session_state.run_errors = []


# ==========================================================
# 4. OLLAMA CONNECTION
# ==========================================================

def check_ollama():

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        response.raise_for_status()

        models = response.json().get("models", [])

        return [
            item["name"] for item in models
        ]

    except Exception:
        return []


# ==========================================================
# 5. GENERATE RESPONSE
# ==========================================================

def generate_response(
    prompt,
    user_input,
    temperature,
    max_tokens
):

    full_prompt = f"""
INSTRUCTIONS:
{prompt}

USER INPUT:
{user_input}

Follow the instructions carefully.
Give a clear and useful response.
"""

    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    start = time.time()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    answer = response.json().get(
        "response", ""
    ).strip()

    duration = round(time.time() - start, 2)

    if not answer:
        raise Exception("The model returned an empty response.")

    return answer, duration


# ==========================================================
# 6. KEYWORD PROCESSING
# ==========================================================

STOPWORDS = {
    "the", "and", "for", "are", "was", "were",
    "with", "this", "that", "from", "have", "has",
    "into", "its", "their", "they", "them", "you",
    "your", "about", "what", "when", "where", "which",
    "will", "would", "could", "should", "can", "not",
    "but", "also", "than", "then", "there", "these",
    "those", "such", "some", "more", "most", "very",
    "using", "used", "use", "each", "both", "given"
}


def get_keywords(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return {
        word for word in words
        if len(word) > 2 and word not in STOPWORDS
    }


def calculate_overlap(required, actual):

    required_words = get_keywords(required)
    actual_words = get_keywords(actual)

    if not required_words:
        return None, [], []

    matched = required_words.intersection(actual_words)

    missing = required_words.difference(actual_words)

    score = round(
        len(matched) / len(required_words) * 5,
        2
    )

    return score, sorted(matched), sorted(missing)


# ==========================================================
# 7. FORMAT VALIDATION
# ==========================================================

def check_format(response, expected_format):

    if expected_format == "Any format":
        return None, "No format requirement selected."

    if expected_format == "Bullet points":
        passed = bool(
            re.search(r"(?m)^\s*[-*•]\s+", response)
        )

    elif expected_format == "Numbered list":
        passed = bool(
            re.search(r"(?m)^\s*\d+[.)]\s+", response)
        )

    elif expected_format == "JSON":

        try:
            import json
            json.loads(response)
            passed = True
        except Exception:
            passed = False

    elif expected_format == "Table":

        passed = (
            "|" in response
            and bool(re.search(r"\|[-: ]+\|", response))
        )

    elif expected_format == "Paragraph":
        passed = not bool(
            re.search(r"(?m)^\s*[-*•]\s+", response)
        )

    else:
        passed = True

    score = 5 if passed else 0

    status = "PASS" if passed else "FAIL"

    return score, status


# ==========================================================
# 8. EVALUATION FUNCTION
# ==========================================================

def evaluate_output(
    response,
    user_input,
    reference_keywords,
    expected_format
):

    relevance, matched_input, missing_input = calculate_overlap(
        user_input,
        response
    )

    accuracy, matched_reference, missing_reference = calculate_overlap(
        reference_keywords,
        response
    )

    compliance, format_status = check_format(
        response,
        expected_format
    )

    available_scores = [
        value for value in [
            accuracy,
            relevance,
            compliance
        ]
        if value is not None
    ]

    overall = round(
        sum(available_scores) / len(available_scores),
        2
    ) if available_scores else 0

    return {
        "accuracy": accuracy,
        "relevance": relevance,
        "format_compliance": compliance,
        "overall": overall,
        "matched_input": matched_input,
        "missing_input": missing_input,
        "matched_reference": matched_reference,
        "missing_reference": missing_reference,
        "format_status": format_status
    }


# ==========================================================
# 9. PROMPT COMPARISON
# ==========================================================

def average_metrics(results, prompt_tag):

    selected = [
        item for item in results
        if item["prompt_tag"] == prompt_tag
    ]

    metric_names = [
        "accuracy",
        "relevance",
        "format_compliance",
        "overall"
    ]

    output = {}

    for metric in metric_names:

        values = [
            item["scores"][metric]
            for item in selected
            if item["scores"][metric] is not None
        ]

        output[metric] = (
            round(sum(values) / len(values), 2)
            if values else None
        )

    return output


def make_comparison(results):

    a = average_metrics(results, "Prompt A")
    b = average_metrics(results, "Prompt B")

    rows = []

    for metric in [
        "accuracy",
        "relevance",
        "format_compliance",
        "overall"
    ]:

        av = a[metric]
        bv = b[metric]

        if av is None or bv is None:
            winner = "Not evaluated"

        elif av > bv:
            winner = "Prompt A"

        elif bv > av:
            winner = "Prompt B"

        else:
            winner = "Tie"

        rows.append({
            "Metric": metric.replace("_", " ").title(),
            "Prompt A": av,
            "Prompt B": bv,
            "Better Result": winner
        })

    return a, b, pd.DataFrame(rows)


# ==========================================================
# 10. HEADER
# ==========================================================

header_left, header_right = st.columns([3, 1])

with header_left:

    st.title("🧠 NexusPrompt IQ")

    st.markdown(
        '<div class="app-subtitle">'
        'Dual-Prompt Heuristic Evaluation Studio'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        "<div style='text-align:right; padding-top:15px;'>"
        "<span style='color:#718096;font-size:13px;'>"
        "EXPERIMENT 16(A)</span></div>",
        unsafe_allow_html=True
    )


models = check_ollama()

model_available = any(
    name == MODEL or name.startswith(MODEL + ":")
    for name in models
)

if model_available:

    st.success(
        f"Connected to Ollama · {MODEL}"
    )

else:

    st.error(
        "Ollama model is not available. Start Ollama and check the model."
    )

st.divider()


# ==========================================================
# 11. MAIN INPUT AND OUTPUT LAYOUT
# ==========================================================

input_column, output_column = st.columns(
    [1, 1.15],
    gap="large"
)


# ==========================================================
# 12. LEFT SIDE: INPUT
# ==========================================================

with input_column:

    st.markdown(
        '<div class="section-heading">✦ Prompt Configuration</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter your own prompts and test input. "
        "No predefined prompts are loaded."
    )

    prompt_a = st.text_area(
        "Prompt A",
        placeholder="Enter your first prompt here...",
        height=150,
        key="prompt_a"
    )

    prompt_b = st.text_area(
        "Prompt B",
        placeholder="Enter your second prompt here...",
        height=150,
        key="prompt_b"
    )

    st.markdown("**Shared Test Input**")

    user_input = st.text_area(
        "Test Input",
        placeholder="Enter the question, topic, or text to test both prompts against...",
        height=130,
        label_visibility="collapsed",
        key="user_input"
    )

    with st.expander(
        "Evaluation Settings",
        expanded=True
    ):

        reference_keywords = st.text_input(
            "Reference Keywords",
            placeholder="Example: data, training, prediction",
            help="Optional. Enter expected concepts separated by commas."
        )

        expected_format = st.selectbox(
            "Expected Output Format",
            [
                "Any format",
                "Bullet points",
                "Numbered list",
                "JSON",
                "Table",
                "Paragraph"
            ]
        )

        number_of_outputs = st.selectbox(
            "Outputs per prompt",
            [1, 2, 3, 5],
            index=1
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.7,
            0.05
        )

        max_tokens = st.slider(
            "Maximum response tokens",
            100,
            1000,
            400,
            50
        )

    st.markdown("")

    run_button = st.button(
        "▶ Generate & Evaluate",
        type="primary",
        use_container_width=True
    )

    clear_button = st.button(
        "Clear Results",
        use_container_width=True
    )

    if clear_button:

        st.session_state.results = []
        st.session_state.last_run_time = None
        st.session_state.run_errors = []

        st.rerun()


# ==========================================================
# 13. GENERATION PROCESS
# ==========================================================

if run_button:

    if not prompt_a.strip():

        st.error("Please enter Prompt A.")

    elif not prompt_b.strip():

        st.error("Please enter Prompt B.")

    elif not user_input.strip():

        st.error("Please enter the shared test input.")

    elif not model_available:

        st.error(
            "Ollama is not connected. Start Ollama and try again."
        )

    else:

        generated_results = []
        errors = []

        total = number_of_outputs * 2
        completed = 0

        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        progress_placeholder.progress(0)

        start_time = time.time()

        prompt_list = [
            ("Prompt A", prompt_a),
            ("Prompt B", prompt_b)
        ]

        for tag, selected_prompt in prompt_list:

            for index in range(1, number_of_outputs + 1):

                completed += 1

                status_placeholder.info(
                    f"Generating {tag}: "
                    f"Output {index}/{number_of_outputs}"
                )

                try:

                    answer, duration = generate_response(
                        selected_prompt,
                        user_input,
                        temperature,
                        max_tokens
                    )

                    scores = evaluate_output(
                        answer,
                        user_input,
                        reference_keywords,
                        expected_format
                    )

                    generated_results.append({
                        "prompt_tag": tag,
                        "generation_index": index,
                        "prompt": selected_prompt,
                        "response_text": answer,
                        "generation_time": duration,
                        "scores": scores
                    })

                except Exception as error:

                    errors.append(
                        f"{tag}, output {index}: {str(error)}"
                    )

                progress_placeholder.progress(
                    completed / total
                )

        st.session_state.results = generated_results

        st.session_state.run_errors = errors

        st.session_state.last_run_time = round(
            time.time() - start_time,
            2
        )

        if generated_results:

            status_placeholder.success(
                f"Completed! {len(generated_results)} "
                "responses generated."
            )

        else:

            status_placeholder.error(
                "No responses were generated. Check Ollama."
            )


# ==========================================================
# 14. RIGHT SIDE: OUTPUT AREA
# ==========================================================

with output_column:

    st.markdown(
        '<div class="section-heading">◈ Generated Results</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Your generated responses and evaluation scores will appear here."
    )

    if not st.session_state.results:

        st.markdown("""
        <div style="
            background:#ffffff;
            border:1px solid #e1e7f0;
            border-radius:12px;
            min-height:380px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
            padding:25px;
        ">
            <div style="font-size:48px;">✦</div>
            <div style="
                font-size:20px;
                font-weight:650;
                color:#263247;
                margin-top:12px;
            ">
                Waiting for your prompts
            </div>
            <div style="
                font-size:13px;
                color:#8995a8;
                margin-top:8px;
            ">
                Enter Prompt A and Prompt B on the left,
                then click Generate & Evaluate.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        results = st.session_state.results

        metric_a, metric_b, comparison = make_comparison(
            results
        )

        st.markdown("**Average Scores**")

        score_a, score_b = st.columns(2)

        score_a.metric(
            "Prompt A",
            f"{metric_a['overall']:.2f}/5"
            if metric_a["overall"] is not None else "N/A"
        )

        score_b.metric(
            "Prompt B",
            f"{metric_b['overall']:.2f}/5"
            if metric_b["overall"] is not None else "N/A"
        )

        st.divider()

        st.markdown("**Generated Responses**")

        for item in results:

            tag = item["prompt_tag"]
            number = item["generation_index"]
            scores = item["scores"]

            with st.expander(
                f"{tag} · Output {number} · "
                f"Score {scores['overall']}/5",
                expanded=(number == 1)
            ):

                st.write(item["response_text"])

                st.divider()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Accuracy",
                    f"{scores['accuracy']:.2f}/5"
                    if scores["accuracy"] is not None else "N/A"
                )

                c2.metric(
                    "Relevance",
                    f"{scores['relevance']:.2f}/5"
                    if scores["relevance"] is not None else "N/A"
                )

                c3.metric(
                    "Format",
                    f"{scores['format_compliance']:.2f}/5"
                    if scores["format_compliance"] is not None else "N/A"
                )

                st.caption(
                    f"Generation time: {item['generation_time']} seconds"
                )

                if scores["format_compliance"] is not None:

                    if scores["format_status"] == "PASS":
                        st.success("Format requirement passed.")

                    else:
                        st.warning("Format requirement not satisfied.")

                if scores["missing_reference"]:

                    st.caption(
                        "Missing reference keywords: "
                        + ", ".join(scores["missing_reference"])
                    )

        if st.session_state.run_errors:

            with st.expander("Generation Errors"):

                for error in st.session_state.run_errors:
                    st.error(error)


# ==========================================================
# 15. COMPARISON AND REPORTS
# ==========================================================

if st.session_state.results:

    st.divider()

    st.header("📊 Evaluation & Comparison")

    results = st.session_state.results

    metric_a, metric_b, comparison = make_comparison(
        results
    )

    tab1, tab2, tab3 = st.tabs([
        "Performance Chart",
        "Comparison Table",
        "Download Report"
    ])

    # ------------------------------------------------------
    # PERFORMANCE CHART
    # ------------------------------------------------------

    with tab1:

        chart_rows = []

        for tag, metrics in [
            ("Prompt A", metric_a),
            ("Prompt B", metric_b)
        ]:

            for metric, value in metrics.items():

                if value is not None:

                    chart_rows.append({
                        "Prompt": tag,
                        "Metric": metric.replace("_", " ").title(),
                        "Score": value
                    })

        if chart_rows:

            chart_data = pd.DataFrame(chart_rows)

            chart = alt.Chart(
                chart_data
            ).mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5
            ).encode(
                x=alt.X(
                    "Metric:N",
                    title="Evaluation Metric"
                ),
                xOffset="Prompt:N",
                y=alt.Y(
                    "Score:Q",
                    title="Score out of 5",
                    scale=alt.Scale(domain=[0, 5])
                ),
                color=alt.Color(
                    "Prompt:N",
                    title="Prompt Version"
                ),
                tooltip=[
                    "Prompt",
                    "Metric",
                    "Score"
                ]
            ).properties(
                height=350
            )

            st.altair_chart(
                chart,
                use_container_width=True
            )

        st.caption(
            "Scores are calculated using keyword overlap "
            "and selected output-format checks."
        )

    # ------------------------------------------------------
    # COMPARISON TABLE
    # ------------------------------------------------------

    with tab2:

        st.subheader("Prompt A vs Prompt B")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        a = metric_a["overall"]
        b = metric_b["overall"]

        if a is not None and b is not None:

            if a > b:

                st.success(
                    f"Prompt A achieved a higher overall score ({a}/5)."
                )

            elif b > a:

                st.success(
                    f"Prompt B achieved a higher overall score ({b}/5)."
                )

            else:

                st.info(
                    "Both prompts achieved the same overall score."
                )

    # ------------------------------------------------------
    # CSV DOWNLOAD
    # ------------------------------------------------------

    with tab3:

        rows = []

        for item in results:

            score = item["scores"]

            rows.append({
                "Prompt": item["prompt_tag"],
                "Output Number": item["generation_index"],
                "Accuracy": score["accuracy"],
                "Relevance": score["relevance"],
                "Format Compliance": score["format_compliance"],
                "Overall Score": score["overall"],
                "Generation Time": item["generation_time"],
                "Generated Response": item["response_text"]
            })

        dataframe = pd.DataFrame(rows)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

        csv_data = dataframe.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download CSV Report",
            data=csv_data,
            file_name="experiment_16a_evaluation.csv",
            mime="text/csv",
            type="primary"
        )


# ==========================================================
# 16. FOOTER
# ==========================================================

st.divider()

st.caption(
    "NexusPrompt IQ · Experiment 16(A) · "
    "Dual-Prompt Heuristic Evaluation · Powered by Ollama"
)
