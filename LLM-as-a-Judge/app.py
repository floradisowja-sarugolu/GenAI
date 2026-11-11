
import streamlit as st
import requests
import pandas as pd
import json
import re
import time
from datetime import datetime

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NexusJudge AI",
    page_icon="⚖️",
    layout="wide"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


# =====================================================
# 2. LIGHT THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #f5f7fb;
    color: #263247;
}

.main .block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem;
}

h1, h2, h3 {
    color: #1e2d45 !important;
}

.subtitle {
    color: #7b879b;
    margin-top: -12px;
    margin-bottom: 22px;
    font-size: 14px;
}

.section-title {
    font-size: 19px;
    font-weight: 650;
    color: #263b59;
    margin-bottom: 12px;
}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: white !important;
    color: #263247 !important;
    border: 1px solid #d8e0eb !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background: white !important;
    border-radius: 8px !important;
}

div.stButton > button {
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
}

div.stButton > button[kind="primary"] {
    background: #2563eb;
    color: white;
    border: none;
}

.result-card {
    background: white;
    border: 1px solid #e1e7f0;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
}

.score-box {
    background: white;
    border: 1px solid #e1e7f0;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}

.small-text {
    font-size: 12px;
    color: #8491a5;
}

hr {
    border-color: #e4eaf2;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. SESSION STATE
# =====================================================

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "history" not in st.session_state:
    st.session_state.history = []


# =====================================================
# 4. CHECK OLLAMA CONNECTION
# =====================================================

def check_ollama():

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        response.raise_for_status()

        models = response.json().get("models", [])

        return any(
            model["name"] == MODEL
            or model["name"].startswith(MODEL + ":")
            for model in models
        )

    except Exception:
        return False


# =====================================================
# 5. BUILD JUDGE PROMPT
# =====================================================

def build_judge_prompt(
    task,
    response_a,
    response_b,
    criteria
):

    criteria_text = ", ".join(criteria)

    return f"""
You are an impartial and professional AI evaluator.

Evaluate two AI-generated responses for the same task.

ORIGINAL TASK:
{task}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

EVALUATION CRITERIA:
{criteria_text}

SCORING RULE:
Give each criterion a score from 1 to 5.

1 = Very poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

Evaluation instructions:

1. Clarity: Evaluate how understandable and well-organized
   the response is.

2. Correctness: Evaluate factual accuracy and whether
   the response correctly addresses the task.
   Do not reward unsupported claims.

3. Relevance: Evaluate how closely the response answers
   the original task.

4. Give a short reason for every score.

5. Identify the strengths and weaknesses of each response.

6. Calculate the average score for each response using
   the selected criteria.

7. Select the better response based on the scores.
   If scores are equal, return "Tie".

Return ONLY valid JSON in this exact structure:

{{
  "response_a": {{
    "scores": {{
      "clarity": 1,
      "correctness": 1,
      "relevance": 1
    }},
    "average_score": 1.0,
    "strengths": ["strength 1"],
    "weaknesses": ["weakness 1"],
    "explanation": "Overall evaluation"
  }},
  "response_b": {{
    "scores": {{
      "clarity": 1,
      "correctness": 1,
      "relevance": 1
    }},
    "average_score": 1.0,
    "strengths": ["strength 1"],
    "weaknesses": ["weakness 1"],
    "explanation": "Overall evaluation"
  }},
  "winner": "Response A",
  "final_reason": "Explain why this response performed better"
}}

Only include the selected criteria in the scores object.
Use numeric scores between 1 and 5.
"""


# =====================================================
# 6. CALL OLLAMA JUDGE
# =====================================================

def evaluate_with_llm(
    task,
    response_a,
    response_b,
    criteria,
    temperature
):

    judge_prompt = build_judge_prompt(
        task,
        response_a,
        response_b,
        criteria
    )

    payload = {
        "model": MODEL,
        "prompt": judge_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature
        }
    }

    start_time = time.time()

    result = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    result.raise_for_status()

    raw_text = result.json().get("response", "")

    duration = round(
        time.time() - start_time,
        2
    )

    try:
        evaluation = json.loads(raw_text)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            raw_text,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "The judge did not return valid JSON. Please try again."
            )

        evaluation = json.loads(match.group())

    return evaluation, duration


# =====================================================
# 7. VALIDATE EVALUATION RESULT
# =====================================================

def validate_result(data, criteria):

    for response_key in ["response_a", "response_b"]:

        if response_key not in data:
            raise ValueError("Judge output is missing a response.")

        result = data[response_key]

        scores = result.get("scores", {})

        for criterion in criteria:

            key = criterion.lower()

            if key not in scores:
                raise ValueError(
                    f"Judge output is missing the {key} score."
                )

            score = float(scores[key])

            if not 1 <= score <= 5:
                raise ValueError(
                    f"Invalid score for {key}."
                )

        average = float(result.get("average_score", 0))

        if not 1 <= average <= 5:
            raise ValueError("Invalid average score.")

    if data.get("winner") not in [
        "Response A", "Response B", "Tie"
    ]:
        raise ValueError("Invalid winner returned by the judge.")


# =====================================================
# 8. APPLICATION HEADER
# =====================================================

left, right = st.columns([3, 1])

with left:

    st.title("⚖️ NexusJudge AI")

    st.markdown(
        '<div class="subtitle">'
        'LLM-as-Judge · Automated Response Evaluation Studio'
        '</div>',
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        "<div style='text-align:right;padding-top:18px;"
        "color:#8491a5;font-size:13px;'>"
        "EXPERIMENT 16(B)</div>",
        unsafe_allow_html=True
    )


if check_ollama():

    st.success(f"Connected to Ollama · {MODEL}")

else:

    st.error(
        "Ollama is not connected or the required model is missing."
    )

st.divider()


# =====================================================
# 9. INPUT AND OUTPUT COLUMNS
# =====================================================

input_col, output_col = st.columns(
    [1, 1.15],
    gap="large"
)


# =====================================================
# 10. LEFT PANEL: USER INPUT
# =====================================================

with input_col:

    st.markdown(
        '<div class="section-title">✦ Evaluation Input</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Provide the original task and the two responses "
        "you want the AI judge to evaluate."
    )

    task = st.text_area(
        "Original Task / Question",
        placeholder="Enter the original question or task...",
        height=110
    )

    response_a = st.text_area(
        "Response A",
        placeholder="Paste the first AI-generated response here...",
        height=170
    )

    response_b = st.text_area(
        "Response B",
        placeholder="Paste the second AI-generated response here...",
        height=170
    )

    with st.expander("Judge Settings", expanded=True):

        st.markdown("**Evaluation Criteria**")

        clarity = st.checkbox(
            "Clarity",
            value=True
        )

        correctness = st.checkbox(
            "Correctness",
            value=True
        )

        relevance = st.checkbox(
            "Relevance",
            value=True
        )

        temperature = st.slider(
            "Judge Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        )

    criteria = []

    if clarity:
        criteria.append("Clarity")

    if correctness:
        criteria.append("Correctness")

    if relevance:
        criteria.append("Relevance")

    evaluate_button = st.button(
        "⚖️ Evaluate Responses",
        type="primary",
        use_container_width=True
    )

    clear_button = st.button(
        "Clear Evaluation",
        use_container_width=True
    )

    if clear_button:

        st.session_state.evaluation = None
        st.rerun()


# =====================================================
# 11. RUN THE EVALUATION
# =====================================================

if evaluate_button:

    if not task.strip():

        st.error("Enter the original task.")

    elif not response_a.strip():

        st.error("Enter Response A.")

    elif not response_b.strip():

        st.error("Enter Response B.")

    elif not criteria:

        st.error("Select at least one evaluation criterion.")

    elif not check_ollama():

        st.error("Please start Ollama and check your model.")

    else:

        try:

            with st.spinner(
                "AI Judge is evaluating both responses..."
            ):

                evaluation, duration = evaluate_with_llm(
                    task,
                    response_a,
                    response_b,
                    criteria,
                    temperature
                )

                validate_result(
                    evaluation,
                    criteria
                )

                evaluation["duration"] = duration

                evaluation["criteria"] = criteria

                evaluation["timestamp"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                evaluation["task"] = task

                evaluation["response_a_text"] = response_a

                evaluation["response_b_text"] = response_b

                st.session_state.evaluation = evaluation

                st.session_state.history.append(evaluation)

            st.success("Evaluation completed successfully!")

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to Ollama. Make sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The evaluation took too long. Try shorter responses."
            )

        except Exception as error:

            st.error(f"Evaluation failed: {error}")


# =====================================================
# 12. RIGHT PANEL: OUTPUT
# =====================================================

with output_col:

    st.markdown(
        '<div class="section-title">◈ Judge Results</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Scores, explanations and comparison will appear here."
    )

    data = st.session_state.evaluation

    if not data:

        st.markdown("""
        <div style="
            background:white;
            border:1px solid #e1e7f0;
            border-radius:12px;
            min-height:430px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
            padding:25px;
        ">
            <div style="font-size:48px;">⚖️</div>
            <div style="
                font-size:20px;
                font-weight:650;
                color:#263247;
                margin-top:14px;
            ">
                Awaiting Evaluation
            </div>
            <div style="
                font-size:13px;
                color:#8995a8;
                margin-top:8px;
            ">
                Add two responses on the left and run
                the AI judge to see the evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        a = data["response_a"]
        b = data["response_b"]

        st.markdown("**Overall Evaluation**")

        col_a, col_b = st.columns(2)

        col_a.metric(
            "Response A",
            f"{a['average_score']:.2f} / 5"
        )

        col_b.metric(
            "Response B",
            f"{b['average_score']:.2f} / 5"
        )

        winner = data["winner"]

        if winner == "Response A":

            st.success("🏆 Response A is the better response.")

        elif winner == "Response B":

            st.success("🏆 Response B is the better response.")

        else:

            st.info("Both responses received equal evaluation.")

        st.divider()

        tab_a, tab_b = st.tabs([
            "Response A",
            "Response B"
        ])

        for tab, result, name in [
            (tab_a, a, "Response A"),
            (tab_b, b, "Response B")
        ]:

            with tab:

                st.markdown(f"**{name} Scores**")

                for criterion in data["criteria"]:

                    key = criterion.lower()

                    score = float(result["scores"][key])

                    st.markdown(
                        f"**{criterion}: {score:.1f} / 5**"
                    )

                    st.progress(score / 5)

                st.markdown("**Strengths**")

                for item in result.get("strengths", []):
                    st.markdown(f"- {item}")

                st.markdown("**Weaknesses**")

                for item in result.get("weaknesses", []):
                    st.markdown(f"- {item}")

                st.markdown("**Judge Explanation**")

                st.write(result.get("explanation", ""))


# =====================================================
# 13. COMPARISON TABLE AND CHART
# =====================================================

if st.session_state.evaluation:

    data = st.session_state.evaluation

    st.divider()

    st.header("📊 Evaluation Comparison")

    a = data["response_a"]
    b = data["response_b"]

    rows = []

    for criterion in data["criteria"]:

        key = criterion.lower()

        score_a = float(a["scores"][key])

        score_b = float(b["scores"][key])

        if score_a > score_b:
            better = "Response A"

        elif score_b > score_a:
            better = "Response B"

        else:
            better = "Tie"

        rows.append({
            "Criterion": criterion,
            "Response A": score_a,
            "Response B": score_b,
            "Better Result": better
        })

    rows.append({
        "Criterion": "Overall",
        "Response A": float(a["average_score"]),
        "Response B": float(b["average_score"]),
        "Better Result": data["winner"]
    })

    comparison = pd.DataFrame(rows)

    chart_tab, table_tab, report_tab = st.tabs([
        "Score Chart",
        "Comparison Table",
        "Evaluation Report"
    ])

    with chart_tab:

        chart_data = []

        for row in rows:

            chart_data.append({
                "Criterion": row["Criterion"],
                "Response": "Response A",
                "Score": row["Response A"]
            })

            chart_data.append({
                "Criterion": row["Criterion"],
                "Response": "Response B",
                "Score": row["Response B"]
            })

        chart_df = pd.DataFrame(chart_data)

        st.bar_chart(
            chart_df,
            x="Criterion",
            y="Score",
            color="Response",
            y_range=(0, 5)
        )

    with table_tab:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    with report_tab:

        st.markdown("**Final Reason**")

        st.write(data.get("final_reason", ""))

        st.markdown("**Evaluation Details**")

        st.write(
            f"Evaluation time: {data['duration']} seconds"
        )

        st.write(
            f"Evaluated at: {data['timestamp']}"
        )

        report = {
            "task": data["task"],
            "response_a": a,
            "response_b": b,
            "winner": data["winner"],
            "final_reason": data["final_reason"],
            "criteria": data["criteria"],
            "evaluation_time": data["duration"],
            "evaluated_at": data["timestamp"]
        }

        st.download_button(
            "⬇ Download Evaluation Report",
            data=json.dumps(
                report,
                indent=4
            ),
            file_name="experiment_16b_report.json",
            mime="application/json",
            type="primary"
        )


# =====================================================
# 14. EVALUATION HISTORY
# =====================================================

with st.expander("Evaluation History"):

    if not st.session_state.history:

        st.info("No previous evaluations in this session.")

    else:

        for index, item in enumerate(
            reversed(st.session_state.history),
            start=1
        ):

            st.markdown(
                f"**Evaluation {index}** · "
                f"{item['timestamp']} · "
                f"Winner: {item['winner']}"
            )


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.caption(
    "NexusJudge AI · Experiment 16(B) · "
    "LLM-as-Judge Evaluation · Powered by Ollama"
)
