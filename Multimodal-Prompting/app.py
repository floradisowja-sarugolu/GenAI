import streamlit as st
import requests
import sqlite3
import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from datetime import datetime
from PIL import Image

st.set_page_config(
    page_title="NEXUS VISION",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")
DATABASE_PATH = os.path.join(EXPERIMENT_DIR, "experiments.db")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

AI_HORDE_URL = "https://aihorde.net/api/v2"
AI_HORDE_KEY = "0000000000"

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"

GENERATION_TIMEOUT = 600

PRESETS = {
    "Desert Observatory": (
        "A futuristic astronomical observatory in a vast desert, "
        "under a deep star-filled night sky, cinematic lighting, "
        "ultra detailed architecture, realistic photography"
    ),
    "Cyberpunk City": (
        "A futuristic cyberpunk city at night, neon signs, "
        "rain-soaked streets, glowing skyscrapers, cinematic "
        "atmosphere, highly detailed digital art"
    ),
    "Mountain Landscape": (
        "A breathtaking mountain landscape with snow-covered peaks, "
        "a crystal-clear lake, pine forests, morning mist, "
        "professional landscape photography"
    ),
    "Fantasy Castle": (
        "A magnificent fantasy castle floating above the clouds, "
        "golden sunlight, waterfalls, magical atmosphere, "
        "highly detailed fantasy concept art"
    ),
    "Wildlife Photography": (
        "A majestic Bengal tiger walking through a dense green forest, "
        "natural sunlight, realistic fur details, wildlife photography"
    )
}


# ============================================================
# 2. CUSTOM GUI DESIGN
# ============================================================


def inject_css():

    st.markdown("""
    <style>

    /* Main application */

    .stApp {
        background-color: #f5f7fb;
        color: #263247;
        font-family: Arial, sans-serif;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */

    #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Headings */

    h1, h2, h3 {
        color: #17243a !important;
        font-weight: 650 !important;
    }

    h1 {
        font-size: 2.3rem !important;
    }

    h2 {
        font-size: 1.6rem !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }

    p, label, .stMarkdown {
        color: #45536a;
    }

    /* Subtitle */

    .subtitle {
        color: #718096;
        font-size: 14px;
        margin-top: -12px;
        margin-bottom: 28px;
    }

    /* Tabs */

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #e1e6ef;
    }

    .stTabs [data-baseweb="tab"] {
        color: #58677e;
        font-size: 14px;
        font-weight: 600;
        height: 48px;
    }

    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Text inputs */

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background-color: #ffffff !important;
        color: #263247 !important;
        border: 1px solid #d7deea !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="input"] input:focus,
    div[data-baseweb="textarea"] textarea:focus {
        border-color: #4b83ee !important;
        box-shadow: 0 0 0 1px #4b83ee !important;
    }

    /* Dropdowns */

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d7deea !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] span {
        color: #263247 !important;
    }

    /* Buttons */

    div.stButton > button {
        background-color: #ffffff;
        color: #34445d;
        border: 1px solid #d7deea;
        border-radius: 8px;
        min-height: 40px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        border-color: #2563eb;
        color: #2563eb;
        background-color: #f0f5ff;
    }

    div.stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: #ffffff;
        border: 1px solid #2563eb;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }

    /* Preview panel */

    .preview-panel {
        background: #ffffff;
        border: 1px solid #e1e6ef;
        border-radius: 12px;
        min-height: 330px;
        padding: 25px;
        text-align: center;
    }

    .preview-icon {
        font-size: 48px;
        margin-top: 55px;
    }

    .preview-title {
        font-size: 21px;
        font-weight: 650;
        color: #263247;
        margin-top: 15px;
    }

    .preview-description {
        font-size: 13px;
        color: #8290a5;
        margin-top: 8px;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e6ef;
        border-radius: 10px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #718096 !important;
    }

    [data-testid="stMetricValue"] {
        color: #2563eb !important;
    }

    /* Expanders */

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e1e6ef;
        border-radius: 9px;
    }

    /* Images */

    [data-testid="stImage"] img {
        border-radius: 10px;
    }

    /* Dividers */

    hr {
        border-color: #e1e6ef !important;
    }

    /* Alerts */

    [data-testid="stAlert"] {
        border-radius: 9px;
    }

    </style>
    """, unsafe_allow_html=True)



# ============================================================
# 3. DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                prompt TEXT,
                negative_prompt TEXT,
                image BLOB,
                model_name TEXT,
                provider TEXT,
                generation_time REAL,
                dimensions TEXT,
                params TEXT,
                analysis TEXT
            )
        """)


def save_experiment(
    prompt,
    negative_prompt,
    image,
    model_name,
    provider,
    generation_time,
    dimensions,
    params
):

    experiment_id = str(uuid.uuid4())

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO experiments (
                id, created_at, prompt, negative_prompt,
                image, model_name, provider, generation_time,
                dimensions, params, analysis
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            created_at,
            prompt,
            negative_prompt,
            sqlite3.Binary(image),
            model_name,
            provider,
            generation_time,
            dimensions,
            json.dumps(params),
            None
        ))

    return experiment_id


def list_experiments():

    with get_connection() as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT id, created_at, prompt, negative_prompt,
                   model_name, provider, generation_time,
                   dimensions, params, analysis
            FROM experiments
            ORDER BY created_at DESC
        """).fetchall()

    results = []

    for row in rows:

        item = dict(row)

        item["params"] = json.loads(
            item["params"] or "{}"
        )

        item["analysis"] = (
            json.loads(item["analysis"])
            if item["analysis"] else None
        )

        results.append(item)

    return results


def get_experiment(experiment_id):

    with get_connection() as conn:

        conn.row_factory = sqlite3.Row

        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,)
        ).fetchone()

    if not row:
        return None

    item = dict(row)

    item["params"] = json.loads(
        item["params"] or "{}"
    )

    item["analysis"] = (
        json.loads(item["analysis"])
        if item["analysis"] else None
    )

    return item


def load_image_bytes(experiment_id):

    with get_connection() as conn:

        row = conn.execute(
            "SELECT image FROM experiments WHERE id = ?",
            (experiment_id,)
        ).fetchone()

    return bytes(row[0]) if row and row[0] else None


def update_experiment_analysis(experiment_id, analysis):

    with get_connection() as conn:

        conn.execute("""
            UPDATE experiments
            SET analysis = ?
            WHERE id = ?
        """, (
            json.dumps(analysis),
            experiment_id
        ))


def delete_experiment(experiment_id):

    with get_connection() as conn:

        conn.execute(
            "DELETE FROM experiments WHERE id = ?",
            (experiment_id,)
        )


def delete_all_experiments():

    with get_connection() as conn:

        conn.execute("DELETE FROM experiments")


initialize_database()


# ============================================================
# 4. IMAGE GENERATION - AI HORDE
# ============================================================

def execute_image_generation(
    prompt,
    aspect,
    seed,
    negative_prompt
):

    start_time = time.time()

    headers = {
        "apikey": AI_HORDE_KEY,
        "Client-Agent": "NexusVision:1.0:Streamlit"
    }

    dimensions = {
        "Square (1:1)": (512, 512),
        "Landscape (4:3)": (512, 384),
        "Portrait (3:4)": (384, 512),
        "Wide (16:9)": (512, 288)
    }

    width, height = dimensions.get(
        aspect, (512, 512)
    )

    full_prompt = prompt.strip()

    if negative_prompt.strip():
        full_prompt += " ### " + negative_prompt.strip()

    params = {
        "width": width,
        "height": height,
        "steps": 25,
        "cfg_scale": 7,
        "seed": str(seed) if seed else "-1",
        "sampler_name": "k_euler",
        "n": 1
    }

    payload = {
        "prompt": full_prompt,
        "params": params,
        "nsfw": False,
        "censor_nsfw": True,
        "trusted_workers": True,
        "slow_workers": True,
        "shared": False
    }

    try:

        response = requests.post(
            f"{AI_HORDE_URL}/generate/async",
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        request_id = response.json().get("id")

        if not request_id:
            raise Exception("No generation request ID returned.")

        deadline = time.time() + GENERATION_TIMEOUT

        while time.time() < deadline:

            status_response = requests.get(
                f"{AI_HORDE_URL}/generate/check/{request_id}",
                headers=headers,
                timeout=30
            )

            status_response.raise_for_status()

            status = status_response.json()

            if status.get("faulted"):
                raise Exception("AI Horde generation failed.")

            if status.get("done"):
                break

            time.sleep(5)

        else:

            raise TimeoutError(
                "Generation timed out. The free community queue "
                "may be busy. Please try again later."
            )

        result_response = requests.get(
            f"{AI_HORDE_URL}/generate/status/{request_id}",
            headers=headers,
            timeout=30
        )

        result_response.raise_for_status()

        generations = result_response.json().get(
            "generations", []
        )

        if not generations:
            raise Exception("No generated image was returned.")

        image_url = generations[0].get("img")

        if not image_url:
            raise Exception("Image URL is missing.")

        image_response = requests.get(
            image_url,
            timeout=60
        )

        image_response.raise_for_status()

        image_bytes = image_response.content

        image = Image.open(BytesIO(image_bytes))
        image.load()

        return {
            "success": True,
            "image": image_bytes,
            "model_name": generations[0].get(
                "model", "AI Horde Model"
            ),
            "provider": "AI Horde",
            "generation_time": round(
                time.time() - start_time, 2
            ),
            "dimensions": f"{image.width} x {image.height}",
            "params": params
        }

    except requests.exceptions.Timeout:

        return {
            "success": False,
            "error": "Image-generation service timed out. Try again."
        }

    except requests.exceptions.RequestException as error:

        return {
            "success": False,
            "error": f"AI Horde network error: {error}"
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# 5. IMAGE ANALYSIS - OLLAMA LLAVA
# ============================================================

def execute_vision_analysis(image_bytes):

    try:

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        prompt = """
        Analyze this image carefully.

        Return a JSON object containing:

        {
            "caption": "Detailed image description",
            "detected_concepts": {
                "objects": [],
                "colors": [],
                "environment": [],
                "style": []
            },
            "composition_summary": "Describe lighting,
            perspective, arrangement and composition."
        }

        Return valid JSON only.
        Describe only what is visible in the image.
        """

        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False,
            "format": "json"
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        answer = response.json().get("response", "")

        analysis = json.loads(answer)

        return {
            "success": True,
            "model_name": VISION_MODEL,
            "caption": analysis.get("caption", ""),
            "detected_concepts": analysis.get(
                "detected_concepts", {}
            ),
            "composition_summary": analysis.get(
                "composition_summary", ""
            )
        }

    except requests.exceptions.ConnectionError:

        return {
            "success": False,
            "error": "Cannot connect to Ollama. Check that Ollama is running."
        }

    except requests.exceptions.Timeout:

        return {
            "success": False,
            "error": "Vision analysis timed out. Please try again."
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "error": "LLaVA returned invalid JSON. Please retry."
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# 6. PROMPT-CAPTION EVALUATION
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on",
    "at", "of", "to", "with", "for", "is", "are",
    "was", "were", "by", "from", "under", "over",
    "into", "this", "that", "it", "its"
}


def tokenize(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 2
    }


def evaluate_prompt_caption_alignment(prompt, caption):

    prompt_words = tokenize(prompt)
    caption_words = tokenize(caption)

    if not prompt_words:

        return {
            "score": 0,
            "matched_words": [],
            "missing_words": [],
            "interpretation": "No prompt available."
        }

    matched = prompt_words.intersection(caption_words)
    missing = prompt_words.difference(caption_words)

    score = round(
        len(matched) / len(prompt_words) * 100,
        2
    )

    if score >= 75:
        interpretation = "High alignment"

    elif score >= 45:
        interpretation = "Moderate alignment"

    else:
        interpretation = "Low alignment"

    return {
        "score": score,
        "matched_words": sorted(matched),
        "missing_words": sorted(missing),
        "interpretation": interpretation,
        "total_prompt_keywords": len(prompt_words),
        "matched_keyword_count": len(matched)
    }


# ============================================================
# 7. PROMPT EDITOR
# ============================================================

def render_prompt_editor():

    st.markdown("### ✦ Prompt Studio")

    preset = st.selectbox(
        "Choose a prompt preset",
        ["Custom"] + list(PRESETS.keys()),
        key="prompt_preset"
    )

    if preset != "Custom":

        if st.session_state.get("last_preset") != preset:

            st.session_state.prompt_input = PRESETS[preset]
            st.session_state.last_preset = preset

    prompt = st.text_area(
        "Describe your image",
        key="prompt_input",
        height=145,
        placeholder="Describe the image you want to generate..."
    )

    negative = st.text_input(
        "Negative prompt",
        placeholder="blurry, low quality, distorted..."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.selectbox(
            "Image generation provider",
            ["AI Horde - Free Community Generation"],
            disabled=True
        )

    with c2:

        aspect = st.selectbox(
            "Aspect ratio",
            [
                "Square (1:1)",
                "Landscape (4:3)",
                "Portrait (3:4)",
                "Wide (16:9)"
            ]
        )

    seed = st.number_input(
        "Seed (0 = random)",
        min_value=0,
        max_value=2147483647,
        value=0
    )

    st.caption(
        "Free anonymous image generation. "
        "Generation time depends on community worker availability."
    )

    generate = st.button(
        "✦ Generate Image",
        type="primary",
        use_container_width=True
    )

    return prompt, aspect, seed, negative, generate


# ============================================================
# 8. IMAGE PREVIEW
# ============================================================

def render_image_preview(image, metadata):

    st.markdown("### ◈ Image Preview")

    if image:

        st.image(
            image,
            use_container_width=True
        )

        if metadata:

            c1, c2 = st.columns(2)

            c1.metric(
                "Image dimensions",
                metadata.get("dimensions", "Unknown")
            )

            c2.metric(
                "Generation time",
                f"{metadata.get('generation_time', 0)} sec"
            )

        analyze = st.button(
            "◎ Analyze with Ollama",
            type="primary",
            use_container_width=True
        )

    else:

        st.markdown("""
        <div class="preview-panel">
            <div class="preview-icon">🔭</div>
            <div class="preview-title">Image Preview</div>
            <div class="preview-description">
                Your generated image will appear here.
                <br><br>
                Start by entering a prompt in the Studio.
            </div>
        </div>
        """, unsafe_allow_html=True)

        analyze = False

    return analyze


# ============================================================
# 9. ANALYSIS DASHBOARD
# ============================================================

def render_results_dashboard(
    prompt,
    vision_model,
    caption,
    composition,
    evaluation,
    metadata,
    detected_concepts
):

    st.divider()

    st.markdown("## ◈ Analysis Results")

    score = evaluation.get("score", 0)

    c1, c2, c3 = st.columns(3)

    c1.metric("Prompt Alignment", f"{score}%")

    c2.metric("Vision Model", vision_model)

    c3.metric(
        "Image Provider",
        metadata.get("provider", "Unknown")
    )

    st.progress(
        min(max(score / 100, 0.0), 1.0)
    )

    st.caption(
        "The alignment score is based on keyword overlap "
        "between the prompt and generated caption."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### Original Prompt")
        st.write(prompt)

        st.markdown("### Vision Caption")
        st.write(caption)

    with right:

        st.markdown("### Composition Analysis")
        st.write(composition)

        st.markdown("### Evaluation")

        st.success(
            evaluation.get("interpretation", "Not evaluated")
        )

        st.markdown("**Matched keywords**")

        st.write(
            ", ".join(evaluation.get("matched_words", []))
            or "None"
        )

        st.markdown("**Missing keywords**")

        st.write(
            ", ".join(evaluation.get("missing_words", []))
            or "None"
        )

    with st.expander("View detected visual concepts"):

        st.json(detected_concepts)


# ============================================================
# 10. SESSION STATE
# ============================================================

defaults = {
    "prompt_input": PRESETS["Desert Observatory"],
    "current_image": None,
    "current_image_metadata": None,
    "current_analysis": None,
    "active_exp_id": None,
    "last_preset": "Desert Observatory",
    "flash": None
}

for key, value in defaults.items():

    st.session_state.setdefault(key, value)


# ============================================================
# 11. MAIN APPLICATION HEADER
# ============================================================

inject_css()

st.title("🔭 NEXUS VISION")

st.markdown(
    '<div class="subtitle">'
    'AI Image Generation  ·  Visual Understanding  ·  '
    'Prompt Alignment Evaluation'
    '</div>',
    unsafe_allow_html=True
)

experiments = list_experiments()

tabs = st.tabs([
    "✦ Studio",
    f"◷ History ({len(experiments)})",
    "⚖ Compare",
    "⚙ Settings"
])


# ============================================================
# 12. STUDIO TAB
# ============================================================

with tabs[0]:

    if st.session_state.flash:

        st.success(st.session_state.flash)
        st.session_state.flash = None

    col1, col2 = st.columns([1.15, 1])

    with col1:

        prompt, aspect, seed, negative, generate = (
            render_prompt_editor()
        )

    with col2:

        analyze = render_image_preview(
            st.session_state.current_image,
            st.session_state.current_image_metadata
        )

    # Image generation

    if generate:

        if not prompt.strip():

            st.warning("Please enter an image prompt.")

        else:

            with st.spinner(
                "Generating your image using AI Horde. "
                "Please wait while a community worker processes it..."
            ):

                result = execute_image_generation(
                    prompt,
                    aspect,
                    seed,
                    negative
                )

            if result["success"]:

                st.session_state.current_image = result["image"]

                st.session_state.current_image_metadata = {
                    "model_name": result["model_name"],
                    "provider": result["provider"],
                    "generation_time": result["generation_time"],
                    "dimensions": result["dimensions"],
                    "prompt": prompt
                }

                st.session_state.current_analysis = None

                st.session_state.active_exp_id = save_experiment(
                    prompt,
                    negative,
                    result["image"],
                    result["model_name"],
                    result["provider"],
                    result["generation_time"],
                    result["dimensions"],
                    result["params"]
                )

                st.session_state.flash = (
                    "Image generated and saved successfully!"
                )

                st.rerun()

            else:

                st.error(result["error"])

    # Image analysis

    if analyze:

        if st.session_state.current_image:

            with st.spinner(
                "Ollama LLaVA is analyzing your image..."
            ):

                result = execute_vision_analysis(
                    st.session_state.current_image
                )

            if result["success"]:

                metadata = (
                    st.session_state.current_image_metadata or {}
                )

                original_prompt = metadata.get(
                    "prompt",
                    st.session_state.prompt_input
                )

                evaluation = evaluate_prompt_caption_alignment(
                    original_prompt,
                    result["caption"]
                )

                analysis = {
                    "vision_model_name": result["model_name"],
                    "vision_caption": result["caption"],
                    "vision_composition": result["composition_summary"],
                    "detected_concepts": result["detected_concepts"],
                    "evaluation": evaluation
                }

                st.session_state.current_analysis = analysis

                if st.session_state.active_exp_id:

                    update_experiment_analysis(
                        st.session_state.active_exp_id,
                        analysis
                    )

                st.rerun()

            else:

                st.error(result["error"])

        else:

            st.warning("Generate an image first.")

    # Show results

    if (
        st.session_state.current_analysis
        and st.session_state.current_image
    ):

        metadata = (
            st.session_state.current_image_metadata or {}
        )

        analysis = st.session_state.current_analysis

        render_results_dashboard(
            metadata.get("prompt", ""),
            analysis["vision_model_name"],
            analysis["vision_caption"],
            analysis["vision_composition"],
            analysis["evaluation"],
            metadata,
            analysis["detected_concepts"]
        )


# ============================================================
# 13. HISTORY TAB
# ============================================================

with tabs[1]:

    st.markdown("## ◷ Generation History")

    if experiments:

        for record in experiments:

            with st.container(border=True):

                st.write(f"**{record['created_at']}**")

                st.write(record["prompt"][:250])

                c1, c2, c3 = st.columns([1, 1, 3])

                with c1:

                    if st.button(
                        "Open",
                        key=f"open_{record['id']}"
                    ):

                        image = load_image_bytes(record["id"])

                        st.session_state.prompt_input = record["prompt"]

                        st.session_state.current_image = image

                        st.session_state.current_image_metadata = {
                            "model_name": record["model_name"],
                            "provider": record["provider"],
                            "generation_time": record["generation_time"],
                            "dimensions": record["dimensions"],
                            "prompt": record["prompt"]
                        }

                        st.session_state.current_analysis = (
                            record["analysis"]
                        )

                        st.session_state.active_exp_id = record["id"]

                        st.session_state.flash = (
                            "Experiment loaded successfully."
                        )

                        st.rerun()

                with c2:

                    if st.button(
                        "Delete",
                        key=f"delete_{record['id']}"
                    ):

                        delete_experiment(record["id"])

                        if (
                            st.session_state.active_exp_id
                            == record["id"]
                        ):

                            st.session_state.current_image = None
                            st.session_state.current_image_metadata = None
                            st.session_state.current_analysis = None
                            st.session_state.active_exp_id = None

                        st.rerun()

                with c3:

                    st.caption(
                        f"Provider: {record['provider']}  |  "
                        f"Model: {record['model_name']}"
                    )

                    st.caption(
                        "Analysis completed"
                        if record["analysis"]
                        else "Not analyzed"
                    )

    else:

        st.info(
            "No experiments saved yet. "
            "Generate your first image in Studio."
        )


# ============================================================
# 14. COMPARE TAB
# ============================================================

with tabs[2]:

    st.markdown("## ⚖ Compare Experiments")

    if len(experiments) >= 2:

        ids = [record["id"] for record in experiments]

        labels = {
            record["id"]: (
                f"{record['created_at']} | "
                f"{record['prompt'][:45]}"
            )
            for record in experiments
        }

        col_a, col_b = st.columns(2)

        with col_a:

            id_a = st.selectbox(
                "Select Experiment A",
                ids,
                format_func=lambda x: labels[x],
                key="compare_a"
            )

        with col_b:

            id_b = st.selectbox(
                "Select Experiment B",
                ids,
                index=1,
                format_func=lambda x: labels[x],
                key="compare_b"
            )

        if id_a == id_b:

            st.warning(
                "Please select two different experiments."
            )

        else:

            left, right = st.columns(2)

            for column, experiment_id in [
                (left, id_a),
                (right, id_b)
            ]:

                record = get_experiment(experiment_id)

                image = load_image_bytes(experiment_id)

                with column:

                    st.markdown(
                        f"### {record['created_at']}"
                    )

                    if image:

                        st.image(
                            image,
                            use_container_width=True
                        )

                    st.markdown("**Original Prompt**")

                    st.write(record["prompt"])

                    analysis = record["analysis"]

                    if analysis:

                        st.markdown("**Vision Caption**")

                        st.write(
                            analysis.get(
                                "vision_caption",
                                "Not available"
                            )
                        )

                        score = analysis.get(
                            "evaluation", {}
                        ).get("score", 0)

                        st.metric(
                            "Alignment Score",
                            f"{score}%"
                        )

                    else:

                        st.info("This image has not been analyzed.")

    else:

        st.info(
            "Generate at least two images to compare your experiments."
        )


# ============================================================
# 15. SETTINGS TAB
# ============================================================

with tabs[3]:

    st.markdown("## ⚙ Application Settings")

    st.markdown("### Image Generation")

    st.write("Provider: AI Horde")
    st.write("Access: Free anonymous community generation")
    st.write("Personal API key: Not required")

    st.divider()

    st.markdown("### Vision Analysis")

    st.write("Provider: Ollama")
    st.write(f"Vision model: {VISION_MODEL}")

    if st.button("Check Ollama Connection"):

        try:

            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=5
            )

            response.raise_for_status()

            models = response.json().get("models", [])

            model_names = [
                model["name"] for model in models
            ]

            installed = any(
                name == VISION_MODEL
                or name.startswith(VISION_MODEL + ":")
                for name in model_names
            )

            if installed:

                st.success(
                    "Ollama is connected and LLaVA is installed."
                )

            else:

                st.warning(
                    "Ollama is running, but LLaVA is not installed. "
                    "Run: ollama pull llava"
                )

        except Exception as error:

            st.error(
                f"Could not connect to Ollama: {error}"
            )

    st.divider()

    st.markdown("### Local Storage")

    st.write("Storage directory:")

    st.code(EXPERIMENT_DIR)

    st.write("Database: SQLite")

    st.code(DATABASE_PATH)

    st.divider()

    st.warning(
        "Deleting all experiments permanently removes "
        "all saved image records and analysis results."
    )

    confirm_delete = st.checkbox(
        "I understand that this action cannot be undone."
    )

    if st.button(
        "Delete All Experiments",
        disabled=not confirm_delete
    ):

        delete_all_experiments()

        st.session_state.current_image = None
        st.session_state.current_image_metadata = None
        st.session_state.current_analysis = None
        st.session_state.active_exp_id = None

        st.session_state.flash = (
            "All experiments have been deleted."
        )

        st.rerun()
