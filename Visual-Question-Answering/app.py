
import base64
import io
import requests
import streamlit as st
from PIL import Image


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VisionDesk AI",
    page_icon="🩷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

OLLAMA_URL = "http://localhost:11434/api/chat"


# =========================================================
# 2. PINK AND LIGHT BLUE THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff7fb, #f0f8ff);
    color: #344054;
}

.main .block-container {
    max-width: 1450px;
    padding: 1.5rem 2rem 2rem;
}

h1, h2, h3 {
    color: #34415e !important;
}

.header-card {
    background: linear-gradient(110deg, #ffe5f0, #dff2ff);
    border: 1px solid #f5d5e5;
    border-radius: 20px;
    padding: 25px 30px;
    margin-bottom: 20px;
}

.header-title {
    font-size: 32px;
    font-weight: 800;
    color: #34415e;
}

.header-subtitle {
    color: #73819a;
    font-size: 14px;
    margin-top: 6px;
}

.status-card {
    background: #e5f8ee;
    border: 1px solid #bcebd0;
    border-radius: 10px;
    padding: 11px 15px;
    color: #18794e;
    margin-bottom: 18px;
    font-size: 14px;
}

.panel {
    background: rgba(255,255,255,0.94);
    border: 1px solid #e4eaf2;
    border-radius: 17px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(105, 126, 160, 0.07);
}

.panel-title {
    font-size: 20px;
    font-weight: 750;
    color: #394968;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: #8793a8;
    font-size: 13px;
    margin-bottom: 18px;
}

div[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e7eaf2;
    border-radius: 13px;
    padding: 13px;
    margin-bottom: 10px;
}

div[data-testid="stChatInput"] {
    background: white;
    border: 1px solid #d9e5f5;
    border-radius: 12px;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 650;
    border: 1px solid #e5d6e7;
    background: #fff0f6;
    color: #6c4565;
}

div.stButton > button:hover {
    background: #ffe1ed;
    border-color: #e9b8d0;
    color: #59364f;
}

div.stButton > button[kind="primary"] {
    background: #b9ddf8;
    color: #24466b;
    border: 1px solid #a5d1f1;
}

div.stButton > button[kind="primary"]:hover {
    background: #a6d1f2;
    color: #203e60;
}

div[data-testid="stFileUploader"] {
    background: #f8fbff;
    border: 1px dashed #b9d9f2;
    border-radius: 12px;
    padding: 10px;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stTextArea"] textarea {
    border-radius: 9px;
}

.image-box {
    background: #f8fbff;
    border: 1px dashed #c5dff4;
    border-radius: 14px;
    padding: 50px 15px;
    text-align: center;
    color: #8798ae;
}

.info-card {
    background: #fff1f7;
    border: 1px solid #f6d8e6;
    border-radius: 12px;
    padding: 13px;
    color: #76536a;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #9aa5b6;
    font-size: 12px;
    padding: 15px;
}

#MainMenu, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SESSION STATE
# =========================================================

defaults = {
    "messages": [],
    "image_bytes": None,
    "image_name": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. CHECK OLLAMA CONNECTION
# =========================================================

def check_ollama():

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        response.raise_for_status()

        return response.json().get("models", [])

    except requests.RequestException:
        return None


# =========================================================
# 5. PREPARE IMAGE
# =========================================================

def prepare_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image.thumbnail((1024, 1024))

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# =========================================================
# 6. ASK THE VISION MODEL
# =========================================================

def ask_vision_model(
    question,
    image_bytes,
    model_name,
    temperature,
    system_prompt
):

    encoded_image = prepare_image(image_bytes)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question,
                "images": [encoded_image]
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"]


# =========================================================
# 7. HEADER
# =========================================================

st.markdown("""
<div class="header-card">
    <div class="header-title">🩷 VisionDesk AI</div>
    <div class="header-subtitle">
        Visual Question Answering Studio
        &nbsp; • &nbsp; Multimodal AI
        &nbsp; • &nbsp; LLaVA Vision Model
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. MODEL SETTINGS
# =========================================================

with st.expander("⚙️ Model Settings"):

    settings_col1, settings_col2 = st.columns(2)

    with settings_col1:

        model_name = st.selectbox(
            "Select Vision Model",
            ["llava"],
            index=0
        )

        temperature = st.slider(
            "Response Creativity",
            0.0,
            1.0,
            0.2,
            0.1
        )

    with settings_col2:

        system_prompt = st.text_area(
            "System Instructions",
            value=(
                "You are a visual question answering assistant. "
                "Carefully inspect the uploaded image and answer "
                "the user's question accurately. Only describe "
                "things that are actually visible. If something "
                "is unclear, clearly mention your uncertainty."
            ),
            height=120
        )


# =========================================================
# 9. OLLAMA STATUS
# =========================================================

models = check_ollama()

if models is None:

    st.error(
        "Ollama is not running. Start Ollama before using this application."
    )

    ollama_ready = False

else:

    available_models = [
        model.get("name", "")
        for model in models
    ]

    ollama_ready = any(
        name == "llava" or name.startswith("llava:")
        for name in available_models
    )

    if ollama_ready:

        st.markdown("""
        <div class="status-card">
            ● Connected to Ollama · LLaVA is ready
        </div>
        """, unsafe_allow_html=True)

    else:

        st.warning(
            "LLaVA is not installed. Run: ollama pull llava"
        )


# =========================================================
# 10. MAIN WORKSPACE
# =========================================================

chat_col, image_col = st.columns(
    [1.55, 1],
    gap="large"
)


# =========================================================
# 11. CHAT PANEL - LEFT SIDE
# =========================================================

with chat_col:

    st.markdown("""
    <div class="panel-title">💬 Conversation</div>
    <div class="panel-subtitle">
        Ask questions and explore the contents of your image.
    </div>
    """, unsafe_allow_html=True)

    chat_area = st.container(height=480)

    with chat_area:

        if not st.session_state.messages:

            st.info(
                "Your conversation will appear here. "
                "Upload an image and ask your first question."
            )

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"],
                avatar="👤" if message["role"] == "user" else "🩷"
            ):

                st.markdown(message["content"])

    question = st.chat_input(
        "Type your question about the image..."
    )

    if question:

        if st.session_state.image_bytes is None:

            st.warning(
                "Please upload an image before asking a question."
            )

        elif not ollama_ready:

            st.error(
                "LLaVA is not ready. Check your Ollama installation."
            )

        else:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.spinner(
                "🩵 Analyzing your image..."
            ):

                try:

                    answer = ask_vision_model(
                        question,
                        st.session_state.image_bytes,
                        model_name,
                        temperature,
                        system_prompt
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.rerun()

                except requests.Timeout:

                    st.error(
                        "The model took too long to respond. "
                        "Try again with a smaller image."
                    )

                except requests.RequestException as error:

                    st.error(
                        f"Ollama connection error: {error}"
                    )

                except Exception as error:

                    st.error(
                        f"An error occurred: {error}"
                    )


# =========================================================
# 12. IMAGE PANEL - RIGHT SIDE
# =========================================================

with image_col:

    st.markdown("""
    <div class="panel-title">🖼️ Image Workspace</div>
    <div class="panel-subtitle">
        Upload an image for visual analysis.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a photograph or any supported image."
    )

    if uploaded_file is not None:

        new_image_bytes = uploaded_file.getvalue()

        if new_image_bytes != st.session_state.image_bytes:

            st.session_state.image_bytes = new_image_bytes

            st.session_state.image_name = uploaded_file.name

            st.session_state.messages = []

        image = Image.open(
            io.BytesIO(st.session_state.image_bytes)
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"📄 {st.session_state.image_name}"
        )

        st.success("Image uploaded successfully")

        if st.button(
            "✕ Remove Image",
            use_container_width=True
        ):

            st.session_state.image_bytes = None
            st.session_state.image_name = None
            st.session_state.messages = []

            st.rerun()

    else:

        st.markdown("""
        <div class="image-box">
            <div style="font-size:45px;">🩵</div>
            <h4>Image Preview</h4>
            <p>Your uploaded image will appear here.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="info-card">
        <b>How it works</b><br><br>
        1. Upload an image.<br>
        2. Enter a question in the chat.<br>
        3. LLaVA examines the image.<br>
        4. View the generated answer.<br>
        5. Ask follow-up questions.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 13. CLEAR CONVERSATION
# =========================================================

st.write("")

clear_col, space_col = st.columns([1, 4])

with clear_col:

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    VisionDesk AI · Experiment 19 ·
    Visual Question Answering using Ollama and LLaVA
</div>
""", unsafe_allow_html=True)
