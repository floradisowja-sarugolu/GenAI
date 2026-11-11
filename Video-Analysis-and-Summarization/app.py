
import io
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=" Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_ID = "facebook/timesformer-base-finetuned-k400"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

NUM_FRAMES = 8
MAX_SEGMENTS = 6


# =========================================================
# 2. PINK AND LIGHT BLUE THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff8fb, #f1f8ff);
    color: #344054;
}

.main .block-container {
    max-width: 1450px;
    padding: 1.5rem 2rem 2rem;
}

h1, h2, h3 {
    color: #34415e !important;
}

.section-title {
    color: #394968;
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #8995a8;
    font-size: 13px;
    margin-bottom: 15px;
}

.panel {
    background: white;
    border: 1px solid #e6eaf2;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(100, 120, 150, 0.05);
}

div.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 650;
    background: #fff0f6;
    border: 1px solid #f0d5e4;
    color: #704c68;
}

div.stButton > button:hover {
    background: #ffe1ed;
    color: #5e3b56;
}

div.stButton > button[kind="primary"] {
    background: #b9ddf8;
    border: 1px solid #a5d1f1;
    color: #24466b;
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

div[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #e8eaf2;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stChatInput"] {
    background: white;
    border-radius: 12px;
}

.metric-card {
    background: #fff0f6;
    border: 1px solid #f4d8e6;
    border-radius: 12px;
    padding: 13px;
    text-align: center;
}

.footer {
    color: #9aa5b6;
    text-align: center;
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
    "video_analysis": None,
    "video_name": None,
    "video_bytes": None,
    "video_duration": None,
    "summary": None,
    "chat_history": [],
    "analysis_time": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. LOAD TIMESFORMER MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_timesformer():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_ID
    )

    model.to(device)
    model.eval()

    return processor, model, device


# =========================================================
# 5. EXTRACT VIDEO INFORMATION
# =========================================================

def get_video_info(video_bytes):

    video_array = np.frombuffer(
        video_bytes,
        dtype=np.uint8
    )

    cap = cv2.VideoCapture(
        io.BytesIO(video_array).read()
    ) if False else None

    import tempfile
    import os

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as temp_file:

            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Unable to open this video. Try another MP4 or AVI file."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps and fps > 0 else 0

        return {
            "fps": fps,
            "frame_count": int(frame_count),
            "width": width,
            "height": height,
            "duration": duration
        }

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================
# 6. EXTRACT FRAMES FROM A VIDEO SEGMENT
# =========================================================

def extract_segment_frames(
    video_bytes,
    start_time,
    end_time,
    number_of_frames=NUM_FRAMES
):

    import tempfile
    import os

    frames = []
    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as temp_file:

            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError("Could not read the uploaded video.")

        timestamps = np.linspace(
            start_time,
            max(start_time, end_time - 0.05),
            number_of_frames
        )

        for timestamp in timestamps:

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                float(timestamp * 1000)
            )

            success, frame = cap.read()

            if success:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(frame)

                image.thumbnail((384, 384))

                frames.append(image)

        if not frames:
            raise ValueError(
                "No frames could be extracted from this segment."
            )

        while len(frames) < number_of_frames:
            frames.append(frames[-1].copy())

        return frames

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================
# 7. TIMESFORMER PREDICTION
# =========================================================

def predict_video_segment(
    frames,
    processor,
    model,
    device
):

    inputs = processor(
        frames,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=-1
        )[0]

    top_values, top_indices = torch.topk(
        probabilities,
        k=3
    )

    results = []

    for score, index in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):

        label = model.config.id2label[index]

        results.append({
            "label": label,
            "confidence": round(score * 100, 2)
        })

    return results


# =========================================================
# 8. ANALYZE VIDEO CHRONOLOGICALLY
# =========================================================

def analyze_video(
    video_bytes,
    duration,
    processor,
    model,
    device,
    progress_bar,
    status_text
):

    segment_duration = max(
        5.0,
        duration / MAX_SEGMENTS
    )

    segments = []

    start_times = np.arange(
        0,
        duration,
        segment_duration
    )[:MAX_SEGMENTS]

    total_segments = len(start_times)

    for index, start_time in enumerate(start_times):

        end_time = min(
            start_time + segment_duration,
            duration
        )

        status_text.text(
            f"Analyzing segment {index + 1}/{total_segments} "
            f"({start_time:.1f}s to {end_time:.1f}s)"
        )

        frames = extract_segment_frames(
            video_bytes,
            start_time,
            end_time
        )

        predictions = predict_video_segment(
            frames,
            processor,
            model,
            device
        )

        segments.append({
            "segment": index + 1,
            "start": round(float(start_time), 2),
            "end": round(float(end_time), 2),
            "predictions": predictions,
            "frames": frames
        })

        progress_bar.progress(
            (index + 1) / total_segments
        )

    return segments


# =========================================================
# 9. PREPARE ANALYSIS TEXT
# =========================================================

def build_analysis_context(segments):

    lines = []

    for segment in segments:

        prediction_text = ", ".join([
            f"{item['label']} ({item['confidence']}%)"
            for item in segment["predictions"]
        ])

        lines.append(
            f"Time {segment['start']}s to {segment['end']}s: "
            f"{prediction_text}"
        )

    return "\n".join(lines)


# =========================================================
# 10. GENERATE SUMMARY USING OLLAMA
# =========================================================

def generate_summary(context, duration):

    prompt = f"""
You are a video analysis assistant.

The following are action classifications produced by a
TimeSformer video classification model.

Video duration: {duration:.2f} seconds.

Chronological model results:
{context}

Write a clear video summary using the results.

Requirements:
1. Organize the summary chronologically.
2. Mention the approximate timestamps.
3. Describe the likely actions.
4. Do not invent people, objects, or events.
5. Mention uncertainty when classifications are ambiguous.
6. Clearly state that this is a summary based on sampled
   frames and model predictions.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# =========================================================
# 11. ANSWER VIDEO QUESTIONS
# =========================================================

def answer_video_question(question, context, summary):

    prompt = f"""
You are a video question-answering assistant.

Answer the user's question using only the available
video analysis information.

VIDEO SUMMARY:
{summary}

TIMESTAMPED TIMESFORMER PREDICTIONS:
{context}

USER QUESTION:
{question}

Instructions:
- Give a direct and understandable answer.
- Use timestamps when relevant.
- Do not invent details that are not supported.
- If the available information is insufficient, say so.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["message"]["content"]



# =========================================================
# 12. HTML HEADER
# =========================================================

html_path = Path(__file__).parent / "index.html"

if html_path.exists():
    components.html(
        html_path.read_text(encoding="utf-8"),
        height=155,
        scrolling=False
    )
else:
    st.title("🎬 Video Intelligence")


# =========================================================
# 13. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.write("**Video model**")
    st.caption("TimeSformer · Kinetics-400")

    st.write("**Summary model**")
    st.caption(OLLAMA_MODEL)

    st.divider()

    st.markdown("### Processing information")
    st.write("Frames per segment:", NUM_FRAMES)
    st.write("Maximum segments:", MAX_SEGMENTS)

    st.write(
        "Device:",
        "GPU" if torch.cuda.is_available() else "CPU"
    )

    st.divider()

    if st.button("Clear analysis", use_container_width=True):

        for key in [
            "video_analysis",
            "video_name",
            "video_bytes",
            "video_duration",
            "summary",
            "chat_history",
            "analysis_time"
        ]:
            st.session_state[key] = (
                [] if key == "chat_history" else None
            )

        st.rerun()


# =========================================================
# 14. TWO-COLUMN MAIN WORKSPACE
# =========================================================

left_col, right_col = st.columns(
    [1, 1.15],
    gap="large"
)


# =========================================================
# LEFT SIDE: VIDEO UPLOAD AND PREVIEW
# =========================================================

with left_col:

    st.markdown(
        '<div class="section-title">📁 Upload Your Video</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Choose a video file to extract frames and analyze actions.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Select video",
        type=["mp4", "avi", "mov", "mkv"],
        help="For your first test, use a short video."
    )

    video_info = None
    video_bytes = None

    if uploaded_file is not None:

        video_bytes = uploaded_file.getvalue()

        if video_bytes != st.session_state.video_bytes:

            st.session_state.video_bytes = video_bytes
            st.session_state.video_name = uploaded_file.name
            st.session_state.video_analysis = None
            st.session_state.video_duration = None
            st.session_state.summary = None
            st.session_state.chat_history = []
            st.session_state.analysis_time = None

        st.markdown("---")

        st.markdown("### 🎞️ Video Preview")

        st.video(video_bytes)

        st.markdown("### 📋 Video Information")

        try:

            video_info = get_video_info(video_bytes)

            st.metric(
                "Duration",
                f"{video_info['duration']:.2f} seconds"
            )

            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.metric(
                    "Resolution",
                    f"{video_info['width']} × {video_info['height']}"
                )

            with info_col2:
                st.metric(
                    "Frame Rate",
                    f"{video_info['fps']:.2f} FPS"
                )

            st.caption(f"File: {uploaded_file.name}")

            st.caption(
                f"File size: {uploaded_file.size / (1024 * 1024):.2f} MB"
            )

        except Exception as error:

            st.error(f"Video reading error: {error}")
            video_info = None

    else:

        st.info("Upload a video to see its preview here.")

        st.markdown(
            """
            <div class="panel">
                <h4>🎬 Video workspace</h4>
                <p>Upload your video to get started.</p>
                <p style="color:#8995a8;font-size:13px;">
                    Supported formats: MP4, AVI, MOV and MKV
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# RIGHT SIDE: AI ANALYSIS AND GENERATED ANSWERS
# =========================================================

with right_col:

    st.markdown(
        '<div class="section-title">✨ AI Video Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Analyze video actions and generate an intelligent summary.'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    if uploaded_file is not None and video_info is not None:

        if st.button(
            "🚀 Analyze Video",
            type="primary",
            use_container_width=True
        ):

            if video_info["duration"] <= 0:

                st.error("Could not determine video duration.")

            else:

                try:

                    start_time = time.time()

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner("Loading TimeSformer model..."):

                        processor, model, device = load_timesformer()

                    status_text.info(
                        "TimeSformer model loaded. Starting video analysis..."
                    )

                    segments = analyze_video(
                        video_bytes,
                        video_info["duration"],
                        processor,
                        model,
                        device,
                        progress_bar,
                        status_text
                    )

                    context = build_analysis_context(segments)

                    status_text.info(
                        "Generating chronological summary using Ollama..."
                    )

                    summary = generate_summary(
                        context,
                        video_info["duration"]
                    )

                    elapsed = round(
                        time.time() - start_time,
                        2
                    )

                    st.session_state.video_analysis = segments
                    st.session_state.summary = summary
                    st.session_state.video_duration = video_info["duration"]
                    st.session_state.analysis_time = elapsed
                    st.session_state.chat_history = []

                    progress_bar.progress(1.0)

                    status_text.success(
                        "Video analysis completed successfully!"
                    )

                    st.rerun()

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to Ollama. Please start Ollama "
                        "and check that llama3.2:3b is installed."
                    )

                except Exception as error:

                    st.error(f"Video analysis failed: {error}")

    else:

        st.button(
            "🚀 Analyze Video",
            disabled=True,
            use_container_width=True
        )

        st.caption("Upload a video on the left to enable analysis.")


    # =====================================================
    # 15. DISPLAY GENERATED RESULTS
    # =====================================================

    if st.session_state.video_analysis:

        st.markdown("---")

        st.markdown(
            '<div class="section-title">📊 Analysis Results</div>',
            unsafe_allow_html=True
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Segments",
                len(st.session_state.video_analysis)
            )

        with metric2:
            st.metric(
                "Duration",
                f"{st.session_state.video_duration:.1f}s"
            )

        with metric3:
            st.metric(
                "Processing",
                f"{st.session_state.analysis_time:.1f}s"
            )

        st.markdown("---")

        tab_summary, tab_timeline, tab_frames, tab_questions = st.tabs([
            "📝 Summary",
            "⏱️ Timeline",
            "🖼️ Frames",
            "💬 Ask Video"
        ])

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        with tab_summary:

            st.markdown("### 📝 Chronological Summary")

            st.markdown(
                """
                <div class="section-subtitle">
                    AI-generated explanation of the analyzed video.
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.container(border=True):

                st.write(st.session_state.summary)

            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="video_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

        # -------------------------------------------------
        # TIMELINE
        # -------------------------------------------------

        with tab_timeline:

            st.markdown("### ⏱️ Detected Actions")

            for segment in st.session_state.video_analysis:

                with st.container(border=True):

                    st.markdown(
                        f"**Segment {segment['segment']}**"
                    )

                    st.caption(
                        f"{segment['start']}s - {segment['end']}s"
                    )

                    for prediction in segment["predictions"]:

                        st.write(
                            f"• **{prediction['label']}** "
                            f"({prediction['confidence']:.2f}%)"
                        )

        # -------------------------------------------------
        # EXTRACTED FRAMES
        # -------------------------------------------------

        with tab_frames:

            st.markdown("### 🖼️ Extracted Video Frames")

            for segment in st.session_state.video_analysis:

                st.markdown(
                    f"**Segment {segment['segment']}** "
                    f"({segment['start']}s - {segment['end']}s)"
                )

                frame_columns = st.columns(4)

                for index, frame in enumerate(segment["frames"]):

                    with frame_columns[index % 4]:

                        st.image(
                            frame,
                            use_container_width=True,
                            caption=f"Frame {index + 1}"
                        )

        # -------------------------------------------------
        # ASK QUESTIONS
        # -------------------------------------------------

        with tab_questions:

            st.markdown("### 💬 Ask Questions About Your Video")

            st.caption(
                "Ask questions using the detected actions and generated summary."
            )

            for message in st.session_state.chat_history:

                with st.chat_message(
                    message["role"],
                    avatar="👤" if message["role"] == "user" else "🎬"
                ):

                    st.markdown(message["content"])

            question = st.chat_input(
                "Ask something about the video..."
            )

            if question:

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })

                context = build_analysis_context(
                    st.session_state.video_analysis
                )

                try:

                    with st.spinner("Preparing your answer..."):

                        answer = answer_video_question(
                            question,
                            context,
                            st.session_state.summary
                        )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Could not answer the question: {error}"
                    )

    else:

        st.markdown("---")

        st.markdown("### 💡 Analysis Pipeline")

        steps = [
            ("01", "Video Frame Extraction"),
            ("02", "TimeSformer Classification"),
            ("03", "Chronological Event Grouping"),
            ("04", "Ollama Summary Generation"),
            ("05", "Video Question Answering")
        ]

        for number, title in steps:

            st.markdown(
                f"""
                <div class="panel" style="margin-bottom:10px;">
                    <span style="
                        color:#bd78a0;
                        font-weight:800;
                        margin-right:12px;
                    ">{number}</span>
                    <b style="color:#394968;">{title}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.info(
            "Your generated summary, timeline, extracted frames and "
            "question-answering section will appear here after analysis."
        )


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Video Analysis and Summarization using Spatiotemporal Transformers
        <br>
        Experiment 20 · Nexus Video Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
