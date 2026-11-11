
import streamlit as st
import requests
import matplotlib.pyplot as plt

from io import BytesIO

from rag_engine import (
    extract_text,
    split_text,
    create_vector_store,
    retrieve_documents,
    generate_answer
)

from style_transfer import neural_style_transfer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */

.stApp {
    background: linear-gradient(
        135deg,
        #f8faff 0%,
        #fff5fa 100%
    );
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #eaf3ff 0%,
        #fceaf3 100%
    );
    border-right: 1px solid #e5e9f2;
}

/* Sidebar branding */

.sidebar-title {
    color: #203858;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.5px;
    padding-top: 12px;
    padding-bottom: 3px;
}

.sidebar-subtitle {
    color: #75839b;
    font-size: 13px;
    margin-bottom: 28px;
}

/* Main welcome heading */

.main-title {
    font-size: 44px;
    font-weight: 850;
    color: #172b4d;
    letter-spacing: -1.2px;
    line-height: 1.3;
    margin-bottom: 8px;
}

/* Page headings */

.page-title {
    font-size: 36px;
    font-weight: 800;
    color: #203858;
    margin-bottom: 8px;
}

/* Subtitle */

.subtitle {
    font-size: 15px;
    color: #71809b;
    margin-bottom: 28px;
}

/* Section headings */

h3 {
    color: #30486d;
    font-weight: 700;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg,
        #a9caff,
        #f2b9d5
    );
    color: #253858;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
    width: 100%;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #91afd9;
    color: #172b4d;
}

/* Download button */

.stDownloadButton > button {
    background: #dceaff;
    color: #29466f;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    min-height: 42px;
}

/* File uploader */

div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #a9bde0;
    border-radius: 12px;
    padding: 10px;
}

/* Chat messages */

div[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #edf0f7;
    border-radius: 14px;
    padding: 12px;
}

/* Radio navigation */

[data-testid="stSidebar"] [data-testid="stRadio"] {
    padding-top: 8px;
}

/* Divider */

hr {
    border-color: #e5eaf2;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False


# --------------------------------------------------
# LEFT SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            ✨ AI Studio
        </div>
        <div class="sidebar-subtitle">
            Your intelligent AI workspace
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "<p style='color:#526681;font-size:12px;"
        "font-weight:700;letter-spacing:1px;'>"
        "NAVIGATION</p>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🧠 AI Knowledge Assistant",
            "🎨 Neural Style Studio"
        ],
        label_visibility="collapsed"
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="main-title">
            Welcome to AI Studio ✨
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Your workspace for intelligent question answering
            and creative image transformation.
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# AI KNOWLEDGE ASSISTANT
# --------------------------------------------------

elif page == "🧠 AI Knowledge Assistant":

    st.markdown(
        """
        <div class="page-title">
            🧠 AI Knowledge Assistant
        </div>
        <div class="subtitle">
            Upload your documents and ask questions
            using Retrieval-Augmented Generation.
        </div>
        """,
        unsafe_allow_html=True
    )

    # DOCUMENT UPLOAD

    st.markdown("### Upload Your Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="rag_upload"
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} document(s) selected."
        )

        if st.button(
            "Process Documents",
            key="process_docs"
        ):

            with st.spinner(
                "Extracting text and creating your knowledge base..."
            ):

                try:

                    documents = extract_text(
                        uploaded_files
                    )

                    if not documents:

                        st.error(
                            "No readable text was found. "
                            "Please check your documents."
                        )

                    else:

                        chunks = split_text(documents)

                        if not chunks:

                            st.error(
                                "No text chunks could be generated."
                            )

                        else:

                            index = create_vector_store(
                                chunks
                            )

                            st.session_state.chunks = chunks

                            st.session_state.vector_index = index

                            st.session_state.documents_ready = True

                            st.session_state.chat_history = []

                            st.success(
                                "Knowledge base created successfully!"
                            )

                            st.info(
                                f"Documents processed: {len(documents)} | "
                                f"Text chunks: {len(chunks)}"
                            )

                except Exception as e:

                    st.error(
                        f"Document processing error: {str(e)}"
                    )

    # KNOWLEDGE BASE STATUS

    if st.session_state.documents_ready:

        st.success(
            f"Knowledge base ready: "
            f"{len(st.session_state.chunks)} chunks indexed."
        )

    st.divider()

    # CHAT SECTION

    st.markdown("### Ask Your Questions")

    if st.session_state.documents_ready:

        if st.button(
            "Clear Chat History",
            key="clear_chat"
        ):

            st.session_state.chat_history = []

            st.rerun()

        # Display previous messages

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if (
                    message["role"] == "assistant"
                    and message.get("sources")
                ):

                    with st.expander(
                        "View Retrieved Sources"
                    ):

                        for source in message["sources"]:

                            st.markdown(
                                f"**{source['source']}**"
                            )

                            st.caption(
                                f"Similarity score: "
                                f"{source['score']:.3f}"
                            )

                            st.write(
                                source["text"]
                            )

        # User question

        question = st.chat_input(
            "Ask anything about your uploaded documents..."
        )

        if question:

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):

                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching documents and generating answer..."
                ):

                    try:

                        # Retrieve relevant document chunks

                        retrieved = retrieve_documents(
                            question,
                            st.session_state.chunks,
                            st.session_state.vector_index,
                            top_k=3
                        )

                        # Generate LLM answer

                        answer = generate_answer(
                            question,
                            retrieved
                        )

                        st.markdown(answer)

                        # Display sources

                        with st.expander(
                            "View Retrieved Sources"
                        ):

                            for source in retrieved:

                                st.markdown(
                                    f"**{source['source']}**"
                                )

                                st.caption(
                                    f"Similarity: "
                                    f"{source['score']:.3f}"
                                )

                                st.write(
                                    source["text"]
                                )

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved
                        })

                    except requests.exceptions.ConnectionError:

                        st.error(
                            "Cannot connect to Ollama. "
                            "Please start Ollama and check the model."
                        )

                    except Exception as e:

                        st.error(
                            f"Error generating answer: {str(e)}"
                        )

    else:

        st.info(
            "Please upload and process your documents "
            "before asking questions."
        )


# --------------------------------------------------
# NEURAL STYLE STUDIO
# --------------------------------------------------

elif page == "🎨 Neural Style Studio":

    st.markdown(
        """
        <div class="page-title">
            🎨 Neural Style Studio
        </div>
        <div class="subtitle">
            Transform your photographs into artistic images
            using Neural Style Transfer and VGG19.
        </div>
        """,
        unsafe_allow_html=True
    )

    # IMAGE UPLOAD SECTION

    st.markdown("### Upload Images")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 📷 Content Image")

        content_file = st.file_uploader(
            "Upload original photograph",
            type=["jpg", "jpeg", "png"],
            key="content_image"
        )

        if content_file:

            st.image(
                content_file,
                caption="Original Content Image",
                use_container_width=True
            )

    with col2:

        st.markdown("#### 🖌️ Style Image")

        style_file = st.file_uploader(
            "Upload artistic reference",
            type=["jpg", "jpeg", "png"],
            key="style_image"
        )

        if style_file:

            st.image(
                style_file,
                caption="Style Reference Image",
                use_container_width=True
            )

    st.divider()

    # TRANSFORMATION SETTINGS

    st.markdown("### Transformation Settings")

    col1, col2, col3 = st.columns(3)

    with col1:

        image_size = st.selectbox(
            "Image Resolution",
            [128, 192, 256, 320],
            index=2
        )

    with col2:

        steps = st.select_slider(
            "Optimization Steps",
            options=[20, 30, 50, 75, 100],
            value=50
        )

    with col3:

        learning_rate = st.selectbox(
            "Learning Rate",
            [0.01, 0.02, 0.03],
            index=1
        )

    st.caption(
        "Higher resolution and more optimization steps "
        "may require additional processing time."
    )

    st.divider()

    # GENERATE IMAGE

    if st.button(
        "✨ Generate Artistic Image",
        key="generate_style"
    ):

        if content_file is None or style_file is None:

            st.warning(
                "Please upload both content and style images."
            )

        else:

            progress_bar = st.progress(0)

            status = st.empty()

            try:

                status.info(
                    "Loading VGG19 and preparing images..."
                )

                result, losses = neural_style_transfer(
                    content_file,
                    style_file,
                    steps=steps,
                    learning_rate=learning_rate,
                    image_size=image_size,
                    progress_callback=progress_bar.progress
                )

                status.success(
                    "Neural Style Transfer completed successfully!"
                )

                st.markdown("### Generated Result")

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        content_file,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        result,
                        caption="AI Generated Image",
                        use_container_width=True
                    )

                # DOWNLOAD GENERATED IMAGE

                image_buffer = BytesIO()

                result.save(
                    image_buffer,
                    format="PNG"
                )

                image_buffer.seek(0)

                st.download_button(
                    label="⬇ Download Generated Image",
                    data=image_buffer,
                    file_name="neural_style_output.png",
                    mime="image/png"
                )

                # OPTIMIZATION LOSS GRAPH

                st.markdown("### Optimization Loss")

                fig, ax = plt.subplots()

                ax.plot(
                    range(1, len(losses) + 1),
                    losses
                )

                ax.set_xlabel("Optimization Steps")

                ax.set_ylabel("Total Loss")

                ax.set_title(
                    "Neural Style Transfer Loss"
                )

                ax.grid(
                    True,
                    alpha=0.3
                )

                st.pyplot(fig)

                plt.close(fig)

            except Exception as e:

                status.error(
                    f"Style transfer failed: {str(e)}"
                )

                st.exception(e)
