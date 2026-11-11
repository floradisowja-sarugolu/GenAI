
import io
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Neural Style Studio",
    page_icon="🎨",
    layout="wide"
)

STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]

CONTENT_LAYERS = ["block5_conv2"]

STYLE_WEIGHT = 1e-2
CONTENT_WEIGHT = 1e4


# =========================================================
# 2. LIGHT THEME AND CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
    color: #263247;
}

.main .block-container {
    max-width: 1500px;
    padding: 1.5rem 2rem 2rem;
}

h1, h2, h3 {
    color: #1d2d48 !important;
}

.hero {
    text-align: center;
    padding: 12px 0 22px;
}

.hero-title {
    font-size: 34px;
    font-weight: 750;
    color: #1d2d48;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 14px;
    color: #78869b;
    margin-top: 6px;
}

.panel-title {
    text-align: center;
    color: #243650;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 12px;
}

.image-panel {
    background: white;
    border: 1px solid #dfe7ef;
    border-radius: 15px;
    padding: 16px;
    min-height: 100%;
    box-shadow: 0 3px 12px rgba(30, 50, 80, 0.04);
}

.settings-panel {
    background: white;
    border: 1px solid #dfe7ef;
    border-radius: 15px;
    padding: 20px;
    margin-top: 18px;
}

div.stButton > button {
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}

div.stButton > button[kind="primary"] {
    background: #203451;
    color: white;
    border: none;
}

div.stButton > button[kind="primary"]:hover {
    background: #31527c;
    color: white;
}

div[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 9px;
    padding: 8px;
}

div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SESSION STATE
# =========================================================

defaults = {
    "generated_image": None,
    "generation_time": None,
    "content_preview": None,
    "style_preview": None,
    "loss_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. LOAD PRETRAINED VGG19 MODEL
# =========================================================

@st.cache_resource
def load_vgg_model():

    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet"
    )

    vgg.trainable = False

    outputs = [
        vgg.get_layer(name).output
        for name in STYLE_LAYERS + CONTENT_LAYERS
    ]

    model = tf.keras.Model(
        inputs=vgg.input,
        outputs=outputs
    )

    model.trainable = False

    return model


# =========================================================
# 5. IMAGE LOADING AND PREPROCESSING
# =========================================================

def load_image(image_file, max_dimension=256):

    image = Image.open(image_file).convert("RGB")

    width, height = image.size

    scale = max_dimension / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    return tf.convert_to_tensor(
        image_array[None, ...],
        dtype=tf.float32
    )


# =========================================================
# 6. GRAM MATRIX
# =========================================================

def gram_matrix(feature_map):

    gram = tf.linalg.einsum(
        "bijc,bijd->bcd",
        feature_map,
        feature_map
    )

    height = tf.cast(
        tf.shape(feature_map)[1],
        tf.float32
    )

    width = tf.cast(
        tf.shape(feature_map)[2],
        tf.float32
    )

    channels = tf.cast(
        tf.shape(feature_map)[-1],
        tf.float32
    )

    normalization = height * width * channels

    return gram / normalization


# =========================================================
# 7. EXTRACT CONTENT AND STYLE FEATURES
# =========================================================

def extract_features(model, image):

    image = image * 255.0

    image = tf.keras.applications.vgg19.preprocess_input(
        image
    )

    outputs = model(image)

    style_outputs = outputs[:len(STYLE_LAYERS)]

    content_outputs = outputs[len(STYLE_LAYERS):]

    style_features = [
        gram_matrix(feature)
        for feature in style_outputs
    ]

    content_features = content_outputs

    return style_features, content_features


# =========================================================
# 8. NEURAL STYLE TRANSFER
# =========================================================

def neural_style_transfer(
    content_image,
    style_image,
    steps,
    learning_rate,
    progress_bar,
    status_text
):

    model = load_vgg_model()

    status_text.text(
        "Extracting content and style features..."
    )

    # Extract style and content reference features

    style_targets, _ = extract_features(
        model,
        style_image
    )

    _, content_targets = extract_features(
        model,
        content_image
    )

    # Initialize generated image using content image

    generated_image = tf.Variable(
        content_image
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.99,
        epsilon=1e-1
    )

    # -----------------------------------------------------
    # Calculate style and content loss
    # -----------------------------------------------------

    def calculate_loss():

        generated_style, generated_content = extract_features(
            model,
            generated_image
        )

        style_loss = tf.add_n([
            tf.reduce_mean(
                tf.square(generated - target)
            )
            for generated, target in zip(
                generated_style,
                style_targets
            )
        ])

        style_loss = style_loss / len(STYLE_LAYERS)

        content_loss = tf.add_n([
            tf.reduce_mean(
                tf.square(generated - target)
            )
            for generated, target in zip(
                generated_content,
                content_targets
            )
        ])

        content_loss = content_loss / len(CONTENT_LAYERS)

        total_loss = (
            STYLE_WEIGHT * style_loss
            + CONTENT_WEIGHT * content_loss
        )

        return total_loss

    # -----------------------------------------------------
    # Optimization loop
    # -----------------------------------------------------

    loss_history = []

    for step in range(steps):

        with tf.GradientTape() as tape:

            total_loss = calculate_loss()

        gradients = tape.gradient(
            total_loss,
            generated_image
        )

        optimizer.apply_gradients([
            (gradients, generated_image)
        ])

        generated_image.assign(
            tf.clip_by_value(
                generated_image,
                0.0,
                1.0
            )
        )

        current_loss = float(total_loss.numpy())

        loss_history.append(current_loss)

        progress = (step + 1) / steps

        progress_bar.progress(progress)

        status_text.text(
            f"Step {step + 1}/{steps} | "
            f"Total Loss: {current_loss:.4f} | "
            f"{progress * 100:.0f}%"
        )

    # -----------------------------------------------------
    # Convert generated tensor into image
    # -----------------------------------------------------

    final_array = (
        generated_image.numpy()[0] * 255.0
    ).clip(0, 255).astype(np.uint8)

    final_image = Image.fromarray(final_array)

    return final_image, loss_history


# =========================================================
# 9. APPLICATION HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🎨 Neural Style Studio</div>
    <div class="hero-subtitle">
        Transform photographs into artistic images
        using Neural Style Transfer and VGG19
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =========================================================
# 10. THREE IMAGE PANELS
# =========================================================

content_col, style_col, result_col = st.columns(
    3,
    gap="medium"
)


# ---------------------------------------------------------
# CONTENT IMAGE PANEL
# ---------------------------------------------------------

with content_col:

    st.markdown(
        '<div class="panel-title">① Content Image</div>',
        unsafe_allow_html=True
    )

    content_file = st.file_uploader(
        "Choose Content Image",
        type=["jpg", "jpeg", "png", "webp"],
        key="content_upload"
    )

    if content_file:

        content_preview = Image.open(
            content_file
        ).convert("RGB")

        st.image(
            content_preview,
            use_container_width=True
        )

        st.caption(
            f"Selected: {content_file.name}"
        )

        st.session_state.content_preview = content_preview

    else:

        st.info(
            "Upload a photograph whose content "
            "you want to preserve."
        )


# ---------------------------------------------------------
# STYLE IMAGE PANEL
# ---------------------------------------------------------

with style_col:

    st.markdown(
        '<div class="panel-title">② Style Image</div>',
        unsafe_allow_html=True
    )

    style_file = st.file_uploader(
        "Choose Style Image",
        type=["jpg", "jpeg", "png", "webp"],
        key="style_upload"
    )

    if style_file:

        style_preview = Image.open(
            style_file
        ).convert("RGB")

        st.image(
            style_preview,
            use_container_width=True
        )

        st.caption(
            f"Selected: {style_file.name}"
        )

        st.session_state.style_preview = style_preview

    else:

        st.info(
            "Upload a painting or artwork "
            "whose style you want to apply."
        )


# ---------------------------------------------------------
# GENERATED IMAGE PANEL
# ---------------------------------------------------------

with result_col:

    st.markdown(
        '<div class="panel-title">③ Generated Image</div>',
        unsafe_allow_html=True
    )

    if st.session_state.generated_image is not None:

        result_image = st.session_state.generated_image

        st.image(
            result_image,
            use_container_width=True
        )

        st.success("Image generated successfully")

        image_buffer = io.BytesIO()

        result_image.save(
            image_buffer,
            format="PNG"
        )

        st.download_button(
            "⬇ Download Generated Image",
            data=image_buffer.getvalue(),
            file_name="neural_style_transfer.png",
            mime="image/png",
            use_container_width=True
        )

    else:

        st.info(
            "Your generated artwork will appear here "
            "after style transfer."
        )


# =========================================================
# 11. GENERATION SETTINGS
# =========================================================

st.markdown(
    '<div class="settings-panel">',
    unsafe_allow_html=True
)

st.markdown("### ⚙️ Generation Settings")

setting1, setting2, setting3 = st.columns(3)

with setting1:

    steps = st.selectbox(
        "Optimization Steps",
        [20, 30, 50, 75, 100],
        index=2
    )

with setting2:

    resolution = st.selectbox(
        "Image Resolution",
        [128, 192, 256, 320],
        index=2
    )

with setting3:

    learning_rate = st.selectbox(
        "Learning Rate",
        [0.01, 0.02, 0.03],
        index=1
    )

st.caption(
    "Higher resolution and more optimization steps "
    "may improve results but require more processing time."
)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 12. GENERATE IMAGE
# =========================================================

generate_button = st.button(
    "✨ Generate Styled Image",
    type="primary",
    use_container_width=True
)

if generate_button:

    if not content_file:

        st.warning(
            "Please upload a content photograph."
        )

    elif not style_file:

        st.warning(
            "Please upload a style reference image."
        )

    else:

        try:

            progress_bar = st.progress(0)

            status_text = st.empty()

            start_time = time.time()

            # Load and preprocess both images

            content_image = load_image(
                content_file,
                resolution
            )

            style_image = load_image(
                style_file,
                resolution
            )

            # Resize style image to match content dimensions

            content_height = int(
                content_image.shape[1]
            )

            content_width = int(
                content_image.shape[2]
            )

            style_image = tf.image.resize(
                style_image,
                (content_height, content_width),
                method="bilinear"
            )

            # Run neural style transfer

            with st.spinner(
                "Applying artistic style using VGG19..."
            ):

                final_image, loss_history = (
                    neural_style_transfer(
                        content_image,
                        style_image,
                        steps,
                        learning_rate,
                        progress_bar,
                        status_text
                    )
                )

            elapsed_time = round(
                time.time() - start_time,
                2
            )

            # Save result in session state

            st.session_state.generated_image = final_image

            st.session_state.generation_time = elapsed_time

            st.session_state.loss_history = loss_history

            progress_bar.progress(1.0)

            status_text.success(
                "Neural style transfer completed successfully!"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Style transfer failed: {error}"
            )


# =========================================================
# 13. GENERATION DETAILS
# =========================================================

if st.session_state.generated_image is not None:

    st.divider()

    st.markdown("### 📊 Generation Details")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Processing Time",
            f"{st.session_state.generation_time} seconds"
        )

    with metric2:

        st.metric(
            "Optimization Steps",
            len(st.session_state.loss_history)
        )

    with metric3:

        result = st.session_state.generated_image

        st.metric(
            "Output Dimensions",
            f"{result.width} × {result.height}"
        )

    with st.expander("View Optimization Loss Graph"):

        st.line_chart(
            st.session_state.loss_history
        )


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.caption(
    "Experiment 18 | Neural Style Transfer | "
    "VGG19 | TensorFlow and Streamlit"
)
