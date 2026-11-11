
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CIFAR-10 Neural Vision",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = "cifar10_cnn.keras"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# =========================================================
# 2. LIGHT THEME
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
    color: #263247;
}

.main .block-container {
    max-width: 1450px;
    padding: 2rem 2.3rem;
}

h1, h2, h3 {
    color: #1e2d45 !important;
}

.subtitle {
    color: #7b879b;
    font-size: 14px;
    margin-top: -12px;
    margin-bottom: 20px;
}

.section-title {
    color: #263b59;
    font-size: 20px;
    font-weight: 650;
    margin-bottom: 12px;
}

.stat-card {
    background: white;
    border: 1px solid #e1e7f0;
    border-radius: 11px;
    padding: 17px;
    text-align: center;
}

.stat-label {
    color: #8491a5;
    font-size: 13px;
}

.stat-value {
    color: #243550;
    font-size: 25px;
    font-weight: 700;
    margin-top: 5px;
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

div[data-baseweb="select"] > div {
    background: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SESSION STATE
# =========================================================

if "model_ready" not in st.session_state:
    st.session_state.model_ready = os.path.exists(MODEL_PATH)

if "training_history" not in st.session_state:
    st.session_state.training_history = None

if "test_metrics" not in st.session_state:
    st.session_state.test_metrics = None

if "predictions" not in st.session_state:
    st.session_state.predictions = []


# =========================================================
# 4. LOAD CIFAR-10 DATASET
# =========================================================

@st.cache_data
def load_dataset():

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return x_train, y_train, x_test, y_test


# =========================================================
# 5. BUILD CNN MODEL
# =========================================================

def build_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(32, 32, 3)),

        # First convolution block
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.20),

        # Second convolution block
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Third convolution block
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.30),

        # Classification layers
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128, activation="relu"
        ),

        tf.keras.layers.Dropout(0.40),

        tf.keras.layers.Dense(
            10, activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================================================
# 6. TRAINING PROGRESS CALLBACK
# =========================================================

class StreamlitProgress(tf.keras.callbacks.Callback):

    def __init__(self, progress_bar, status_text, total_epochs):

        super().__init__()

        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        progress = (epoch + 1) / self.total_epochs

        self.progress_bar.progress(progress)

        self.status_text.text(
            f"Completed epoch {epoch + 1}/{self.total_epochs} | "
            f"Accuracy: {logs.get('accuracy', 0):.4f} | "
            f"Loss: {logs.get('loss', 0):.4f}"
        )


# =========================================================
# 7. TRAIN MODEL
# =========================================================

def train_model(epochs, batch_size):

    x_train, y_train, x_test, y_test = load_dataset()

    model = build_cnn()

    progress_bar = st.progress(0)

    status_text = st.empty()

    callback = StreamlitProgress(
        progress_bar,
        status_text,
        epochs
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=0
    )

    model.save(MODEL_PATH)

    progress_bar.progress(1.0)

    status_text.success("CNN training completed successfully.")

    return model, history.history


# =========================================================
# 8. LOAD SAVED MODEL
# =========================================================

@st.cache_resource
def load_saved_model():

    return tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# 9. EVALUATE MODEL
# =========================================================

def evaluate_model(model):

    x_train, y_train, x_test, y_test = load_dataset()

    predictions = model.predict(
        x_test,
        batch_size=128,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predicted_classes
    )

    precision = precision_score(
        y_test,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predicted_classes
    )

    report = classification_report(
        y_test,
        predicted_classes,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": matrix,
        "report": report,
        "actual": y_test,
        "predicted": predicted_classes
    }


# =========================================================
# 10. IMAGE PREPROCESSING
# =========================================================

def preprocess_uploaded_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (32, 32),
        Image.Resampling.LANCZOS
    )

    image_array = np.array(
        image,
        dtype="float32"
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# 11. HEADER
# =========================================================

header, header_info = st.columns([3, 1])

with header:

    st.title("🧠 CIFAR-10 Neural Vision")

    st.markdown(
        '<div class="subtitle">'
        'Deep Convolutional Neural Network · '
        '10 Image Categories · Image Classification'
        '</div>',
        unsafe_allow_html=True
    )

with header_info:

    st.markdown(
        "<div style='text-align:right;padding-top:20px;"
        "color:#8491a5;font-size:13px;'>"
        "EXPERIMENT 17</div>",
        unsafe_allow_html=True
    )


if st.session_state.model_ready:

    st.success("CNN model file detected.")

else:

    st.info("CNN model is not trained yet. Start training below.")


st.divider()


# =========================================================
# 12. APPLICATION TABS
# =========================================================

tab_train, tab_predict, tab_evaluate, tab_info = st.tabs([
    "Model Training",
    "Image Classification",
    "Model Evaluation",
    "About CNN"
])


# =========================================================
# 13. TRAINING TAB
# =========================================================

with tab_train:

    st.markdown(
        '<div class="section-title">⚙️ Train Your CNN Model</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Train a convolutional neural network using "
        "the CIFAR-10 image dataset."
    )

    col1, col2 = st.columns(2)

    with col1:

        epochs = st.selectbox(
            "Number of Epochs",
            [3, 5, 8, 10, 15],
            index=1
        )

    with col2:

        batch_size = st.selectbox(
            "Batch Size",
            [32, 64, 128],
            index=1
        )

    st.caption(
        "For your first run, use 3 or 5 epochs. "
        "More epochs generally require more training time."
    )

    train_button = st.button(
        "🚀 Start CNN Training",
        type="primary",
        use_container_width=True
    )

    if train_button:

        try:

            with st.spinner(
                "Loading CIFAR-10 and preparing the CNN..."
            ):

                model, history = train_model(
                    epochs,
                    batch_size
                )

            st.session_state.model_ready = True

            st.session_state.training_history = history

            st.session_state.test_metrics = None

            load_saved_model.clear()

            st.success(
                "Your CNN model has been trained and saved."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Training failed: {error}"
            )

    if st.session_state.training_history:

        history = st.session_state.training_history

        st.divider()

        st.subheader("Training Performance")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:

            st.markdown("**Accuracy**")

            accuracy_fig, accuracy_ax = plt.subplots()

            accuracy_ax.plot(
                history["accuracy"],
                label="Training Accuracy"
            )

            accuracy_ax.plot(
                history["val_accuracy"],
                label="Validation Accuracy"
            )

            accuracy_ax.set_xlabel("Epoch")
            accuracy_ax.set_ylabel("Accuracy")
            accuracy_ax.legend()
            accuracy_ax.grid(alpha=0.25)

            st.pyplot(accuracy_fig)

            plt.close(accuracy_fig)

        with graph_col2:

            st.markdown("**Loss**")

            loss_fig, loss_ax = plt.subplots()

            loss_ax.plot(
                history["loss"],
                label="Training Loss"
            )

            loss_ax.plot(
                history["val_loss"],
                label="Validation Loss"
            )

            loss_ax.set_xlabel("Epoch")
            loss_ax.set_ylabel("Loss")
            loss_ax.legend()
            loss_ax.grid(alpha=0.25)

            st.pyplot(loss_fig)

            plt.close(loss_fig)


# =========================================================
# 14. IMAGE CLASSIFICATION TAB
# =========================================================

with tab_predict:

    st.markdown(
        '<div class="section-title">🔍 Image Classification</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an image to predict which CIFAR-10 "
        "category it belongs to."
    )

    if not os.path.exists(MODEL_PATH):

        st.warning(
            "Train the CNN model first using the Model Training tab."
        )

    else:

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="image_upload"
        )

        if uploaded_file:

            try:

                image = Image.open(uploaded_file)

                left, right = st.columns(2)

                with left:

                    st.image(
                        image,
                        caption="Uploaded Image",
                        use_container_width=True
                    )

                with right:

                    model = load_saved_model()

                    processed_image = preprocess_uploaded_image(
                        image
                    )

                    probabilities = model.predict(
                        processed_image,
                        verbose=0
                    )[0]

                    predicted_index = int(
                        np.argmax(probabilities)
                    )

                    predicted_label = CLASS_NAMES[
                        predicted_index
                    ]

                    confidence = float(
                        probabilities[predicted_index]
                    )

                    st.markdown("### Prediction")

                    st.success(
                        f"Predicted Class: {predicted_label.title()}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    st.markdown("**Top 3 Predictions**")

                    top_indices = np.argsort(
                        probabilities
                    )[-3:][::-1]

                    for index in top_indices:

                        label = CLASS_NAMES[index]

                        score = float(probabilities[index])

                        st.write(
                            f"{label.title()} · {score * 100:.2f}%"
                        )

                        st.progress(score)

            except Exception as error:

                st.error(
                    f"Could not classify this image: {error}"
                )


# =========================================================
# 15. MODEL EVALUATION TAB
# =========================================================

with tab_evaluate:

    st.markdown(
        '<div class="section-title">📊 CNN Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Evaluate the trained CNN using the CIFAR-10 test dataset."
    )

    if not os.path.exists(MODEL_PATH):

        st.warning(
            "Train the model before running evaluation."
        )

    else:

        if st.button(
            "Run Model Evaluation",
            type="primary"
        ):

            try:

                with st.spinner(
                    "Evaluating the model on test images..."
                ):

                    model = load_saved_model()

                    metrics = evaluate_model(model)

                    st.session_state.test_metrics = metrics

                st.success(
                    "Model evaluation completed."
                )

            except Exception as error:

                st.error(
                    f"Evaluation failed: {error}"
                )

        metrics = st.session_state.test_metrics

        if metrics:

            st.divider()

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Test Accuracy",
                    f"{metrics['accuracy'] * 100:.2f}%"
                )

            with metric2:

                st.metric(
                    "Precision",
                    f"{metrics['precision'] * 100:.2f}%"
                )

            with metric3:

                st.metric(
                    "Recall",
                    f"{metrics['recall'] * 100:.2f}%"
                )

            st.subheader("Confusion Matrix")

            matrix_fig, matrix_ax = plt.subplots(
                figsize=(10, 8)
            )

            matrix_ax.imshow(
                metrics["confusion_matrix"],
                interpolation="nearest"
            )

            matrix_ax.set_title("CIFAR-10 Confusion Matrix")

            matrix_ax.set_xlabel("Predicted Label")

            matrix_ax.set_ylabel("Actual Label")

            matrix_ax.set_xticks(
                range(10),
                CLASS_NAMES,
                rotation=45,
                ha="right"
            )

            matrix_ax.set_yticks(
                range(10),
                CLASS_NAMES
            )

            for i in range(10):

                for j in range(10):

                    matrix_ax.text(
                        j,
                        i,
                        metrics["confusion_matrix"][i, j],
                        ha="center",
                        va="center",
                        fontsize=7
                    )

            matrix_fig.tight_layout()

            st.pyplot(matrix_fig)

            plt.close(matrix_fig)

            st.subheader("Classification Report")

            report_df = pd.DataFrame(
                metrics["report"]
            ).transpose()

            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )


# =========================================================
# 16. ABOUT CNN TAB
# =========================================================

with tab_info:

    st.markdown(
        '<div class="section-title">🧠 Understanding the CNN</div>',
        unsafe_allow_html=True
    )

    st.write(
        "A Convolutional Neural Network is a deep learning "
        "architecture designed to recognize patterns in images."
    )

    st.markdown("### Architecture used")

    architecture = pd.DataFrame([
        ["Input", "32 × 32 × 3"],
        ["Convolution Block 1", "32 filters + ReLU"],
        ["Convolution Block 2", "64 filters + ReLU"],
        ["Convolution Block 3", "128 filters + ReLU"],
        ["Pooling", "Max Pooling"],
        ["Regularization", "Dropout + Batch Normalization"],
        ["Fully Connected", "128 neurons"],
        ["Output", "10 classes + Softmax"]
    ], columns=["Layer", "Description"])

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### CIFAR-10 Categories")

    category_cols = st.columns(5)

    for index, name in enumerate(CLASS_NAMES):

        category_cols[index % 5].info(
            name.title()
        )

    st.markdown("### How the system works")

    st.markdown("""
    1. Load the CIFAR-10 image dataset.
    2. Normalize pixel values.
    3. Build the CNN architecture.
    4. Train the model using convolution and pooling layers.
    5. Predict categories for unseen images.
    6. Evaluate the model using accuracy, precision, recall
       and a confusion matrix.
    """)


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    "CIFAR-10 Neural Vision · Experiment 17 · "
    "Deep CNN Image Classification · TensorFlow / Keras"
)
