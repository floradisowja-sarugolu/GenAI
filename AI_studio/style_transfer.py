
import tensorflow as tf
import numpy as np

from PIL import Image

# VGG19 configuration
CONTENT_LAYER = "block5_conv2"

STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]

# Load and preprocess image
def load_image(uploaded_file, size=256):

    image = Image.open(uploaded_file).convert("RGB")

    image = image.resize((size, size))

    image = np.array(image, dtype=np.float32)

    image = tf.convert_to_tensor(image)

    return tf.expand_dims(image, axis=0)

# Calculate Gram matrix
def gram_matrix(features):

    gram = tf.linalg.einsum(
        "bijc,bijd->bcd",
        features,
        features
    )

    shape = tf.shape(features)

    height = tf.cast(shape[1], tf.float32)
    width = tf.cast(shape[2], tf.float32)
    channels = tf.cast(shape[3], tf.float32)

    return gram / (height * width * channels)

# Build VGG19 feature extraction model
def build_vgg_model():

    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet"
    )

    vgg.trainable = False

    outputs = [
        vgg.get_layer(name).output
        for name in STYLE_LAYERS + [CONTENT_LAYER]
    ]

    model = tf.keras.Model(
        inputs=vgg.input,
        outputs=outputs
    )

    return model

# Extract content and style features
def extract_features(model, image):

    image = tf.keras.applications.vgg19.preprocess_input(
        image
    )

    outputs = model(image)

    style_features = outputs[:len(STYLE_LAYERS)]

    content_features = outputs[len(STYLE_LAYERS):]

    style_features = [
        gram_matrix(feature)
        for feature in style_features
    ]

    return style_features, content_features

# Perform Neural Style Transfer
def neural_style_transfer(
    content_file,
    style_file,
    steps=50,
    learning_rate=0.02,
    image_size=256,
    progress_callback=None
):

    content_image = load_image(
        content_file, image_size
    )

    style_image = load_image(
        style_file, image_size
    )

    model = build_vgg_model()

    # Extract original image features
    target_style, _ = extract_features(
        model, style_image
    )

    _, target_content = extract_features(
        model, content_image
    )

    # Initialize generated image
    generated_image = tf.Variable(
        content_image
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    total_losses = []

    for step in range(steps):

        with tf.GradientTape() as tape:

            current_style, current_content = extract_features(
                model, generated_image
            )

            # Content loss
            content_loss = tf.reduce_mean(
                tf.square(
                    current_content[0] -
                    target_content[0]
                )
            )

            # Style loss
            style_loss = tf.add_n([
                tf.reduce_mean(
                    tf.square(
                        current_style[i] -
                        target_style[i]
                    )
                )
                for i in range(len(STYLE_LAYERS))
            ]) / len(STYLE_LAYERS)

            # Combined loss
            total_loss = (
                1e-3 * content_loss +
                1e-1 * style_loss
            )

        gradients = tape.gradient(
            total_loss,
            generated_image
        )

        optimizer.apply_gradients([
            (gradients, generated_image)
        ])

        generated_image.assign(
            tf.clip_by_value(
                generated_image, 0, 255
            )
        )

        total_losses.append(float(total_loss.numpy()))

        if progress_callback:
            progress_callback(
                (step + 1) / steps
            )

    # Convert generated image to PIL
    result = tf.cast(
        generated_image[0], tf.uint8
    ).numpy()

    result_image = Image.fromarray(result)

    return result_image, total_losses