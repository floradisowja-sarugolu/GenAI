
import io
import random

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import streamlit as st
import torch
import torch.nn as nn


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MuseGAN Music Studio",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRACKS = ["Piano", "Bass", "Melody", "Drums"]
PITCHES = 48
STEPS = 32
LATENT_DIM = 64


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "generator": None,
    "discriminator": None,
    "trained": False,
    "generated_music": None,
    "training_losses": [],
    "training_complete": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CUSTOM LIGHT RED + LIGHT BLUE THEME
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(
            115deg,
            #fff8f8 0%,
            #f2f8ff 100%
        );
        color: #35445f;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: #394b68 !important;
    }

    p, label, span {
        color: #586981;
    }

    .hero {
        background: linear-gradient(110deg, #fce4e4, #e5f1ff);
        border: 1px solid #f0dfe6;
        border-radius: 19px;
        padding: 27px;
        margin-bottom: 22px;
    }

    .hero-tag {
        font-size: 11px;
        letter-spacing: 2px;
        color: #a35c6e;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 31px;
        font-weight: 800;
        color: #354765;
        margin-bottom: 7px;
    }

    .hero-subtitle {
        color: #687b94;
        font-size: 14px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 750;
        color: #394b68;
        margin: 8px 0 13px;
    }

    .section-subtitle {
        color: #8793a6;
        font-size: 12px;
        margin-bottom: 12px;
    }

    .metric-box {
        background: linear-gradient(120deg, #fceaea, #eaf3ff);
        border: 1px solid #e8e7ef;
        border-radius: 13px;
        padding: 15px;
        text-align: center;
    }

    .metric-label {
        color: #8190a5;
        font-size: 12px;
    }

    .metric-value {
        color: #394b68;
        font-size: 22px;
        font-weight: 800;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(100deg, #f5dada, #dcecff);
        border: 1px solid #e8cbd3;
        border-radius: 11px;
        color: #394b68;
        font-weight: 750;
        min-height: 43px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #cda8b6;
        color: #293c5a;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.75);
        border-radius: 15px;
        border: 1px solid #e9eaf1;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #cbd9ed;
        border-radius: 12px;
    }

    hr {
        border-color: #e6e9f0;
    }

    .footer {
        text-align: center;
        color: #98a4b6;
        font-size: 12px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">GENERATIVE AI LABORATORY</div>
    <div class="hero-title">🎹 MuseGAN Music Studio</div>
    <div class="hero-subtitle">
        Generate creative multi-track music using Generative
        Adversarial Networks.
    </div>
    <div style="
        display:inline-block;
        margin-top:15px;
        background:#ffffffb8;
        padding:6px 13px;
        border-radius:20px;
        color:#8c6277;
        font-size:11px;
        font-weight:bold;
    ">
        
   
</div>
""", unsafe_allow_html=True)


# =========================================================
# GENERATOR
# =========================================================

class MusicGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(
                1024,
                len(TRACKS) * PITCHES * STEPS
            ),
            nn.Sigmoid()
        )

    def forward(self, noise):

        output = self.network(noise)

        return output.view(
            -1,
            len(TRACKS),
            PITCHES,
            STEPS
        )


# =========================================================
# DISCRIMINATOR
# =========================================================

class MusicDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                len(TRACKS) * PITCHES * STEPS,
                512
            ),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, music):

        music = music.view(music.size(0), -1)

        return self.network(music)


# =========================================================
# SYNTHETIC MULTI-TRACK TRAINING DATA
# =========================================================

def create_training_data(number_of_samples=256):

    data = np.zeros(
        (
            number_of_samples,
            len(TRACKS),
            PITCHES,
            STEPS
        ),
        dtype=np.float32
    )

    for sample in range(number_of_samples):

        root = random.choice([0, 2, 4, 5, 7, 9])

        # PIANO: chord-like patterns
        for step in range(0, STEPS, 4):

            for interval in [0, 4, 7]:

                pitch = 24 + (root + interval) % 12

                data[sample, 0, pitch, step:step + 3] = 1

        # BASS: repeated low notes
        for step in range(0, STEPS, 4):

            pitch = root + 3

            data[sample, 1, pitch, step:step + 3] = 1

        # MELODY: simple higher notes
        for step in range(STEPS):

            if step % 2 == 0:

                pitch = random.randint(25, 40)

                data[sample, 2, pitch, step] = 1

        # DRUMS: rhythmic percussion pattern
        for step in range(STEPS):

            if step % 4 == 0:
                data[sample, 3, 0, step] = 1

            if step % 8 == 4:
                data[sample, 3, 4, step] = 1

    return torch.tensor(data, dtype=torch.float32)


# =========================================================
# TRAIN GAN
# =========================================================

def train_gan(epochs, learning_rate):

    generator = MusicGenerator().to(DEVICE)
    discriminator = MusicDiscriminator().to(DEVICE)

    real_data = create_training_data().to(DEVICE)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
    )

    criterion = nn.BCELoss()

    batch_size = 32

    progress = st.progress(0)
    status = st.empty()

    loss_history = []

    for epoch in range(epochs):

        indices = torch.randperm(len(real_data))

        generator_loss_total = 0
        discriminator_loss_total = 0
        batches = 0

        for start in range(0, len(real_data), batch_size):

            batch_indices = indices[start:start + batch_size]

            real_music = real_data[batch_indices]

            current_batch = real_music.size(0)

            real_labels = torch.ones(
                current_batch,
                1,
                device=DEVICE
            )

            fake_labels = torch.zeros(
                current_batch,
                1,
                device=DEVICE
            )

            # ---------------------------------------------
            # TRAIN DISCRIMINATOR
            # ---------------------------------------------

            discriminator_optimizer.zero_grad()

            real_output = discriminator(real_music)

            real_loss = criterion(
                real_output,
                real_labels
            )

            noise = torch.randn(
                current_batch,
                LATENT_DIM,
                device=DEVICE
            )

            fake_music = generator(noise)

            fake_output = discriminator(fake_music.detach())

            fake_loss = criterion(
                fake_output,
                fake_labels
            )

            discriminator_loss = real_loss + fake_loss

            discriminator_loss.backward()

            discriminator_optimizer.step()

            # ---------------------------------------------
            # TRAIN GENERATOR
            # ---------------------------------------------

            generator_optimizer.zero_grad()

            noise = torch.randn(
                current_batch,
                LATENT_DIM,
                device=DEVICE
            )

            generated_music = generator(noise)

            output = discriminator(generated_music)

            generator_loss = criterion(
                output,
                real_labels
            )

            generator_loss.backward()

            generator_optimizer.step()

            generator_loss_total += generator_loss.item()
            discriminator_loss_total += discriminator_loss.item()

            batches += 1

        avg_g = generator_loss_total / batches
        avg_d = discriminator_loss_total / batches

        loss_history.append({
            "epoch": epoch + 1,
            "generator": avg_g,
            "discriminator": avg_d
        })

        progress.progress(
            (epoch + 1) / epochs
        )

        status.info(
            f"Training epoch {epoch + 1}/{epochs} | "
            f"Generator loss: {avg_g:.4f} | "
            f"Discriminator loss: {avg_d:.4f}"
        )

    progress.empty()
    status.success("GAN training completed!")

    return generator, discriminator, loss_history


# =========================================================
# GENERATE MUSIC
# =========================================================

def generate_music(generator, temperature):

    generator.eval()

    with torch.no_grad():

        noise = torch.randn(
            1,
            LATENT_DIM,
            device=DEVICE
        ) * temperature

        generated = generator(noise)

    music = generated[0].cpu().numpy()

    # Convert probabilities to note activations.
    # This is a simple threshold-based piano-roll conversion.
    music = (music > 0.5).astype(np.float32)

    return music


# =========================================================
# CREATE MIDI FILE
# =========================================================

def create_midi(music, tempo):

    midi = pretty_midi.PrettyMIDI(
        initial_tempo=tempo
    )

    programs = [
        pretty_midi.instrument_name_to_program("Acoustic Grand Piano"),
        pretty_midi.instrument_name_to_program("Electric Bass (finger)"),
        pretty_midi.instrument_name_to_program("Lead 1 (square)"),
        0
    ]

    track_names = [
        "Piano",
        "Bass",
        "Melody",
        "Drums"
    ]

    for track_index, track_name in enumerate(track_names):

        is_drum = track_index == 3

        instrument = pretty_midi.Instrument(
            program=programs[track_index],
            name=track_name,
            is_drum=is_drum
        )

        piano_roll = music[track_index]

        for pitch_index in range(PITCHES):

            active_steps = np.where(
                piano_roll[pitch_index] > 0
            )[0]

            for step in active_steps:

                start = step * 0.25

                end = start + 0.2

                if is_drum:

                    actual_pitch = [36, 38, 42, 46][
                        pitch_index % 4
                    ]

                else:

                    actual_pitch = 36 + pitch_index

                velocity = 95

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=int(actual_pitch),
                    start=float(start),
                    end=float(end)
                )

                instrument.notes.append(note)

        midi.instruments.append(instrument)

    output = io.BytesIO()

    midi.write(output)

    output.seek(0)

    return output.getvalue()


# =========================================================
# PIANO ROLL VISUALIZATION
# =========================================================

def plot_music(music):

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 9),
        sharex=True
    )

    track_colors = [
        "#D987A0",
        "#83A9D8",
        "#A78BC7",
        "#8CB9B0"
    ]

    for index, ax in enumerate(axes):

        ax.imshow(
            music[index],
            aspect="auto",
            origin="lower",
            cmap="Blues" if index in [1, 2] else "Reds",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

        ax.set_ylabel(TRACKS[index], fontsize=10)

        ax.set_yticks([])

        ax.grid(
            axis="x",
            alpha=0.15
        )

        ax.set_facecolor("#FBFCFF")

    axes[-1].set_xlabel("Time Steps")

    fig.suptitle(
        "Generated Multi-Track Piano Roll",
        fontsize=15,
        color="#394B68"
    )

    fig.tight_layout()

    return fig


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🎛️ Music Studio")

    st.caption("Experiment 21 settings")

    st.divider()

    st.markdown("### GAN Training")

    epochs = st.slider(
        "Training epochs",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )

    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.0001, 0.0002, 0.0005, 0.001],
        value=0.0002
    )

    st.divider()

    st.markdown("### Music Settings")

    tempo = st.slider(
        "Music tempo (BPM)",
        min_value=60,
        max_value=180,
        value=110
    )

    temperature = st.slider(
        "Generation variation",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    st.divider()

    st.markdown("### System")

    st.write(
        "Processing device:",
        "GPU" if torch.cuda.is_available() else "CPU"
    )

    st.caption("Local PyTorch implementation")


# =========================================================
# MAIN WORKSPACE
# =========================================================

left_col, right_col = st.columns(
    [0.9, 1.1],
    gap="large"
)


# =========================================================
# LEFT: TRAINING AND GENERATION CONTROLS
# =========================================================

with left_col:

    st.markdown(
        '<div class="section-title">🎼 Music Generation Controls</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Train the GAN and create a new multi-track composition.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.markdown("### 🧠 GAN Model Training")

        st.write(
            "Train the Generator and Discriminator using "
            "synthetic multi-track training patterns."
        )

        if st.button(
            "🚀 Train GAN Model",
            use_container_width=True
        ):

            try:

                with st.spinner("Training Generator and Discriminator..."):

                    generator, discriminator, losses = train_gan(
                        epochs,
                        learning_rate
                    )

                st.session_state.generator = generator
                st.session_state.discriminator = discriminator
                st.session_state.training_losses = losses
                st.session_state.trained = True
                st.session_state.training_complete = True

                st.success("GAN model trained successfully!")

            except Exception as error:

                st.error(f"Training error: {error}")

    if st.session_state.trained:

        st.success("Model ready for music generation.")

    with st.container(border=True):

        st.markdown("### ✨ Generate New Music")

        st.write(
            "Create a new composition using the trained Generator."
        )

        if st.button(
            "🎶 Generate Music",
            use_container_width=True,
            disabled=not st.session_state.trained
        ):

            music = generate_music(
                st.session_state.generator,
                temperature
            )

            st.session_state.generated_music = music

            st.success("New composition generated!")


# =========================================================
# RIGHT: RESULTS
# =========================================================

with right_col:

    st.markdown(
        '<div class="section-title">🎧 Generated Music Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'View the generated tracks and export the MIDI composition.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.generated_music is not None:

        music = st.session_state.generated_music

        with st.container(border=True):

            st.markdown("### 📊 Composition Overview")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Tracks", "4")

            with col2:
                st.metric("Tempo", f"{tempo} BPM")

            with col3:
                st.metric("Steps", str(STEPS))

        with st.container(border=True):

            st.markdown("### 🎼 Piano Roll Visualization")

            figure = plot_music(music)

            st.pyplot(
                figure,
                use_container_width=True
            )

            plt.close(figure)

        with st.container(border=True):

            st.markdown("### 🎹 Track Activity")

            for index, track in enumerate(TRACKS):

                active_notes = int(
                    np.sum(music[index])
                )

                st.write(
                    f"**{track}:** {active_notes} active notes"
                )

                st.progress(
                    min(active_notes / (PITCHES * STEPS), 1.0)
                )

        midi_bytes = create_midi(
            music,
            tempo
        )

        st.download_button(
            "⬇️ Download Multi-Track MIDI",
            data=midi_bytes,
            file_name="musegan_composition.mid",
            mime="audio/midi",
            use_container_width=True
        )

    else:

        with st.container(border=True):

            st.markdown("### 🎵 Your Music Will Appear Here")

            st.info(
                "First train the GAN model. Then click Generate Music "
                "to create your multi-track composition."
            )

            st.markdown("""
            <div style="
                text-align:center;
                padding:35px 10px;
                background:linear-gradient(110deg,#fff0f0,#edf5ff);
                border-radius:14px;
                margin-top:15px;
            ">
                <div style="font-size:55px;">🎹 🎸 🎵 🥁</div>
                <h3 style="color:#394b68;">Waiting for your composition</h3>
                <p style="color:#8793a6;font-size:13px;">
                    Four instrument tracks will be generated here.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # TRAINING HISTORY
    # -----------------------------------------------------

    if st.session_state.training_losses:

        st.markdown("---")

        st.markdown("### 📈 GAN Training History")

        history = st.session_state.training_losses

        fig, ax = plt.subplots(figsize=(9, 3.5))

        ax.plot(
            [x["epoch"] for x in history],
            [x["generator"] for x in history],
            label="Generator Loss"
        )

        ax.plot(
            [x["epoch"] for x in history],
            [x["discriminator"] for x in history],
            label="Discriminator Loss"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_facecolor("#FBFCFF")
        fig.patch.set_facecolor("#FFFFFF")
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div class="footer">
    MuseGAN Music Studio · Experiment 21
    <br>
    Multi-Track Music Generation using Generative Adversarial Networks
</div>
""", unsafe_allow_html=True)