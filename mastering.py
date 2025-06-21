import streamlit as st
from pydub import AudioSegment, effects
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

AudioSegment.converter = os.path.abspath("bin/ffmpeg")
os.system("chmod +x " + AudioSegment.converter)

st.set_page_config(page_title="🎧 Free Music Mastering Tool", layout="wide")
st.title("🎛️ Free AI-Powered Music Mastering Tool")
st.markdown("Upload your raw track. We'll simulate a bad mix and show how AI mastering improves it!")

uploaded_file = st.file_uploader("Upload your audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.read()
    original_io = io.BytesIO(uploaded_bytes)
    audio_io = io.BytesIO(uploaded_bytes)

    clean_audio = AudioSegment.from_file(audio_io)
    clean_audio = clean_audio.set_frame_rate(44100).set_channels(2).set_sample_width(2)

    def muddy_mix(audio):
        muddy = audio.low_pass_filter(2000)
        muddy = muddy.high_pass_filter(120)
        muddy = muddy.apply_gain(-4)
        return muddy

    muddy_audio = muddy_mix(clean_audio)

    def apply_simple_eq(audio_seg):
        return audio_seg.high_pass_filter(80).low_pass_filter(14000)

    def soft_compress(audio_seg, gain_db=-6):
        return audio_seg.apply_gain(gain_db).normalize(headroom=1.5)

    normalized = effects.normalize(clean_audio, headroom=1.5)
    eq_applied = apply_simple_eq(normalized)
    mastered = soft_compress(eq_applied)

    original_io = io.BytesIO()
    muddy_audio.export(original_io, format="wav", parameters=["-ar", "44100"])
    original_io.seek(0)

    mastered_io = io.BytesIO()
    mastered.export(mastered_io, format="wav", parameters=["-ar", "44100"])
    mastered_io.seek(0)

    st.subheader("🎧 Compare: Simulated Raw Mix vs AI Mastered")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Simulated Raw Mix** (Muddy)")
        st.audio(original_io, format='audio/wav')
    with col2:
        st.markdown("**AI Mastered Version**")
        st.audio(mastered_io, format='audio/wav')

    st.download_button("⬇️ Download Mastered Audio", data=mastered_io, file_name="mastered_track.wav", mime="audio/wav")

    st.subheader("📈 Waveform Visualization")
    def plot_waveform(audio_file, title):
        y, sr = librosa.load(audio_file, sr=None)
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        st.pyplot(fig)

    original_io.seek(0)
    plot_waveform(original_io, "Simulated Raw Mix (Muddy)")
    mastered_io.seek(0)
    plot_waveform(mastered_io, "AI Mastered Audio")

st.markdown("---")
st.markdown(
    'Created with ❤️ by [Amal Alexander](https://www.linkedin.com/in/amal-alexander-305780131/)',
    unsafe_allow_html=True
)