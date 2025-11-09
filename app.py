import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import tempfile
import soundfile as sf
from scipy.stats import skew, kurtosis
from audio_recorder_streamlit import audio_recorder

le_command = pickle.load(open("model/le_command.pkl", "rb"))
model_command = pickle.load(open("model/model_command.pkl", "rb"))
scaler_command = pickle.load(open("model/scaler_command.pkl", "rb"))

le_speaker = pickle.load(open("model/le_speaker.pkl", "rb"))
model_speaker = pickle.load(open("model/model_speaker.pkl", "rb"))
scaler_speaker = pickle.load(open("model/scaler_speaker.pkl", "rb"))

THRESHOLD = 70  

def agg_stats(x):
    return {
        'mean': np.mean(x),
        'std': np.std(x),
        'median': np.median(x),
        'skew': skew(x) if len(x) > 2 else 0.0,
        'kurtosis': kurtosis(x) if len(x) > 2 else 0.0
    }

def extract_features(file_path):
    sr = 16000
    y, _ = librosa.load(file_path, sr=sr)

    features = {}

    # temporal
    z = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    features.update({f'zcr_{k}': v for k, v in agg_stats(z).items()})
    features.update({f'rms_{k}': v for k, v in agg_stats(rms).items()})
    features['duration'] = len(y) / sr

    # spectral
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]

    features.update({f'sc_{k}': v for k, v in agg_stats(sc).items()})
    features.update({f'sb_{k}': v for k, v in agg_stats(sb).items()})
    features.update({f'roll_{k}': v for k, v in agg_stats(roll).items()})
    features.update({f'contrast_{k}': v for k, v in agg_stats(spec_contrast).items()})
    features.update({f'flat_{k}': v for k, v in agg_stats(flat).items()})

    # mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        stats = agg_stats(mfcc[i])
        features.update({f'mfcc{i+1}_{k}': v for k, v in stats.items()})

    # f0 (pitch)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        features.update({f'f0_{k}': v for k, v in agg_stats(f0).items()})
    except Exception:
        features.update({
            'f0_mean': 0.0,
            'f0_std': 0.0,
            'f0_median': 0.0,
            'f0_skew': 0.0,
            'f0_kurtosis': 0.0
        })

    return features

selected_features_command = [
    'mfcc8_mean', 'mfcc8_median', 'mfcc11_mean', 'sc_mean', 'mfcc3_median'
]

selected_features_speaker = [
    'mfcc3_mean', 'mfcc1_std', 'mfcc4_median', 'mfcc4_mean', 'mfcc3_median',
    'contrast_mean', 'contrast_median', 'mfcc11_median', 'mfcc5_skew',
    'mfcc7_std', 'flat_median', 'mfcc11_mean', 'f0_median', 'f0_std', 'mfcc3_std'
]

def predict_from_file(file_path):

    feats = extract_features(file_path)
    df = pd.DataFrame([feats])

    X_command = df[selected_features_command]
    X_speaker = df[selected_features_speaker]

    X_command_scaled = scaler_command.transform(X_command)
    X_speaker_scaled = scaler_speaker.transform(X_speaker)

    pred_command = model_command.predict(X_command_scaled)[0]
    pred_speaker = model_speaker.predict(X_speaker_scaled)[0]

    pred_command_prob = model_command.predict_proba(X_command_scaled)[0]
    pred_speaker_prob = model_speaker.predict_proba(X_speaker_scaled)[0]

    confidence_command = float(max(pred_command_prob) * 100)
    confidence_speaker = float(max(pred_speaker_prob) * 100)

    label_command = le_command.inverse_transform([pred_command])[0]
    speaker_label = le_speaker.inverse_transform([pred_speaker])[0]

    recognized = (confidence_command >= THRESHOLD) and (confidence_speaker >= THRESHOLD)

    return {
        "recognized": recognized,
        "label_command": label_command,
        "speaker_label": speaker_label,
        "confidence_command": confidence_command,
        "confidence_speaker": confidence_speaker
    }

st.set_page_config(page_title="Indentifikasi Suara Buka-Tutup & Pengenalan Suara", page_icon="🎧")

st.title("Indentifikasi Suara Buka-Tutup & Pengenalan Suara")
st.write("Deteksi **perintah suara** dan **siapa pembicara** dari audio yang kamu upload atau rekam.")

input_mode = st.radio(
    "Pilih sumber input:",
    ["Upload File", "Rekam Suara"],
    horizontal=True
)

sr = 16000  # sample rate global

# ---------- MODE UPLOAD ----------
if input_mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload file audio (disarankan .wav)", 
        type=["wav", "mp3", "ogg"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Deteksi dari File"):
            with st.spinner("Sedang memproses audio..."):
                # Simpan ke file sementara
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result = predict_from_file(tmp_path)

            st.subheader("Hasil Prediksi")
            if result["recognized"]:
                st.success(f"Dikenali sebagai : **{result['speaker_label']}**")
                st.info(f"Identifikasi perintah : **{result['label_command']}**")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Command", f"{result['confidence_command']:.2f}%")
                with col2:
                    st.metric("Confidence Speaker", f"{result['confidence_speaker']:.2f}%")
            else:
                st.error("Suara tidak dikenali (bukan 2 orang yang terdaftar).")
                st.write(
                    f"- Confidence command: **{result['confidence_command']:.2f}%** "
                    f"(threshold {THRESHOLD}%)"
                )
                st.write(
                    f"- Confidence speaker: **{result['confidence_speaker']:.2f}%** "
                    f"(threshold {THRESHOLD}%)"
                )

else:
    st.subheader("Rekam Suara dari Browser")
    st.write("Klik tombol di bawah untuk mulai/stop rekaman.")

    # Komponen perekam audio di browser
    audio_bytes = audio_recorder(
        text="Klik untuk merekam / berhenti",
        recording_color="#e55353",
        neutral_color="#6c757d",
        icon_size="2x",
    )

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Deteksi Rekaman Ini"):
            # Simpan bytes ke file sementara .wav
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with st.spinner("Sedang memproses rekaman..."):
                result = predict_from_file(tmp_path)

            st.subheader("Hasil Prediksi")
            if result["recognized"]:
                st.success(f"Dikenali sebagai : **{result['speaker_label']}**")
                st.info(f"Identifikasi perintah : **{result['label_command']}**")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Command", f"{result['confidence_command']:.2f}%")
                with col2:
                    st.metric("Confidence Speaker", f"{result['confidence_speaker']:.2f}%")
            else:
                st.error("Suara tidak dikenali (bukan 2 orang yang terdaftar).")
                st.write(
                    f"- Confidence command: **{result['confidence_command']:.2f}%** "
                    f"(threshold {THRESHOLD}%)"
                )
                st.write(
                    f"- Confidence speaker: **{result['confidence_speaker']:.2f}%** "
                    f"(threshold {THRESHOLD}%)"
                )

