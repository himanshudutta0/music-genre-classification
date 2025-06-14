import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import gdown
import os
from tensorflow.image import resize

# ---------------------------------------
# 📦 Load Model from Google Drive
# ---------------------------------------
@st.cache_resource
def load_model():
    file_id = "1sJgfdAoW1S5-EtY8QFY07Lq5hzAM4qu1"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "Music_Genre_Classification_model.h5"

    if not os.path.exists(output):
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, output, quiet=False)
        st.success("✅ Download complete!")

    model = tf.keras.models.load_model(output)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# ---------------------------------------
# 🎧 Load and Preprocess Audio
# ---------------------------------------
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# ---------------------------------------
# 🔮 Predict Genre
# ---------------------------------------
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)

    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# ---------------------------------------
# 🎵 Streamlit UI
# ---------------------------------------
st.header("🎶 MUSIC GENRE CLASSIFICATION")

test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

if test_mp3 is not None:
    filepath = 'Test_Music/' + test_mp3.name

    # Play Button
    if st.button("Play Audio"):
        st.audio(test_mp3)

    # Predict Button
    if st.button("Predict"):
        with st.spinner("Please Wait.."):
            X_test = load_and_preprocess_data(filepath)
            result_index = model_prediction(X_test)
            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            st.balloons()
            st.markdown("**:blue[Model Prediction:] It's a :red[{}] music**".format(label[result_index]))
