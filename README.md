# 🎵 Music Genre Classification

This project is a deep learning-based system for classifying the genre of a music track from an uploaded `.mp3` file. It uses mel spectrogram features and a trained neural network to predict one of ten common music genres.

---

## 📌 Features

- 🎶 Upload an `.mp3` file and play it directly in the web app.
- 🤖 Predict the music genre using a pre-trained deep learning model.
- 🎨 Clean and interactive user interface built with Streamlit.
- 🧠 Uses Mel Spectrograms for feature extraction from audio.
- 🎯 Supports 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, and `rock`.

---

## 📂 Use Cases

1. **Automatic Tagging of Unknown Music**  
   Useful for apps like Saavn and Wynk to categorize new or uncategorized songs automatically.

2. **Music Recommendation Systems**  
   Improves user recommendations by understanding and predicting music genres.

3. **Music Information Retrieval (MIR)**  
   Helps organize and retrieve music efficiently from large databases.

4. **Machine Learning & Music Analysis**  
   Assists researchers in understanding trends, user preferences, and audio content patterns.

---

## 🚀 How It Works

1. **Upload MP3**  
   User uploads a `.mp3` file through the UI.

2. **Feature Extraction**  
   Audio is chunked and converted to Mel Spectrograms using `librosa`.

3. **Prediction**  
   The processed data is passed through a trained model (`.h5` file) to predict the genre.

4. **Result**  
   The predicted genre is displayed with a success animation.

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit  
- **Audio Processing:** Librosa  
- **Model:** TensorFlow / Keras  
- **Language:** Python  
- **Deployment:** Streamlit Cloud (optional)

---

## 🧪 Requirements

```bash
pip install -r requirements.txt
