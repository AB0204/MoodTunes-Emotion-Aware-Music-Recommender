# 🎵 MoodTunes: Facial Emotion-Based Music Recommender

An intelligent, interactive music recommendation system that captures real-time facial expressions to predict user emotions and fetch customized, mood-matching playlists using the Spotify Web API.

---

## 🏗️ Architecture & Stack

MoodTunes connects a deep learning vision pipeline with real-time web services and a desktop client:

1. **Facial Emotion Classifier (Keras CNN):**
   - Built on a Convolutional Neural Network (CNN) trained on facial expression datasets.
   - Detects seven core human emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`.
   - Utilizes OpenCV to capture real-time webcam frame sequences and run sliding-window face detection (Haar Cascade).

2. **Spotify Integration (Spotipy API):**
   - Connects to the Spotify Web API via OAuth token flows.
   - Maps classified facial emotions directly to targeted Spotify audio features:
     - **Happy / Surprise:** High Valence (positivity), High Energy.
     - **Sad / Fear:** Lower Valence, Lower Tempo.
     - **Angry:** High Energy, High Tempo.
   - Dynamically aggregates playlists and fetches song IDs in real-time.

3. **Client Interface (PyQt5):**
   - A desktop GUI showcasing live camera frames, real-time emotion classifications with confidence bars, and a dynamic recommendation feed with album artwork.

---

## 📂 Project Structure

- `PyQt5.ipynb`: Desktop GUI code that coordinates OpenCV webcam streams, Keras models, and Spotify calls.
- `Spotipy.ipynb` & `Training_Spotipy.ipynb`: Scripting to authenticate with Spotify and build recommendation filters.
- `best_model.h5`: Saved pre-trained weights for the Keras CNN facial emotion classifier.
- `Modal.ipynb`: Analysis notebook describing model exploration and system workflows.

---

## 🚀 Getting Started

### 1. Requirements
Ensure you are using Python 3.8+ inside a virtual environment:
```bash
pip install tensorflow keras opencv-python spotipy PyQt5 pandas numpy
```

### 2. Run the Desktop Client
Activate your virtual environment and execute the PyQt5 GUI notebook cells or export to a script:
```bash
python -m ipykernel install --user --name=venv3.11
# Or run the PyQt5 interface directly from VS Code
```
