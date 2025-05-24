# Voice and Text Emotion Detection Flask App

This project is a simple Flask web application that detects emotions from both voice recordings and text input. It leverages pre-trained models from HuggingFace to automatically classify emotions without any training needed.

## Features

- **Voice Emotion Detection**: Record voice directly in the browser (3 seconds) and detect emotions like happy, sad, angry, neutral, etc.
- **Text Emotion Detection**: Enter text and get the inferred sentiment or emotion.
- Uses state-of-the-art freely available pre-trained models for speech and text emotion classification.

## Getting Started

Follow these instructions to get a copy of the project running on your local machine easily.

### Prerequisites

- Python 3.7 or higher installed.  
  Download from https://python.org/downloads/

### Installation and Running

1. **Clone or download the repository** to your local machine.
     **Run the following command** to clone the repository:
    ``` bash
    git clone https://github.com/H-M-Abdullah-Khan/Emotional-Detection-By-Text-And-Voice/
    ```

2. **Navigate to the project directory** in your terminal or command prompt.

3. **Create and activate a virtual environment** (recommended):

   - On Windows (PowerShell):
     ```
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - On macOS/Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

4. **Install required packages**:
```
pip install flask torch transformers librosa soundfile numpy
```

5. **Run the Flask application**:

```
python app.py
```

6. **Open your web browser** and go to:

```
http://127.0.0.1:5000
```

7. **Use the app**:
- Click **Record 3 Seconds** and speak to detect voice emotion.
- Type text in the text box and click **Analyze Text** to detect text emotion.

## Project Structure

- `app.py`: The main Flask application with routes and logic.
- `README.md`: This file.
- `requirements.txt`: (optional) List of Python packages required.

## Notes

- Make sure your browser has microphone permission to record audio.
- The backend automatically converts recorded audio to WAV for processing.
- The pre-trained models used are large and may take some time to download on first run.

## Contribution

Feel free to fork the project and submit pull requests for improvements or new features.

## License

This project is open-source and available under the MIT License.

---

Happy emotion detecting! 😊
