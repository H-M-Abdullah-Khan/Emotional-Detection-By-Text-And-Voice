from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, pipeline
import torch
import librosa
import numpy as np
import io
import soundfile as sf

app = Flask(__name__)

# Load pretrained speech emotion model
MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model.eval()

# Load pretrained text sentiment model
text_model = pipeline("sentiment-analysis")

id2label = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
    5: "disgusted",
    6: "surprised"
}

@app.route('/')
def index():
    return render_template_string("""
<!doctype html>
<html lang="en">
<head>
  <title>Voice and Text Emotion Detection</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 30px auto; }
    button { font-size: 18px; padding: 10px 20px; margin-bottom: 10px; }
    textarea { width: 100%; font-size: 16px; padding: 8px; }
    #result, #textResult { margin-top: 20px; font-size: 22px; font-weight: bold; color: #1a73e8; }
  </style>
</head>
<body>
  <h1>Voice and Text Emotion Detection</h1>
  
  <h2>Record Voice</h2>
  <button onclick="startRecording()">Record 3 Seconds</button>
  <p id="status"></p>
  <div id="result"></div>

  <h2>Text Input</h2>
  <textarea id="textInput" rows="4" placeholder="Type your text here..."></textarea><br>
  <button onclick="analyzeText()">Analyze Text</button>
  <div id="textResult"></div>

<script>
  let mediaRecorder;
  let audioChunks = [];

  function startRecording() {
    document.getElementById('result').textContent = '';
    document.getElementById('status').textContent = 'Requesting microphone...';

    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const wavBuffer = encodeWAV(audioBuffer);
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        
        const reader = new FileReader();
        reader.onloadend = () => {
          fetch('/predict_emotion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: reader.result,
          }).then(res => res.json())
            .then(data => {
              if (data.error) {
                document.getElementById('status').textContent = '';
                document.getElementById('result').textContent = 'Error: ' + data.error;
              } else {
                document.getElementById('status').textContent = '';
                document.getElementById('result').textContent = 'Detected Emotion: ' + data.emotion;
              }
            }).catch(() => {
              document.getElementById('status').textContent = '';
              document.getElementById('result').textContent = 'Failed to get response from server.';
            });
        };
        reader.readAsArrayBuffer(wavBlob);
      };

      mediaRecorder.start();
      document.getElementById('status').textContent = 'Recording for 3 seconds...';
      setTimeout(() => {
        mediaRecorder.stop();
        document.getElementById('status').textContent = 'Processing audio...';
      }, 3000);
    }).catch(e => alert('Error accessing microphone: ' + e));
  }

  // WAV encoder implementation
  function encodeWAV(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1;
    const bitsPerSample = 16;

    let samples;
    if (numChannels === 2) {
      const left = audioBuffer.getChannelData(0);
      const right = audioBuffer.getChannelData(1);
      samples = new Float32Array(left.length * 2);
      for (let i = 0; i < left.length; i++) {
        samples[2 * i] = left[i];
        samples[2 * i + 1] = right[i];
      }
    } else {
      samples = audioBuffer.getChannelData(0);
    }

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bitsPerSample / 8, true);
    view.setUint16(32, numChannels * bitsPerSample / 8, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    floatTo16BitPCM(view, 44, samples);

    return buffer;
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7FFF;
      output.setInt16(offset, s, true);
    }
  }

  // Text analysis
  function analyzeText() {
    const text = document.getElementById('textInput').value;
    fetch('/analyze_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    }).then(res => res.json())
      .then(data => {
        if(data.error) {
          document.getElementById('textResult').textContent = 'Error: ' + data.error;
        } else {
          document.getElementById('textResult').textContent = 'Detected Emotion: ' + data.emotion;
        }
      }).catch(() => {
        document.getElementById('textResult').textContent = 'Failed to get response from server.';
      });
  }
</script>
</body>
</html>
    """)

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        audio_bytes = io.BytesIO(request.data)
        audio_bytes.seek(0)
        waveform, sample_rate = sf.read(audio_bytes)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        # Resample if needed
        if sample_rate != feature_extractor.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=feature_extractor.sampling_rate)
            sample_rate = feature_extractor.sampling_rate

        inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_id = logits.argmax().item()
            emotion = id2label.get(predicted_id, "unknown")

        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "No text provided."})

        result = text_model(text)
        emotion = result[0]['label']

        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
