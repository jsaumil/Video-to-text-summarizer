# from fastapi import FastAPI, File, UploadFile
# import whisper
# import ffmpeg
# import shutil
# import os
# from langchain_community.llms import Ollama  # Correct import
# from fastapi.responses import JSONResponse

# app = FastAPI()
# ollama_model = Ollama(base_url="http://localhost:11434", model="llama3.2:1b")

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Extract audio from video
# def extract_audio(video_path, audio_path):
#     (
#         ffmpeg
#         .input(video_path)
#         .output(audio_path, format="mp3")
#         .run(overwrite_output=True)
#     )
#     return audio_path

# # Transcribe audio using Whisper
# def transcribe_audio(audio_path):
#     model = whisper.load_model("base")
#     result = model.transcribe(audio_path)
#     return result["text"]

# # Summarize text using Ollama
# def summarize_text(text):
#     return ollama_model(text)

# @app.post("/upload/")
# async def upload_video(file: UploadFile = File(...)):
#     video_path = os.path.join(UPLOAD_DIR, file.filename)
#     audio_path = os.path.splitext(video_path)[0] + ".mp3"
    
#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # Process Video
#     extract_audio(video_path, audio_path)
#     transcribed_text = transcribe_audio(audio_path)
#     summary = summarize_text(transcribed_text)
    
#     return JSONResponse(content={
#         "transcription": transcribed_text,
#         "summary": summary
#     })

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=3000)

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import whisper
import ffmpeg
import torch
from langchain_ollama import OllamaLLM  # Updated import
from dotenv import load_dotenv
import os

# Load environment variables (if needed)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Ollama LLM with llama3.2:1b model
ollama_model = OllamaLLM(base_url="http://localhost:11434", model="llama3.2:1b")

# Step 1: Extract Audio from Video using ffmpeg
def extract_audio_from_video(video_path, audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format="mp3")
        .run(overwrite_output=True)
    )
    print(f"Audio extracted and saved to {audio_path}")

# Step 2: Transcribe Audio to Text using Whisper
def transcribe_audio_to_text(audio_path):
    model = whisper.load_model("base")  # Change model size as needed
    result = model.transcribe(audio_path)
    transcribed_text = result["text"]
    print("Transcription completed.")
    return transcribed_text

# Step 3: Summarize Text using Ollama
def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    result = ollama_model(prompt)
    return result

# API Endpoint to Process Video
@app.route("/process-video", methods=["POST"])
def process_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    video_file = request.files["file"]
    video_path = "uploaded_video.mp4"
    audio_path = "extracted_audio.mp3"

    # Save the uploaded video file
    video_file.save(video_path)

    # Process the video
    extract_audio_from_video(video_path, audio_path)
    transcribed_text = transcribe_audio_to_text(audio_path)
    summary = summarize_text(transcribed_text)

    # Clean up temporary files
    os.remove(video_path)
    os.remove(audio_path)

    return jsonify({"summary": summary})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)