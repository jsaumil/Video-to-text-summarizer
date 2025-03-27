import os
import streamlit as st
import whisper
import ffmpeg
from langchain_ollama import OllamaLLM  # Updated import
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# Initialize Ollama LLM
ollama_model = OllamaLLM(model="summarizer")

# Step 1: Extract Audio from Video using ffmpeg
def extract_audio_from_video(video_path, audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format="mp3")
        .run(overwrite_output=True)
    )
    st.success(f"Audio extracted and saved to {audio_path}")

# Step 2: Transcribe Audio to Text using Whisper
def transcribe_audio_to_text(audio_path):
    model = whisper.load_model("base")  # Change model size as needed
    result = model.transcribe(audio_path)
    transcribed_text = result["text"]
    st.success("Transcription completed.")
    return transcribed_text

# Step 3: Summarize Text using Ollama
def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    result = ollama_model.invoke(prompt)  # Fixed method call
    return result

# Main Function to Run the Pipeline
def video_to_summary(video_path):
    audio_path = "extracted_audio.mp3"
    extract_audio_from_video(video_path, audio_path)
    transcribed_text = transcribe_audio_to_text(audio_path)
    
    st.subheader("Transcribed Text:")
    st.write(transcribed_text)
    
    summary = summarize_text(transcribed_text)
    st.subheader("Summary:")
    st.write(summary)

# Streamlit App
def main():
    st.title("Video to Summary Converter")
    st.write("Upload a video file to extract audio, transcribe it, and generate a summary.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        video_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video(video_path)
        
        # Process the video
        if st.button("Generate Summary"):
            with st.spinner("Processing video..."):
                video_to_summary(video_path)

# Run the Streamlit app
if __name__ == "__main__":
    main()  # ✅ Removed asyncio, runs synchronously
