# import whisper
# import ffmpeg
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_ollama.chat_models import ChatOllama
# from langchain.llms import ollama
# from dotenv import load_dotenv

# # Load environment variables (if needed)
# load_dotenv()

# # Step 1: Extract Audio from Video using ffmpeg
# def extract_audio_from_video(video_path, audio_path):
#     (
#         ffmpeg
#         .input(video_path)
#         .output(audio_path, format="mp3")
#         .run(overwrite_output=True)
#     )
#     print(f"Audio extracted and saved to {audio_path}")

# # Step 2: Transcribe Audio to Text using Whisper
# def transcribe_audio_to_text(audio_path):
#     model = whisper.load_model("base")  # Change model size as needed
#     result = model.transcribe(audio_path)
#     transcribed_text = result["text"]
#     print("Transcription completed.")
#     return transcribed_text

# # Step 3: Summarize Text using ChatOllama
# def summarize_text(text):
#     model = ChatOllama(model="llama3.2:1b")
#     prompt = f"Summarize the following text:\n\n{text}"
#     result = model.invoke(prompt)
#     return result.content

# # Main Function to Run the Pipeline
# def video_to_summary(video_path):
#     audio_path = "extracted_audio.mp3"
#     extract_audio_from_video(video_path, audio_path)
#     transcribed_text = transcribe_audio_to_text(audio_path)
#     print("Transcribed Text:")
#     print(transcribed_text)
    
#     summary = summarize_text(transcribed_text)
#     print("\nSummary:")
#     print(summary)

# # Run the Pipeline
# if __name__ == "__main__":
#     video_path = "videoplayback.mp4"  # Replace with your video file path
#     video_to_summary(video_path)

import whisper
import ffmpeg
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.llms import Ollama
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# Initialize Ollama LLM
ollama_model = Ollama(base_url="http://localhost:11434", model="summarizer")


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

# Main Function to Run the Pipeline
def video_to_summary(video_path):
    audio_path = "extracted_audio.mp3"
    extract_audio_from_video(video_path, audio_path)
    transcribed_text = transcribe_audio_to_text(audio_path)
    print("Transcribed Text:")
    # print(transcribed_text)
    
    summary = summarize_text(transcribed_text)
    print("\nSummary:")
    print(summary)

# Run the Pipeline
if __name__ == "__main__":
    # test_ollama()  # Check if Ollama is working correctly
    video_path = "videoplayback.mp4"  # Replace with your video file path
    video_to_summary(video_path)
