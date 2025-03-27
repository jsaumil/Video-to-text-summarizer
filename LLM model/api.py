# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv


# load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",# paste the model link
#     task = "text-generation"
# )


# model = ChatHuggingFace(llm=llm)


# result =  model.invoke("Make a summary of this .txt file:") # ask your question here


# print(result.content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables (e.g., Hugging Face API key)
load_dotenv()

# Initialize the Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Model to use
    task="text-generation"  # Task type
)

# Wrap the model in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Read the content of the .txt file
txt_file_path = "transcribed_text.txt"  # Replace with the path to your .txt file
with open(txt_file_path, "r") as file:
    file_content = file.read()

# Create a prompt for summarization
prompt = f"Summarize the following text in English language:\n\n{file_content}"

# Generate the summary using the Hugging Face model
result = model.invoke(prompt)

# Print the summary
print("Summary:")
print(result.content)