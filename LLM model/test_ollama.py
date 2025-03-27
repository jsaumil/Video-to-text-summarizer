
from langchain_ollama.chat_models import ChatOllama

# Initialize the chat model
model = ChatOllama(model="llama3.2:1b")

result = model.invoke("What is the capital of India")

print(result.content)
