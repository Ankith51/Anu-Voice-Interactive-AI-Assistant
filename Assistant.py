from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from stt import recognize_speech
from tts import speak
import os

template = """
You are Anu, a helpful AI assistant. Answer the user's queries conversationally.

Here is the conversation history:
{context}

User: {question}

Anu:
"""

model = OllamaLLM(model="llama3.2:1b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def handle_conversation():
    context = ""
    print("Hello! I am Anu, your AI assistant. Say 'exit' to stop.")
    
    while True:
        print("Listening...")
        user_input = recognize_speech()
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Goodbye! Have a great day!")
            speak("Goodbye! Have a great day!")
            break

        result = chain.invoke({"context": context, "question": user_input})
        print("Anu:", result)
        speak(result)

        context += f"\nUser: {user_input}\nAnu: {result}"  # Store conversation history
        clear_screen()

if __name__ == "__main__":
    handle_conversation()
