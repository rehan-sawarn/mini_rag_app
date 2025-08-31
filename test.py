from groq import Client
import os

client = Client(api_key=os.getenv("GROQ_API_KEY"))
print("Groq client initialized successfully!")