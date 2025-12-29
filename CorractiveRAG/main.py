import sys
import os

# Proje kök dizinini (Corrective RAG) path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "What is your name?"}))