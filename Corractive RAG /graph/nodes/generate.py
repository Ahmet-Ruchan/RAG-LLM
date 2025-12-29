import sys
import os

# Proje kök dizinini (Corrective RAG) path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph.chains.generation import generation_chain
from graph.state import GraphState
from typing import Any, Dict

def generate(state: GraphState) -> Dict[str, Any]:

    print("Generating answer from LLM...")
    question = state["question"]
    documents = state["documents"]

    generate = generation_chain.invoke(
        {"context": documents, "question": question}
    )

    return {"generation": generate["text"]}