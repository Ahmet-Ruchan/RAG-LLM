import sys
import os

# Proje kök dizinini (Corrective RAG) path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from CorractiveRAG.ingestion.ingest import retriever
from CorractiveRAG.graph.state import GraphState
from typing import Any, Dict

def retrieve(state: GraphState) -> Dict[str, Any]:

    print("Retrieving relevant documents...")

    question = state["question"]

    documents = retriever.invoke(question)

    return {"question": question, "documents": documents}