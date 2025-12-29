"""
import sys
import os

# Proje kök dizinini (Corrective RAG) path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ingestion.ingest import retriever
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field


load_dotenv()

llm = ChatOpenAI(
    temperature=0
)

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents
    """

    binary_score : str = Field(
        description="Documents are relevance to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments) # Bu, LLM'nin çıktısını GradeDocuments modeline göre yapılandırmasını sağlar

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
If the document contains keyword or semantic meaning related to question, grade it as relevant. 
\n Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved documents: {documents} User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

"""
if __name__ == "__main__":
    user_question = "What is Bi-encoders?"
    docs = retriever.get_relevant_documents(user_question)
    retrieved_document = docs[0].page_content
    response = retrieval_grader.invoke(
            {"question": user_question, "documents": retrieved_document}
        )
    print(response)
"""