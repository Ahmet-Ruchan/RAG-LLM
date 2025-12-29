from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

"""
load_dotenv()
"""

class RouteQuery(BaseModel):
    """
    Route a user query to the most revelant datasource
    """

    datasource : Literal["vectorstore", "websearch"] = Field(
        ..., # Bu şu demek: kullanıcının sorgusuna göre "vectorstore" veya "websearch" seçeneklerinden birini seçmesi gerekiyor
        description="Given a user question choose to route it to web search or a vectorstore"
    )

llm = ChatOpenAI(
    temperature=0
)

structured_llm_router = llm.with_structured_output(RouteQuery) # Bu, LLM'nin çıktısını RouteQuery modeline göre yapılandırmasını sağlar

system_prompt = """
You are an expert at routing a user question to a vectorstore or a web search tool.
The vectorstore contains documents related to Bi-encoders and Cross-encoders for Sentence Pair Similarity Scoring, AugSBERT: Bi-encoders and 
Chunking Strategies For RAG. Use the vectorstore for questions related to these topics. For all other questions, use the web search tool.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router



"""
if __name__ == "__main__":
    response = question_router.invoke(
                {"question" : "What is the Bi-encoders?"}
            )
    print(response)
"""