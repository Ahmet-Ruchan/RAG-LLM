from dotenv import load_dotenv
from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

# This session new imports
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough



load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


vector_store = Chroma.from_documents(
    documents=documents,
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"), # OpenAIEmbeddings()
    #embedding=OpenAIEmbeddings()
)

retriever = RunnableLambda(vector_store.similarity_search).bind(k=1) # Retrieve top 1 similar document

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0)

message = """
Answer the question using only the provided context.
{question}

Context : {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", message)
    ]
)

chain = {"context" : retriever, "question" : RunnablePassthrough()} | prompt | llm






if __name__ == "__main__":
    # print("Hello World!")
    # print(vector_store.similarity_search("Dog"))
    # print(vector_store.similarity_search_with_score("Dog"))
    #
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004").embed_query("Dog")
    # #embeddings = OpenAIEmbeddings().embed_query("Dog")
    # print(vector_store.similarity_search_by_vector(embeddings))

    #print(retriever.batch(["dog", "shark"]))

    response = chain.invoke("Tell me about cats.")
    print(response.content)