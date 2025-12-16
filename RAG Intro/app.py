import bs4
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

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), # You can add more URLs here
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(
            class_ = ("post-content", "post-title", "post-header") # Adjust based on the website structure HTML classes
        )
    )
)

docs = loader.load()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(
    documents=splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)

retriever = vector_store.as_retriever() # Retrieve top 1 similar document

# RAG prompt
prompt = hub.pull("rlm/rag-prompt") # You can also create your own prompt and push it to LangChain Hub

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    #print(docs)
    #print(format_docs(docs))

    for chunk in rag_chain.stream("What is maximum inner product search?"):
        print(chunk, end="", flush=True)