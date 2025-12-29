from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = ["https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag?ref=dailydoseofds.com",
        "https://www.dailydoseofds.com/bi-encoders-and-cross-encoders-for-sentence-pair-similarity-scoring-part-1/",
        "https://www.dailydoseofds.com/augsbert-bi-encoders-cross-encoders-for-sentence-pair-similarity-scoring-part-2/",
        "https://www.dailydoseofds.com/llmops-crash-course-part-1/"
        ]

docs = [WebBaseLoader(url).load() for url in urls]

doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)

splits = text_splitter.split_documents(documents=doc_list)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    collection_name="corractive-rag",
    persist_directory="./.chromadb/corractive-rag"
)

retreiver = Chroma(
    collection_name="corractive-rag",
    persist_directory="./.chromadb/corractive-rag",
    embedding_function=OpenAIEmbeddings()
).as_retriever()







































if __name__ == "__main__":
    print(doc_list)