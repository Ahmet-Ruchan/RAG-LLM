from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

llm = ChatOpenAI(
    temperature=0
)

prompt = hub.pull("rlm/rag-prompt") # You can also create your own prompt and push it to LangChain Hub

generation_chain = prompt | llm | StrOutputParser()