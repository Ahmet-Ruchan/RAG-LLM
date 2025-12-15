from dotenv import load_dotenv
from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

load_dotenv() # Load environment variables from .env file

#model_openai = ChatOpenAI(model="gpt-4o", temperature=0.1) # Initialize the ChatOpenAI model)
model_gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

system_prompt = "Translate the following text to {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{text}")
    ]
)

parser = StrOutputParser()

chain = prompt_template | model_gemini | parser # prompt_template al -> model_gemini ver, çıktıyı al -> parser'a ver -> parser'ın çıktısını al

app = FastAPI(
    title="Gemini Response",
    description="An API to get responses from Gemini model",
    version="1.0.0"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":

    uvicorn.run(app, host="localhost", port=8000)