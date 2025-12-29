from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv() # Load environment variables from .env file

#model_openai = ChatOpenAI(model="gpt-4o", temperature=0.1) # Initialize the ChatOpenAI model)
model_gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.1)

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to Turkish."),
    HumanMessage(content="Hello, how are you?")
]


if __name__ == "__main__":
    #response_openai = model_openai.invoke(messages) # Get the model's response to the messages
    response_gemini = model_gemini.invoke(messages)

    print("\nOpenAI Response:")
    #print(response_openai , "\n")
    print("*" * 50)
    print("\nGemini Response:")
    print(response_gemini.content , "\n")

