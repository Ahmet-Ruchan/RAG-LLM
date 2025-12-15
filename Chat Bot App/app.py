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

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0)
#model = ChatOpenAI(model="gpt-4o", temperature=0.1) # Initialize the ChatOpenAI model

# messages = [
#     HumanMessage(content="Hello my name is John."),
#     AIMessage(content="Hello Jhon, how can I assist you today?"), # Now AI know your name is John
#     HumanMessage(content="What is my name?")
# ]

# response = model.invoke(messages)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory() # Create a new in-memory chat history for the session

    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="messages") # Placeholder for chat history
    ]
)

chain = prompt | model
config = {"configurable" : {"session_id" : "xyz123"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)


if __name__ == '__main__':

    #print(model.invoke([HumanMessage(content="Hello my name is John.")]).content) # Now AI don't know your name is John
    #print(response.content) # Now AI know your name is John

    while True:
        user_input = input(">")

        # Bu response u stream oalrak kelime kelime de alabiliriz.

        for r in with_message_history.stream(
            [
                HumanMessage(content=user_input),
            ],
            config=config
        ):
            print(r.content, end='\n') # flush=True ile anında ekrana yazdırıyoruz.

        '''
        response = with_message_history.invoke(
            [
                HumanMessage(content=user_input)
            ],
            config=config,
        )

        print(response.content)
        '''