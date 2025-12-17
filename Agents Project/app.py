from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
search = TavilySearchResults(max_results=2)
tools = [search]

# Memory oluştur
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Prompt oluştur (chat_history eklendi)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent oluştur
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

if __name__ == '__main__':
    print("Chat started! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        result = agent_executor.invoke({"input": user_input})
        print(f"\nAssistant: {result['output']}\n")