from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() # Load environment variables from .env file

#model_openai = ChatOpenAI(model="gpt-4o", temperature=0.1) # Initialize the ChatOpenAI model)
model_gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.1)

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to Turkish."),
    HumanMessage(content="Hello, how are you?")
]

parser = StrOutputParser()
#response = model_gemini.invoke(messages)
# Aşağıdaki satırda modelin cevabını alıp ardından parse ediyoruz yani ayrıştırıyoruz. Bu sayede cevabın sadece metin kısmını elde ediyoruz.
# Ve bu response değişkenini kullanmamış oluyoruz.

chain = model_gemini | parser # Create a chain that first gets the model response and then parses it
# Chain oluşturuyoruz. İlk olarak modelin cevabını alıyor, ardından parser ile ayrıştırıyoruz. Bunu zincirleme bir şekilde yapıyoruz.

if __name__ == "__main__":

    print("*" * 50)
    print("\nGemini Response:")
    print(chain.invoke(messages) , "\n")