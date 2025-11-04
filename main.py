from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
Sen sana sorulan SUT (Sağlık Uygulama Tebliği) kuralları hakkında bir uzmanın cevabını verirsin.

İlgili SUT kuralları: {context}

Yukarıdaki kurallara göre aşağıdaki soruya cevap ver: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    context = retriever.invoke(question)
    result = chain.invoke({"context": context, "question": question})
    print(result)