from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field


llm = ChatOpenAI(
    temperature=0
)

class GradeAnswer(BaseModel):
    """
    Binary score for hallucination present in generated answer
    """

    binary_score : str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeAnswer) # Bu, LLM'nin çıktısını GradeAnswer modeline göre yapılandırmasını sağlar

system_prompt ="""
You are a grader assessing whether an answer addresses / resolves a question 
\n Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
