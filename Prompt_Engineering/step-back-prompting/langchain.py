from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.tools import DuckDuckGoSearchResults
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# Few Shot Examples
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?"
    },
    {
        "input": "Jan Sindel’s was born in what country?", 
        "output": "what is Jan Sindel’s personal history?"
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{normal_context}
{step_back_context}

Original Question: {question}
Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)


system_prompt = """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    few_shot_prompt,
    ("user", "{question}"),
])


search = DuckDuckGoSearchAPIWrapper(max_results=4)

def retriever(query):
    return search.run(query)

def get_step_back_prompt_chain(llm):
    question_gen = prompt | llm | StrOutputParser()
    chain = {
        "normal_context": RunnableLambda(lambda x: x['question']) | retriever,
        "step_back_context": question_gen | retriever,
        "question": lambda x: x["question"]
        } | response_prompt | llm | StrOutputParser()
    return chain