from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.schema import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import PostgresChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from doc_intelligence import AzureAIDocumentIntelligenceLoader


import os
import uuid
import chainlit as cl
from pathlib import Path
from typing import List
from operator import itemgetter

chunk_size = 1024
chunk_overlap = 50
PDF_STORAGE_PATH = "./pdfs"

# Load environment variables
from dotenv import load_dotenv
load_dotenv('azure.env')


import io
import re

from IPython.display import HTML, display
from PIL import Image
import base64
import os

from uuid import uuid4
from typing import Optional


llm = AzureChatOpenAI(api_key = os.environ["AZURE_OPENAI_API_KEY"],  
                      api_version = "2023-12-01-preview",
                      azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
                      model= "gpt-4-1106-preview",
                      streaming=True)

aoai_embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-12-01-preview",
    azure_endpoint =os.environ["AZURE_OPENAI_ENDPOINT"]
)

embedding_function = aoai_embeddings.embed_query
# Return the retrieved documents or certain source metadata from the documents
import re
def get_clean_text(text, pattern):
    pattern = re.compile(pattern, re.M)
    clean_text = re.sub(pattern, '', text)
    return clean_text

prompt = hub.pull("rlm/rag-prompt")

def get_image_text(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if doc.metadata['image']:
            b64_images.append(doc.metadata['image'])
        else:
            texts.append(doc.page_content)
    return {"images": b64_images, "texts": texts}

matched_str = ["<!-- Footnote=", "<!-- PageFooter=", "<!-- PageNumber="]
def format_data(data_dict):
    formatted_texts = "\n".join([item for item in data_dict["context"]["texts"]])
    for pattern in matched_str:
        formatted_texts = get_clean_text(formatted_texts, pattern)
    return formatted_texts
    
def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = format_data(data_dict)
    messages = []
    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
            f"User-provided question: {data_dict['question']}\n\n"
            f"Context: {formatted_texts}"
        ),
    }
    messages.append(text_message)
    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"{image}"},
            }
            messages.append(image_message)
    return [HumanMessage(content=messages)]

welcome_message = 'Hi, Welcome to Multimodal Chatbot!!'
index_name: str = "langchain-vector-demo-custom2"
vector_store_multi_modal: AzureSearch = AzureSearch(
    azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    azure_search_key=os.environ["AZURE_SEARCH_KEY"],
    index_name=index_name,
    embedding_function=embedding_function,
)
retriever_multi_modal = vector_store_multi_modal.as_retriever(search_type="similarity",search_kwargs = {"k" : 5})



@cl.on_chat_start
async def start():
    identifier = "admin"
    memory = ConversationBufferMemory(return_messages=True)
    message_history = PostgresChatMessageHistory(
        connection_string=os.environ["POSTGRES_CONNECTION_STRING"],
        session_id= identifier + str(uuid4()),
    )
    chain_multimodal_rag_with_source = RunnableMap(
            {
                "documents": retriever_multi_modal , 
                "question": RunnablePassthrough()
            }
        ) | {
            "source_documents": lambda input: [doc.metadata for doc in input["documents"]],
            "answer": RunnableLambda(lambda input: {"context": get_image_text(input["documents"]), "question": input["question"]}) 
          | RunnableLambda(img_prompt_func) 
          | llm 
          | StrOutputParser()
        }
    cl.user_session.set("chain", chain_multimodal_rag_with_source)
    cl.user_session.set("message_history", message_history)


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    llm_chain = cl.user_session.get("chain")
    message_history = cl.user_session.get("message_history")
    inputs = {"input": message.content, "history":message_history.messages}
    config = RunnableConfig(
        callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]
    )
    res = await llm_chain.ainvoke(message.content, config = config)
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    elements = []
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            if source_doc.get("image"):
                base64_data = source_doc.get('image').split(',')[1]
                elements.append(cl.Image(content = base64.b64decode(base64_data), name = source_name))
            else:
                elements.append(
                    cl.Text(content=source_doc.get("header"), name=source_name)
                )
        source_names = [text_el.name for text_el in elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=elements).send()