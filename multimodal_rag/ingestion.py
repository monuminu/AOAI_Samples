from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv('azure.env')

from langchain import hub
from langchain_openai import AzureChatOpenAI
#from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from doc_intelligence import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_core.documents import Document
from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights
)
# Split the document into chunks base on markdown headers.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),  
    ("#######", "Header 7"), 
    ("########", "Header 8")
]





import re
from typing import Any, List, Optional

from langchain_text_splitters.base import Language, TextSplitter

class CustomCharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(
        self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )
        splits = re.split(separator, text, flags=re.DOTALL) 
        splits = [part for part in splits if part.strip()]
        return splits

import re
import json
def find_figure_indices(text):
    pattern = r'!\[\]\(figures/(\d+)\)'
    matches = re.findall(pattern, text)
    indices = [int(match) for match in matches]
    return indices

llm = AzureChatOpenAI(api_key = os.environ["AZURE_OPENAI_API_KEY"],  
                      api_version = "2024-03-01-preview",
                      azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
                      model= "gpt-4-1106-preview",
                      streaming=False,
                      max_tokens = 1024)


aoai_embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2024-03-01-preview",
    azure_endpoint =os.environ["AZURE_OPENAI_ENDPOINT"]
)

embedding_function = aoai_embeddings.embed_query
fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=False,
    ),
    # Additional field to store the title
    SearchableField(
        name="header",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="image",
        type=SearchFieldDataType.String,
        filterable=False,
        searchable=False,
    ),
]



index_name: str = "langchain-vector-demo-custom2"

vector_store_multi_modal: AzureSearch = AzureSearch(
    azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    azure_search_key=os.environ["AZURE_SEARCH_KEY"],
    index_name=index_name,
    embedding_function=embedding_function,
    fields=fields,
)

import tempfile

def process(pdf_path):
    loader = AzureAIDocumentIntelligenceLoader(file_path=pdf_path, 
                                               api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"), 
                                               api_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
                                               api_model="prebuilt-layout",
                                               api_version="2024-02-29-preview",
                                               mode='markdown',
                                               analysis_features = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION])
    docs = loader.load()
    image_metadata = docs[-1].metadata['images']
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs_result = text_splitter.split_text(docs[0].page_content)
    text_splitter = CustomCharacterTextSplitter(separator=r'(<figure>.*?</figure>)', is_separator_regex=True)
    child_docs  = text_splitter.split_documents(docs_result)
    lst_docs = []
    for doc in child_docs:
        figure_indices = find_figure_indices(doc.page_content)
        if figure_indices:
            for figure_indice in figure_indices:
                image = image_metadata[figure_indice]
                doc_result = Document(page_content = doc.page_content, metadata={"header": json.dumps(doc.metadata), "source": "sam.pdf", "image": image})
                lst_docs.append(doc_result)
        else:
            doc_result = Document(page_content = doc.page_content, metadata={"header": json.dumps(doc.metadata), "source": "sam.pdf", "image": None})
            lst_docs.append(doc_result)
    vector_store_multi_modal.add_documents(documents=lst_docs)



import streamlit as st

# Title of the web application
st.title('File Upload and Process Example')

# File uploader widget
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the file path
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Write the uploaded PDF file to the temporary directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        process(file_path)
        st.success(f'File "{uploaded_file.name}" saved in temporary directory.')