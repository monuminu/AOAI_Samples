import azure.cognitiveservices.speech as speechsdk
import datetime
import io
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import openai
import os
import random
import requests
import sys
import time

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
)
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.cognitiveservices.speech import (
    AudioDataStream,
    SpeechConfig,
    SpeechSynthesizer,
    SpeechSynthesisOutputFormat,
)
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from azure.search.documents.models import VectorizedQuery,VectorizableTextQuery

from dotenv import load_dotenv
from io import BytesIO
from IPython.display import Audio
from PIL import Image
import os
import base64
import re
from datetime import datetime, timedelta

import requests
import os
from tenacity import (
    Retrying,
    retry_if_exception_type,
    wait_random_exponential,
    stop_after_attempt
)
import json
import mimetypes

params = {  
        "api-version": "2023-02-01-preview",
        "overload": "stream",
        "modelVersion": "latest"
}


load_dotenv("azure.env")
# Azure Open AI
openai_api_type = os.getenv("azure")
openai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_version = os.getenv("AZURE_API_VERSION")
openai_api_key = os.getenv("AZURE_OPENAI_KEY")

# Azure Cognitive Search
acs_endpoint = os.getenv("ACS_ENDPOINT")
acs_key = os.getenv("ACS_KEY")

# Azure Computer Vision 4
acv_key = os.getenv("ACV_KEY")
acv_endpoint = os.getenv("ACV_ENDPOINT")

blob_connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = os.getenv("CONTAINER_NAME")

# Azure Cognitive Search index name to create
index_name = "azure-fashion-demo"

# Azure Cognitive Search api version
api_version = "2023-02-01-preview"


def text_embedding(prompt):
    """
    Text embedding using Azure Computer Vision 4.0
    """
    version = "?api-version=" + api_version + "&modelVersion=latest"
    vec_txt_url = f"{acv_endpoint}/computervision/retrieval:vectorizeText{version}"
    headers = {"Content-type": "application/json", "Ocp-Apim-Subscription-Key": acv_key}
    payload = {"text": prompt}
    response = requests.post(vec_txt_url, json=payload, headers=headers)

    if response.status_code == 200:
        text_emb = response.json().get("vector")
        return text_emb
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def image_embedding(image_path):
    url = f"{acv_endpoint}/computervision/retrieval:vectorizeImage"  
    mime_type, _ = mimetypes.guess_type(image_path)
    headers = {  
        "Content-Type": mime_type,
        "Ocp-Apim-Subscription-Key": acv_key  
    }
    for attempt in Retrying(
        retry=retry_if_exception_type(requests.HTTPError),
        wait=wait_random_exponential(min=15, max=60),
        stop=stop_after_attempt(15)
    ):
        with attempt:
            with open(image_path, 'rb') as image_data:
                response = requests.post(url, params=params, headers=headers, data=image_data)  
                if response.status_code != 200:  
                    response.raise_for_status()
    vector = response.json()["vector"]
    return vector

def get_translation(text, lang, max_retries=3, retry_delay=1):
    """
    Text translation using Azure Open AI
    """
    sentence = f"Translate the following text to {lang}: {text}"

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Should be deployed in the AOAI studio
            prompt=sentence,
            temperature=0.1,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            stop=None,
        )
        resp = response["choices"][0]["text"]

        return resp

    except Exception as e:
        print("Error:", str(e))
        
        
def index_stats(index_name):
    """
    Get statistics about Azure Cognitive Search index
    """
    url = (
        acs_endpoint
        + "/indexes/"
        + index_name
        + "/stats?api-version=2021-04-30-Preview"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": acs_key,
    }
    response = requests.get(url, headers=headers)
    print("Azure Cognitive Search index status for:", index_name, "\n")

    if response.status_code == 200:
        res = response.json()
        print(json.dumps(res, indent=2))

    else:
        print("Request failed with status code:", response.status_code)
        
def index_status(index_name):
    """
    Azure Cognitive Search index status
    """
    print("Azure Cognitive Search Index:", index_name, "\n")

    headers = {"Content-Type": "application/json", "api-key": acs_key}
    params = {"api-version": "2021-04-30-Preview"}
    index_status = requests.get(
        acs_endpoint + "/indexes/" + index_name, headers=headers, params=params
    )

    try:
        print(json.dumps((index_status.json()), indent=5))
    except Exception as e:
        print("Error:", str(e))
        
        
def visual_search(clothes_list):
    """
    Displaying search results
    """
    idx = 1

    for item in clothes_list:
        print(f"Item {idx}: {item}")
        images = prompt_search(item, topn=5)
        display_images(images, disp_cosine=False, source="prompt")
        idx += 1

    print("\033[1;31;35m")
    print(
        datetime.datetime.today().strftime("%d-%b-%Y %H:%M:%S"),
        " End of search - Powered by Azure AI",
    )
    
def display_images(images_list, num_cols=5, disp_cosine=False, source="prompt"):
    """
    Display multiple images
    """
    num_images = len(images_list)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))

    if disp_cosine and source == "prompt":
        cos_list = [
            get_cosine_similarity(text_embedding(prompt), image_embedding(image))
            for image in images_list
        ]

    if disp_cosine and source == "image":
        cos_list = [
            get_cosine_similarity(
                url_image_embedding(image_url), image_embedding(image)
            )
            for image in images_list
        ]

    for idx, ax in enumerate(axes.flat):
        if idx < num_images:
            image_path = images_list[idx]+ ".jpg"
            blob_client = container_client.get_blob_client(image_path)
            blob_image = blob_client.download_blob().readall()
            ax.axis("off")
            # Display image
            image = Image.open(io.BytesIO(blob_image))
            ax.imshow(image)
            topnid = idx + 1
            title = f"Top {topnid:02}\n{images_list[idx]}"

            if disp_cosine:
                title += f"\nCosine similarity = {round(cos_list[idx], 5)}"
            ax.set_title(title)

        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    
def get_list(mylist):
    """
    Get list of items to buy
    """
    pattern = r"\d+\.\s(.+)"
    matches = re.findall(pattern, mylist)
    clothes_list = list(matches)
    idx = 1

    for item in clothes_list:
        print(f"Item {idx}: {item}.")
        idx += 1

    return clothes_list



def prompt_search(prompt, topn=5, disp=False):
    """
    Azure Cognitive visual search using a prompt
    """
    results_list = []
    # Initialize the Azure Cognitive Search client
    search_client = SearchClient(acs_endpoint, index_name, AzureKeyCredential(acs_key))
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    # Perform vector search
    vector_query = VectorizedQuery(vector=text_embedding(prompt), k_nearest_neighbors=topn, fields="image_vector")
    response = search_client.search(
        search_text=prompt, vector_queries= [vector_query], select=["description"], top = 2
    )    
    for nb, result in enumerate(response, 1):
        blob_name = result["description"] + ".jpg"
        blob_client = container_client.get_blob_client(blob_name)
        image_url = blob_client.url
        sas_token = generate_blob_sas(
                                        blob_service_client.account_name,
                                        container_name,
                                        blob_name,
                                        account_key=blob_client.credential.account_key,
                                        permission=BlobSasPermissions(read=True),
                                        expiry=datetime.utcnow() + timedelta(hours=1)
                                    )
        sas_url = blob_client.url + "?" + sas_token
        results_list.append({"buy_now_link" : sas_url,"price_of_the_product": result["description"], "product_image_url": sas_url})
    return results_list


def download_and_convert_to_base64(url):
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(url.split("/")[-1])
    blob_image = blob_client.download_blob().readall()
    base64_image = base64.b64encode(blob_image).decode('utf-8')
    return "data:image/png;base64," + base64_image

def replace_urls_with_base64_images(text):
    url_pattern = re.compile(r'https?://[^\s)]+')
    urls = url_pattern.findall(text)
    for url in urls:
        try:
            base64_image = download_and_convert_to_base64(url)
            text = text.replace(url, base64_image)
        except Exception as e:
            print(f"Failed to process URL {url}: {e}")
    return text