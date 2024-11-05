Title: Realtime Assistant
Tags: [multimodal, audio]

# From Zero to Hero: Building Your First Voice Bot with GPT-4o Real-Time API using Python

Voice technology is transforming how we interact with machines, making conversations with AI feel more natural than ever before. With the public beta release of the Realtime API powered by GPT-4o, developers now have the tools to create low-latency, multimodal voice experiences in their apps, opening up endless possibilities for innovation.

Gone are the days when building a voice bot required stitching together multiple models for transcription, inference, and text-to-speech conversion. With the Realtime API, developers can now streamline the entire process with a single API call, enabling fluid, natural speech-to-speech conversations. This is a game-changer for industries like customer support, education, and real-time language translation, where fast, seamless interactions are crucial.

## Key Features

- **Realtime Python Client**: Based off https://github.com/openai/openai-realtime-api-beta
- **Multimodal experience**: Speak and write to the assistant at the same time
- **Tool calling**: Ask the assistant to perform tasks and see their output in the UI
- **Visual Presence**: Visual cues indicating if the assistant is listening or speaking

Plead read my Blog for more details https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/from-zero-to-hero-building-your-first-voice-bot-with-gpt-4o-real/ba-p/4269038


The following files are also included in the repository:
- requirements.txt: Lists the required Python packages.
- Dockerfile: Used to build a Docker image for the application.
- .env: Contains the environment variables.
- build-docker-image.sh: A script to build the Docker image.
- run-docker-image.sh: A script to run the Docker image locally.
- push-docker-image.sh: A script to push the Docker image to an Azure Container Registry
- variables.sh: contains the variables for the Azure Container Registry, and the Docker image.

## Quickstart

### Prerequisites:
- An active [Azure Subscription](https://learn.microsoft.com/en-us/azure/guides/developer/azure-developer-guide#understanding-accounts-subscriptions-and-billing). If you don't have one, create a [free Azure account](https://azure.microsoft.com/en-gb/free/) before you begin.
- [VS Code](https://code.visualstudio.com/) as a code editor.
- [Docker](https://www.docker.com/) installed on your local machine.
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed on your local machine.
- [Azure OpenAI account](https://azure.microsoft.com/en-us/services/cognitive-services/openai/). You will need to create a resource and obtain your OpenAI Endpoint, API Key, deploy text-embedding-ada-002 and gpt-35-turbo-16k model.
- Python 3.11 or higher installed on your local machine.
- (Optional) [Azure Container Registry](https://docs.microsoft.com/en-us/azure/container-registry/) to store the Docker image. This step is optional, if you want to deploy the application to Azure Container Apps for example.

### Setup the environment variables

1. Create an .env file and update the following environment variables:

    ```
        AZURE_OPENAI_API_KEY=XXXX
        # replace with your Azure OpenAI API Key

        AZURE_OPENAI_ENDPOINT=wss://xxxx.openai.azure.com/
        # replace with your Azure OpenAI Endpoint

        AZURE_OPENAI_DEPLOYMENT=gpt-4o-realtime-preview
        #Create a deployment for the gpt-4o-realtime-preview model and place the deployment name here. You can name the deployment as per your choice and put the name here.

        AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION=2024-10-01-preview
        #You don't need to change this unless you are willing to try other versions.
    ```

Once you have updated the .env file, please save the changes and you are ready to proceed to the next step.

### Option 1: Run the application locally

1. Install dependencies: 
Open the terminal and navigate to the src folder of the repository. Then run the following command to install the necesairly Python packages:

    ```pip
    pip install -r requirements.txt
    ```

2. Run the application: Run the following command to start the application:

    ```chainlit
     chainlit run app.py -w
    ```
3. Test the application: Open a new terminal and run the following command to test the application:

    ```chainlit
     http://localhost:8000/
    ```

### Option 2: Run the application in a Docker container

1. Navigate to the src folder of the repository

2. Open the file build-docker-image.sh and depending on the architecture of your local machine (linux/arm64 or linux/amd64), uncomment the respective line and comment the other line. Then save the file. In my case I built the image to run it locally on my M1 Mac, so I have uncommented the line for linux/arm64 and commented the line for linux/amd64. If you plan to build the image for a different architecture, you can uncomment the respective line and comment the other line.

3. Run the following command to build the Docker image:

    ```build-docker-image
     ./build-docker-image.sh
    ```
4. Run the following command to run the Docker image:

    ```run-docker-image
     ./run-docker-image.sh
    ```
5. Test the application: Open a new terminal and run the following command to test the application:

    ```chainlit
     http://localhost:8000/
    ```

6. (optional) Push the Docker image to an Azure Container Registry

    If you want to deploy the application to Azure, you can push the Docker image to an Azure Container Registry. To do this, you need to have an Azure Container Registry and the Docker image name and the Azure Container Registry name in the variables.sh file. Once you have updated the variables.sh file, run the following Azure CLI command to connect to your Azure Subscription:
    
    ```azure
    az login
    ```

    Then run the following command to push the Docker image to the Azure Container Registry:

    ```push-docker-image
    ./push-docker-image.sh
    ```
