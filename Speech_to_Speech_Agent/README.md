Here is an updated README that includes the environment variables needed for the `Speech_to_Speech_Agent`:

---

# Speech to Speech Agent

This project is part of the `AOAI_Samples` repository, demonstrating a Speech to Speech Agent using Azure OpenAI services.

## Overview

The Speech to Speech Agent is designed to convert spoken language input into spoken language output, leveraging Azure OpenAI's capabilities. This agent can be used in various applications, including virtual assistants, customer service bots, and interactive voice response systems.

## Folder Structure

- `app.py`: The main application file for the Speech to Speech Agent.
- `requirements.txt`: Contains the list of dependencies required to run the application.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Azure account with OpenAI services enabled

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/monuminu/AOAI_Samples.git
    cd AOAI_Samples/Speech_to_Speech_Agent
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

The following environment variables need to be set for the application to function properly. You can set these variables in your operating system or use a `.env` file.

- `OPEN_AI_KEY`: Your Azure OpenAI API key
- `OPEN_AI_ENDPOINT`: Your Azure OpenAI endpoint, e.g., `https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/`
- `OPEN_AI_DEPLOYMENT_NAME`: The deployment name of your Azure OpenAI model
- `SPEECH_KEY`: Your Azure Cognitive Services Speech API key
- `SPEECH_REGION`: The region of your Azure Cognitive Services Speech API

You can set these variables in your terminal session as follows:
```bash
export OPEN_AI_KEY="your_open_ai_key"
export OPEN_AI_ENDPOINT="your_open_ai_endpoint"
export OPEN_AI_DEPLOYMENT_NAME="your_open_ai_deployment_name"
export SPEECH_KEY="your_speech_key"
export SPEECH_REGION="your_speech_region"
```

Alternatively, you can create a `.env` file in the project directory with the following content:
```
OPEN_AI_KEY=your_open_ai_key
OPEN_AI_ENDPOINT=your_open_ai_endpoint
OPEN_AI_DEPLOYMENT_NAME=your_open_ai_deployment_name
SPEECH_KEY=your_speech_key
SPEECH_REGION=your_speech_region
```

### Running the Application

Execute the following command to start the Speech to Speech Agent:
```bash
python app.py
```

## Usage

Once the application is running, it will listen for spoken input, process it using Azure OpenAI services, and provide a spoken response.

## Contribution

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

---

For more information, please refer to the [official Azure OpenAI documentation](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/) or the [GitHub repository](https://github.com/monuminu/AOAI_Samples).

---
