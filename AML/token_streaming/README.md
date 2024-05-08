# Deploy a Llama-2-7b-chat model and stream the output results

1. Create a custom environment 
    az ml environment create --name streaming-env --version 1 --file environment.yaml  --image hkotesova/streaming:latest --resource-group <resource-group-name> --workspace-name <workspace-name>

2. Create a custom deployment using the custom environment
    az ml online-deployment create -f online_deployment.yaml

3. install openai
    pip install openai==0.28.1

3. Use OpenAI APIs to stream output response
    python example.py