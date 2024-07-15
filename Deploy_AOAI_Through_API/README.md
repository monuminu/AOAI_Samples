# README.md

## Azure OpenAI Model Deployment Script

This script automates the deployment of an OpenAI model in Azure. Follow the steps below to set up and run the script.

### Prerequisites

- Python 3.x
- Azure CLI
- Azure subscription
- Azure Cognitive Services resource

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install required Python libraries:**
   ```bash
   pip install requests azure-identity
   ```

### Configuration

1. **Add your Azure OpenAI account details:**

   Update the following variables in the script with your Azure account details:

   ```python
   subscription = "<AOAI-ACCOUNT-SUBSCRIPTION-ID>"
   resource_group = "<AOAI-ACCOUNT-RESOURCE-GROUP>"
   resource_name = "<AOAI-RESOURCE-NAME>"
   model_deployment_name = "<NEW-AOAI-DEPLOYMENT-NAME>"
   registered_model_name = 'gpt-4o'
   registered_model_version = '2024-05-13'
   ```

### Usage

1. **Run the script:**

   ```bash
   python deploy_model.py
   ```

### Script Explanation

1. **Import Libraries:**

   The script begins by importing necessary libraries.

   ```python
   import requests
   import json
   import subprocess
   from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
   ```

2. **Azure Credentials:**

   The script attempts to get an access token using `DefaultAzureCredential`. If it fails, it falls back to `InteractiveBrowserCredential`.

   ```python
   try:
       credential = DefaultAzureCredential()
       access_token = credential.get_token("https://management.azure.com/.default")
       token = access_token.token
   except Exception as ex:
       credential = InteractiveBrowserCredential()
   ```

3. **Deployment Configuration:**

   The deployment configuration is set up, including the model details and SKU capacity.

   ```python
   deploy_data = {
       "displayName": model_deployment_name,
       "properties": {
           "model": {
               "format": "OpenAI",
               "name": registered_model_name,
               "version": registered_model_version
           },
           "versionUpgradeOption": "OnceNewDefaultVersionAvailable",
           "dynamicThrottlingEnabled": "true",
           "raiPolicyName": "Microsoft.Default"
       },
       "sku": {
           "name": "Standard",
           "capacity": 150
       }
   }
   ```

4. **Deploy the Model:**

   The script sends a PUT request to deploy the model and prints the response.

   ```python
   request_url = f"https://management.azure.com//subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}"
   r = requests.put(   
       request_url, 
       params={"api-version": "2024-04-01-preview"}, 
       headers={
           "Authorization": f"Bearer {token}",
           "Content-Type": "application/json",
       }, 
       data=json.dumps(deploy_data)
   )

   print(r.json())
   ```

### Notes

- Ensure you have the necessary permissions to deploy models in your Azure subscription.
- For more details on Azure OpenAI and deployment options, refer to the [official documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/).

### License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to customize the README.md as per your project requirements.