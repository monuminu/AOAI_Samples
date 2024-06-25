This workbook contains 3 main parts:
- **Overview** - Holistic view of Azure OpenAI resources
- **Monitor** - Holistic view of Azure OpenAI resources Metrics
- **Insights** - Holistic view of Azure OpenAI resources Logs
  - Requires by enabling Diagnostic Settings to Log Analytics Workspace.

### Views

Types of views this workbook provides:

- **Overview**
  - Azure OpenAI Resources by
    - SubscriptionId
    - Resource Group
    - Location
    - Kind
    - Public Network Access
    - Private Network Access
  - All Azure OpenAI Resources

> The information displayed uses [KQL](https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/) queries to query the [Azure Resource Graph](https://learn.microsoft.com/en-us/azure/governance/resource-graph/overview).

- **Monitor**
  - Overview
    - Requests
    - Processed Inference Tokens
    - Processed Prompt Tokens
    - Generated Completions Tokens
    - Processed FineTuned Training Hours
    - Provisioned-managed Utilization
  - HTTP Requests
    - by Model Name
    - by Model Version
    - by Model Deployment Name
    - by Status Code
    - by StreamType
    - by Operation Name
    - by API Name
    - by Region
  - Token-Based Usage
    - Processed Inference Tokens
      - by Model Name
      - by Model Deployment Name
    - Processed Prompt Tokens
      - by Model Name
      - by Model Deployment Name
    - Generate Completitions Tokens
      - by Model Name
      - by Model Deployment Name
    - Active Tokens
      - by Model Name
      - by Model Deployment Name
  - PTU Utilization
    - Provisioned-managed Utilization
      - by Model Name
      - Model Version
      - by Model Deployment Name
      - by StreamType
  - Fine-tuning
    - Processed FineTuned Training Hours
      - by Model Name
      - by Model Deployment Name

> The information displayed uses Azure OpenAI Platform Metrics and presented by multiple dimensions.

- **Insights**
  - Overview
    - Requests
      - by Resource
      - by Location
      - by StreamType
      - by Api Version
      - by Model Deployment Name + Operation Name
      - by Model Deployment Name
      - by Model Name + Operation Name
      - by Model Name
      - by Operation Name
      - by Avg Duration (ms)
      - by Avg Request Length (bytes)
      - by Avg Response Length (bytes)
  - By CallerIP
    - Requests
    - Operation Name
    - Model Deployment Name + Operation Name
    - Model Name + Operation Name
    - Avg Duration (ms)
    - Avg Request Length (bytes)
    - Avg Response Length (bytes)
  - All Logs
    - Successful requests
  - Failures
    - Failed requests
      - by Resources
      - by Api Version
      - by Operation name
      - by Stream Type