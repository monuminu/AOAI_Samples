# Azure AI Agents FastAPI Backend

A FastAPI backend service that exposes Azure AI Agents functionality with real-time streaming support for web applications.

## Features

- 🚀 **FastAPI Backend**: High-performance async web framework
- 📡 **Real-time Streaming**: Server-Sent Events (SSE) for live progress updates
- 🤖 **Azure AI Agents**: Integrated with Azure AI Project Client
- 🌐 **Browser Automation**: Playwright integration for web scraping tasks
- 🔄 **Dual Endpoints**: Both streaming and non-streaming query options
- 🛡️ **Error Handling**: Comprehensive error handling and validation
- 📚 **Auto Documentation**: Built-in Swagger/OpenAPI documentation

## Prerequisites

- Python 3.8+
- Azure subscription with AI Services
- Azure CLI installed and authenticated
- Playwright connection configured in your Azure AI Project

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd azure-agents-fastapi
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn azure-ai-agents azure-ai-projects azure-identity python-dotenv
   ```

3. **Set up environment variables**
   
   Create an `azure.env` file in the project root:
   ```env
   PROJECT_ENDPOINT=https://your-project-endpoint.cognitiveservices.azure.com/
   # Add any other required Azure configuration
   ```

4. **Authenticate with Azure CLI**
   ```bash
   az login
   ```

## Quick Start

1. **Start the server**
   ```bash
   python app.py
   ```
   
   The server will start on `http://localhost:8000`

2. **View API Documentation**
   
   Open your browser and navigate to:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

3. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"message": "What is the weather in Pune?"}'
   ```

## API Endpoints

### POST `/query`
Execute an agent query and return the complete result.

**Request Body:**
```json
{
    "message": "Your query message",
    "model": "gpt-4.1",
    "agent_name": "playwright-agent",
    "instructions": "use the tool to respond"
}
```

**Response:**
```json
{
    "run_id": "run_123",
    "status": "completed",
    "agent_id": "agent_456",
    "thread_id": "thread_789",
    "steps": [...],
    "response": {
        "text_messages": ["The temperature in Pune is..."],
        "citations": [...]
    }
}
```

### POST `/query/stream`
Execute an agent query with real-time streaming of intermediate events.

**Request Body:** Same as `/query`

**Response:** Server-Sent Events stream with the following event types:
- `connection_setup` - Setting up Azure connections
- `agent_created` - Agent creation completed
- `thread_created` - Conversation thread created
- `message_created` - User message added to thread
- `run_created` - Agent run initiated
- `step_processed` - Individual execution step completed
- `response_ready` - Final response available
- `completed` - Full execution completed

### GET `/health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-08-11T10:30:00.000Z"
}
```

### GET `/`
Root endpoint with API information.

## Usage Examples

### JavaScript (Streaming)
```javascript
async function streamQuery(message) {
    const response = await fetch('http://localhost:8000/query/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const eventData = JSON.parse(line.slice(6));
                console.log(`${eventData.event_type}:`, eventData.data);
            }
        }
    }
}

// Usage
streamQuery("What's the temperature in Mumbai?");
```

### Python (Non-streaming)
```python
import requests

def query_agent(message):
    response = requests.post(
        'http://localhost:8000/query',
        json={'message': message}
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['response']['text_messages'][0]
    else:
        raise Exception(f"API Error: {response.text}")

# Usage
answer = query_agent("What's the current time in New York?")
print(answer)
```

### cURL (Non-streaming)
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What is the population of Tokyo?",
       "model": "gpt-4.1"
     }'
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PROJECT_ENDPOINT` | Azure AI Project endpoint URL | Yes |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | string | - | The query message (required) |
| `model` | string | "gpt-4.1" | Azure OpenAI model to use |
| `agent_name` | string | "playwright-agent" | Name for the created agent |
| `instructions` | string | "use the tool to respond" | Instructions for the agent |

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request (e.g., empty message)
- `500` - Internal Server Error (Azure connection issues, etc.)

Error responses include detailed error messages:
```json
{
    "detail": "Failed to create project client: Authentication failed"
}
```

## Development

### Running in Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Support
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Ensure Azure CLI is logged in: `az login`
   - Verify your Azure subscription has the required permissions

2. **Playwright Connection Not Found**
   - Check that "playwright-eastus" connection exists in your Azure AI Project
   - Update the connection name in the code if different

3. **Model Not Available**
   - Verify the specified model (default: "gpt-4.1") is deployed in your Azure OpenAI resource

4. **CORS Issues**
   - Adjust the CORS settings in the FastAPI app for your domain
   - For production, replace `allow_origins=["*"]` with specific domains

### Logging

Enable detailed logging by adding:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in this repository
- Check the Azure AI Services documentation
- Review the FastAPI documentation

---

**Note**: This is a development example. For production use, implement proper security, authentication, rate limiting, and monitoring.
