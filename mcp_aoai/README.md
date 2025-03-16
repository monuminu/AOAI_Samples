# Azure OpenAI MCP Integration

A Chainlit-based chat interface that connects to Azure OpenAI API with support for Microsoft Copilot Plugins (MCP).

## Features
- 🤖 Interactive chat interface powered by MCP
- 🔌 Integration with Azure OpenAI API
- 🔧 Tool calling functionality
- 🧩 Support for One Click Deployment
- 📊 Streaming responses with proper handling of function calls

## Prerequisites
- Azure OpenAI API key and endpoint
- Python 3.8 or higher

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install chainlit openai python-dotenv aiohttp
   ```
3. Create an `azure.env` file with the following variables:
   ```env
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_MODEL=your_deployment_name_here
   ```

## Usage

1. Start the Chainlit application:
   ```bash
   chainlit run app.py
   ```
2. Open your browser at `http://localhost:8000`
3. Connect MCP tools using the interface
4. Start chatting with the AI assistant

## How It Works

The application uses Chainlit's event system to manage chat sessions and tool interactions:

- `@cl.on_chat_start`: Initializes a new chat session
- `@cl.on_message`: Processes user messages and generates responses
- `@cl.on_mcp_connect`: Handles connection to Microsoft Copilot Plugins
- `@cl.step(type="tool")`: Manages tool call execution

## Configuration

You can modify the Chainlit configuration in the `config.toml` file. The current configuration has:

- Chain of Thought (CoT) display mode set to "full"
- Support for custom CSS and JavaScript
- Option to add custom header links

## Customization

To modify the welcome screen, edit the `chainlit.md` file at the root of your project.

## Learn More

- [Chainlit Documentation](https://docs.chainlit.io)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/cognitive-services/openai/)
