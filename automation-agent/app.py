import os
import asyncio
import json
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import MessageRole
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv('azure.env', override=True)

app = FastAPI(title="Azure AI Agents API", version="1.0.0")

# Mount static files (for CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str
    model: str = "gpt-4.1"
    agent_name: str = "playwright-agent"
    instructions: str = "use the tool to respond"

class StreamEvent(BaseModel):
    timestamp: str
    event_type: str
    data: Dict[Any, Any]

def create_project_client():
    """Create and return Azure AI Project Client"""
    try:
        return AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=AzureCliCredential(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project client: {str(e)}")

async def stream_agent_events(
    project_client: AIProjectClient,
    query: str,
    model: str,
    agent_name: str,
    instructions: str
) -> AsyncGenerator[str, None]:
    """Stream events from the agent execution process"""
    
    def create_event(event_type: str, data: Dict[Any, Any]) -> str:
        event = StreamEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            data=data
        )
        return f"data: {event.json()}\n\n"
    
    try:
        # Get playwright connection
        yield create_event("connection_setup", {"status": "getting playwright connection"})
        playwright_connection = project_client.connections.get(name="playwright-eastus")
        yield create_event("connection_ready", {"connection_id": playwright_connection.id})
        
        # Create agent
        yield create_event("agent_creation", {"status": "creating agent"})
        agent = project_client.agents.create_agent(
            model=model,
            name=agent_name,
            instructions=instructions,
            tools=[{
                "type": "browser_automation",
                "browser_automation": {
                    "connection": {
                        "id": playwright_connection.id,
                    }
                }
            }],
        )
        yield create_event("agent_created", {"agent_id": agent.id, "name": agent_name})
        
        # Create thread
        yield create_event("thread_creation", {"status": "creating thread"})
        thread = project_client.agents.threads.create()
        yield create_event("thread_created", {"thread_id": thread.id})
        
        # Create message
        yield create_event("message_creation", {"status": "creating message", "content": query})
        message = project_client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )
        yield create_event("message_created", {"message_id": message['id'], "content": query})
        
        # Create and process run
        yield create_event("run_creation", {"status": "creating and processing run"})
        run = project_client.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
        )
        yield create_event("run_created", {"run_id": run.id})
        
        # Monitor run progress (you might want to poll here for real-time updates)
        yield create_event("run_processing", {"status": "processing", "run_id": run.id})
        
        # Wait for completion (in a real implementation, you might want to poll the status)
        await asyncio.sleep(1)  # Brief pause to allow processing
        
        yield create_event("run_completed", {"status": run.status, "run_id": run.id})
        
        if run.status == "failed":
            yield create_event("run_error", {"error": str(run.last_error)})
        
        # Get run steps
        yield create_event("steps_retrieval", {"status": "retrieving run steps"})
        run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
        
        steps_data = []
        for step in run_steps:
            step_info = {
                "step_id": step['id'],
                "status": step['status'],
                "tool_calls": []
            }
            
            # Check for tool calls
            step_details = step.get("step_details", {})
            tool_calls = step_details.get("tool_calls", [])
            
            for call in tool_calls:
                tool_call_info = {
                    "call_id": call.get('id'),
                    "type": call.get('type'),
                    "function_name": call.get("function", {}).get('name')
                }
                step_info["tool_calls"].append(tool_call_info)
            
            steps_data.append(step_info)
            yield create_event("step_processed", step_info)
        
        # Get final response
        yield create_event("response_retrieval", {"status": "retrieving agent response"})
        response_message = project_client.agents.messages.get_last_message_by_role(
            thread_id=thread.id, 
            role=MessageRole.AGENT
        )
        
        response_data = {
            "text_messages": [],
            "citations": []
        }
        
        if response_message:
            # Extract text messages
            for text_message in response_message.text_messages:
                response_data["text_messages"].append(text_message.text.value)
            
            # Extract citations
            for annotation in response_message.url_citation_annotations:
                citation = {
                    "title": annotation.url_citation.title,
                    "url": annotation.url_citation.url
                }
                response_data["citations"].append(citation)
        
        yield create_event("response_ready", response_data)
        
        # Cleanup
        yield create_event("cleanup", {"status": "deleting agent"})
        project_client.agents.delete_agent(agent.id)
        yield create_event("cleanup_complete", {"status": "agent deleted"})
        
        # Final completion event
        yield create_event("completed", {
            "status": "success",
            "final_response": response_data
        })
        
    except Exception as e:
        yield create_event("error", {
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        raise

@app.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """Stream the agent query execution process"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        project_client = create_project_client()
        
        async def event_generator():
            async for event in stream_agent_events(
                project_client=project_client,
                query=request.message,
                model=request.model,
                agent_name=request.agent_name,
                instructions=request.instructions
            ):
                yield event
        
        return StreamingResponse(
            event_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.post("/query")
async def query_agent(request: QueryRequest):
    """Non-streaming version that returns the final result"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        project_client = create_project_client()
        
        with project_client:
            # Get playwright connection
            playwright_connection = project_client.connections.get(name="playwright-eastus")
            
            # Create agent
            agent = project_client.agents.create_agent(
                model=request.model,
                name=request.agent_name,
                instructions=request.instructions,
                tools=[{
                    "type": "browser_automation",
                    "browser_automation": {
                        "connection": {
                            "id": playwright_connection.id,
                        }
                    }
                }],
            )
            
            # Create thread
            thread = project_client.agents.threads.create()
            
            # Create message
            message = project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=request.message
            )
            
            # Create and process run
            run = project_client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id,
            )
            
            result = {
                "run_id": run.id,
                "status": run.status,
                "agent_id": agent.id,
                "thread_id": thread.id,
                "steps": [],
                "response": {
                    "text_messages": [],
                    "citations": []
                }
            }
            
            if run.status == "failed":
                result["error"] = str(run.last_error)
            
            # Get run steps
            run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)
            
            for step in run_steps:
                step_info = {
                    "step_id": step['id'],
                    "status": step['status'],
                    "tool_calls": []
                }
                
                step_details = step.get("step_details", {})
                tool_calls = step_details.get("tool_calls", [])
                
                for call in tool_calls:
                    tool_call_info = {
                        "call_id": call.get('id'),
                        "type": call.get('type'),
                        "function_name": call.get("function", {}).get('name')
                    }
                    step_info["tool_calls"].append(tool_call_info)
                
                result["steps"].append(step_info)
            
            # Get final response
            response_message = project_client.agents.messages.get_last_message_by_role(
                thread_id=thread.id,
                role=MessageRole.AGENT
            )
            
            if response_message:
                for text_message in response_message.text_messages:
                    result["response"]["text_messages"].append(text_message.text.value)
                
                for annotation in response_message.url_citation_annotations:
                    citation = {
                        "title": annotation.url_citation.title,
                        "url": annotation.url_citation.url
                    }
                    result["response"]["citations"].append(citation)
            
            # Cleanup
            project_client.agents.delete_agent(agent.id)
            
            return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/")
async def root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Azure AI Agents FastAPI Backend",
        "endpoints": {
            "GET /": "Main HTML interface",
            "POST /query": "Execute agent query (non-streaming)",
            "POST /query/stream": "Execute agent query with streaming events",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)