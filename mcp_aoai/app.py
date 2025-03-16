import json
from mcp import ClientSession
import os
import re
from aiohttp import ClientSession
import chainlit as cl
from openai import AzureOpenAI, AsyncAzureOpenAI
import traceback
from dotenv import load_dotenv
load_dotenv("azure.env")

SYSTEM_PROMPT = "you are a helpful assistant."

class ChatClient:
    def __init__(self) -> None:
        self.deployment_name = os.environ["AZURE_OPENAI_MODEL"]
        self.client = AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview",
            )
        self.messages = []
        self.system_prompt = SYSTEM_PROMPT
    async def process_response_stream(self, response_stream, tools, temperature=0):
        """
        Recursively process response streams to handle multiple sequential function calls.
        This function can call itself when a function call is completed to handle subsequent function calls.
        """
        function_arguments = ""
        function_name = ""
        tool_call_id = ""
        is_collecting_function_args = False
        collected_messages = []
        
        try:
            async for part in response_stream:
                if part.choices == []:
                    continue
                delta = part.choices[0].delta
                finish_reason = part.choices[0].finish_reason
                
                # Process assistant content
                if delta.content:
                    collected_messages.append(delta.content)
                    yield delta.content
                
                # Handle tool calls
                if delta.tool_calls:
                    if len(delta.tool_calls) > 0:
                        tool_call = delta.tool_calls[0]
                        
                        # Get function name
                        if tool_call.function.name:
                            function_name = tool_call.function.name
                            tool_call_id = tool_call.id
                        
                        # Process function arguments delta
                        if tool_call.function.arguments:
                            function_arguments += tool_call.function.arguments
                            is_collecting_function_args = True
                
                # Check if we've reached the end of a tool call
                if finish_reason == "tool_calls" and is_collecting_function_args:
                    # Process the current tool call
                    print(f"function_name: {function_name} function_arguments: {function_arguments}")
                    function_args = json.loads(function_arguments)
                    mcp_tools = cl.user_session.get("mcp_tools", {})
                    mcp_name = None
                    for connection_name, session_tools in mcp_tools.items():
                        if any(tool.get("name") == function_name for tool in session_tools):
                            mcp_name = connection_name
                            break
                    
                    reply_to_customer = function_args.get('reply_to_customer')
                    print(f"reply_to_customer: {reply_to_customer}")
                    # Output any replies to the customer
                    if reply_to_customer:
                        tokens = re.findall(r'\s+|\w+|[^\w\s]', reply_to_customer)
                        for token in tokens:
                            yield token
                    
                    # Add the assistant message with tool call
                    self.messages.append({
                        "role": "assistant", 
                        "content": reply_to_customer,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "function": {
                                    "name": function_name,
                                    "arguments": function_arguments
                                },
                                "type": "function"
                            }
                        ]
                    })
                    func_response = await call_tool(mcp_name, function_name, function_args)
                    # Add the tool response
                    self.messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": func_response,
                    })
                    
                    # Create a new stream to continue processing
                    new_response = await self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=self.messages,
                        tools=tools,
                        parallel_tool_calls=False,
                        stream=True,
                        temperature=temperature
                    )
                    
                    # Use a separate try block for recursive processing
                    try:
                        async for token in self.process_response_stream(new_response, tools, temperature):
                            yield token
                    except GeneratorExit:
                        return
                    return
                
                # Check if we've reached the end of assistant's response
                if finish_reason == "stop":
                    # Add final assistant message if there's content
                    if collected_messages:
                        final_content = ''.join([msg for msg in collected_messages if msg is not None])
                        if final_content.strip():
                            self.messages.append({"role": "assistant", "content": final_content})
                    return
        except GeneratorExit:
            return
        except Exception as e:
            print(f"Error in process_response_stream: {e}")
            traceback.print_exc()
    
    # Main entry point that uses the recursive function
    async def generate_response(self, human_input, tools, temperature=0):
        print(f"human_input: {human_input}")
        self.messages.append({"role": "user", "content": human_input})
        response_stream = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=self.messages,
            tools=tools,
            parallel_tool_calls=False,
            stream=True,
            temperature=temperature
        )
        try:
        # Process the initial stream with our recursive function
            async for token in self.process_response_stream(response_stream, tools, temperature):
                yield token
        except GeneratorExit:
            return


def flatten(xss):
    return [x for xs in xss for x in xs]

@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "parameters": t.inputSchema,
        } for t in result.tools]
    
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.step(type="tool") 
async def call_tool(mcp_name, function_name, function_args):
    try:
        print(f"Function Name: {function_name} Function Args: {function_args}")
        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
        func_response = await mcp_session.call_tool(function_name, function_args)
    except Exception as e:
        traceback.print_exc()
        func_response = json.dumps({"error": str(e)})
    return str(func_response.content)

@cl.on_chat_start
async def start_chat():
    client = ChatClient()
    cl.user_session.set("messages", [])
    cl.user_session.set("system_prompt", SYSTEM_PROMPT)
    
@cl.on_message
async def on_message(message: cl.Message):
    mcp_tools = cl.user_session.get("mcp_tools", {})
    tools = flatten([tools for _, tools in mcp_tools.items()])
    tools = [{"type": "function", "function": tool} for tool in tools]
    
    # Create a fresh client instance for each message
    client = ChatClient()
    # Restore conversation history
    client.messages = cl.user_session.get("messages", [])
    
    msg = cl.Message(content="")
    async for text in client.generate_response(human_input=message.content, tools=tools):
        await msg.stream_token(text)
    
    # Update the stored messages after processing
    cl.user_session.set("messages", client.messages)