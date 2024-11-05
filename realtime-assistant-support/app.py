import os
import asyncio
from openai import AsyncAzureOpenAI

import chainlit as cl
from uuid import uuid4
from chainlit.logger import logger

from realtime import RealtimeClient
from realtime.tools import tools

client = AsyncAzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                          azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                          azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                          api_version="2024-10-01-preview")    

async def setup_openai_realtime(system_prompt: str):
    """Instantiate and configure the OpenAI Realtime Client"""
    openai_realtime = RealtimeClient(system_prompt = system_prompt)
    cl.user_session.set("track_id", str(uuid4()))
    
    async def handle_conversation_updated(event):
        item = event.get("item")
        delta = event.get("delta")
        """Currently used to stream audio back to the client."""
        if delta:
            # Only one of the following will be populated for any given event
            if 'audio' in delta:
                audio = delta['audio']  # Int16Array, audio added
                await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="pcm16", data=audio, track=cl.user_session.get("track_id")))
                
            if 'arguments' in delta:
                arguments = delta['arguments']  # string, function arguments added
                pass
            
    async def handle_item_completed(item):
        """Generate the transcript once an item is completed and populate the chat context."""
        try:
            transcript = item['item']['formatted']['transcript']
            if transcript != "":
                await cl.Message(content=transcript).send()
        except:
            pass
    
    async def handle_conversation_interrupt(event):
        """Used to cancel the client previous audio playback."""
        cl.user_session.set("track_id", str(uuid4()))
        await cl.context.emitter.send_audio_interrupt()
        
    async def handle_input_audio_transcription_completed(event):
        item = event.get("item")
        delta = event.get("delta")
        if 'transcript' in delta:
            transcript = delta['transcript']
            if transcript != "":
                await cl.Message(author="You", type="user_message", content=transcript).send()
        
    async def handle_error(event):
        logger.error(event)
        
    
    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted', handle_conversation_interrupt)
    openai_realtime.on('conversation.item.input_audio_transcription.completed', handle_input_audio_transcription_completed)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    coros = [openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools]
    await asyncio.gather(*coros)
    

system_prompt = """Provide helpful and empathetic support responses to customer inquiries for ShopMe, addressing their requests, concerns, or feedback professionally.

Maintain a friendly and service-oriented tone throughout the interaction to ensure a positive customer experience.

# Steps

1. **Identify the Issue:** Carefully read the customer's inquiry to understand the problem or question they are presenting.
2. **Gather Relevant Information:** Check for any additional data needed, such as order numbers or account details, while ensuring the privacy and security of the customer's information.
3. **Formulate a Response:** Develop a solution or informative response based on the understanding of the issue. The response should be clear, concise, and address all parts of the customer's concern.
4. **Offer Further Assistance:** Invite the customer to reach out again if they need more help or have additional questions.
5. **Close Politely:** End the conversation with a polite closing statement that reinforces the service commitment of ShopMe.

# Output Format

Provide a clear and concise paragraph addressing the customer's inquiry, including:
- Acknowledgment of their concern
- Suggested solution or response
- Offer for further assistance
- Polite closing

# Notes
- Greet user with Welcome to ShopMe For the first time only
- Ensure all customer data is handled according to relevant privacy and data protection laws and ShopMe's privacy policy.
- In cases of high sensitivity or complexity, escalate the issue to a human customer support agent.
- Keep responses within a reasonable length to ensure they are easy to read and understand."""

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hi, Welcome to ShopMe. How can I help you?. Press `P` to talk!"
    ).send()
    await setup_openai_realtime(system_prompt=system_prompt + "\n\n Customer ID: 12121")

@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.send_user_message_content([{ "type": 'input_text', "text": message.content}])
    else:
        await cl.Message(content="Please activate voice mode before sending messages!").send()

@cl.on_audio_start
async def on_audio_start():
    try:
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        # TODO: might want to recreate items to restore context
        # openai_realtime.create_conversation_item(item)
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime:            
        if openai_realtime.is_connected():
            await openai_realtime.append_input_audio(chunk.data)
        else:
            logger.info("RealtimeClient is not connected")

@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()