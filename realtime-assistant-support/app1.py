import chainlit as cl


@cl.on_chat_start
async def start():
    with open('order_status_template.html', 'r') as file:
        html_content = file.read()
    elements = [
        cl.Text(name="Order Status", content=html_content, display="inline")
    ]

    await cl.Message(
        content="Check out this text element!",
        elements=elements,
    ).send()
