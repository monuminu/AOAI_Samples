
import openai

openai.api_base = "https://streaming-endpoint.<Your Region>.inference.ml.azure.com/v1"
openai.api_key = ""

message_text = [{"role":"system","content":"You are a helpful assistant"}, 
                {"role": "user", "content": "write a python code to read csv?"}] 

# streaming example
completion = openai.ChatCompletion.create(
  model="llama-2-7b",
  headers = {"azureml-model-deployment": "streaming-deployment"},
  messages = message_text,
  temperature=0.7,
  max_tokens=512,
  stream=True,
)

for c in completion:
  if c.choices[0].delta.get("content"):
    print(c.choices[0].delta.content, end = "")
  
# non-streaming example
"""
completion = openai.ChatCompletion.create(
  model="llama-2-7b",
  headers = {"azureml-model-deployment": "streaming-deployment"},
  messages = message_text,
  temperature=0.7,
  max_tokens=25,
)

print(completion)
"""
