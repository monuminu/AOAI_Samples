# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
An example to show an application using Opentelemetry tracing api and sdk. Custom dependencies are
tracked via spans and telemetry is exported to application insights with the AzureMonitorTraceExporter.
"""
# mypy: disable-error-code="attr-defined"
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from dotenv import load_dotenv
load_dotenv('azure.env')
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

exporter = AzureMonitorTraceExporter.from_connection_string(
    os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
)

tracer_provider = TracerProvider()
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
trace_api.set_tracer_provider(tracer_provider)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=60000)
trace.get_tracer_provider().add_span_processor(span_processor)
LangChainInstrumentor().instrument()

from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
llm = AzureChatOpenAI(api_key = os.environ['AZURE_OPENAI_API_KEY'],
                      azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'], 
                      api_version = '2024-06-01', 
                      model= os.environ['AZURE_OPENAI_GPT_DEPLOYMENT'])


chain = LLMChain(llm=llm, prompt=prompt, metadata={"category": "jokes"})
completion = chain.predict(adjective="funny", metadata={"variant": "funny"})
print(completion)