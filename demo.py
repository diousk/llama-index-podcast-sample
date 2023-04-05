import os
from llama_index import ServiceContext, GPTSimpleVectorIndex, PromptHelper, SimpleDirectoryReader, SimpleWebPageReader, LLMPredictor, OpenAIEmbedding, download_loader
from llama_index.node_parser import SimpleNodeParser
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
from color import bcolors
import gradio as gr

load_dotenv()

# setup reader from llamahub
print('loading AudioTranscriber')
AudioTranscriber = download_loader("AudioTranscriber")
loader = AudioTranscriber()
print('loading audio')
filePath = Path('./podcast.mp3')
documents = loader.load_data(file=filePath)

print('setup nodes')
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Configure prompt parameters and initialise helper
max_input_size = 4096
num_output = 1024
max_chunk_overlap = 20

print('setup prompt')
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=num_output),
)

# construct the index
print('setup index')
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index = GPTSimpleVectorIndex(nodes, service_context=service_context)

# setup UI & query function
def index_query(question):
    print("start query: ", question)
    response = index.query(question)
    print(f"{bcolors.WARNING}", response, bcolors.ENDC)
    return question + ": \n" + str(response)

demo = gr.Interface(fn = index_query, inputs="text", outputs="text")
demo.launch()
