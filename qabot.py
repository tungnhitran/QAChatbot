import os
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

model_id = 'ibm/granite-3-2b-instruct'
project_id = os.getenv('WATSONX_PROJECT_ID')
watsonx_url = os.getenv('WATSONX_URL')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

# LLM
def get_llm_model():
    parameters = {
        GenParams.TEMPERATURE: 0.5,
        GenParams.MAX_NEW_TOKENS: 256
    }
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url=watsonx_url,
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

# Load documents
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_doc = loader.load()
    return loaded_doc
# Text splitting
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks
# Create embeddings
def watsonx_embedding():
    embed_params = {
        GenParams.TEMPERATURE: 0.0,
        GenParams.MAX_NEW_TOKENS: 512
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id='intfloat/multilingual-e5-large',
        url=watsonx_url,
        project_id=project_id,
        params=embed_params,
    )
    return watsonx_embedding
# Vector database
def vector_db(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
    return vectordb

# Retrieval Info
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_db(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# QA Chain
def retrieval_qa(file, query):
    model = get_llm_model()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True
    )
    response = qa.invoke(query)
    return response['result']

# Gradio UI
chatbot = gr.Interface(
    fn=retrieval_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count='single', type='filepath', file_types=['.pdf']),
        gr.Textbox(lines=2, label="Input Prompt", placeholder="Enter your question about the document here..."),
    ],
    outputs=gr.Textbox(lines=5, label="Answer"),
    title="PDF Document QA Chatbot",
    description="Upload a PDF document and ask questions about its content using the IBM Watsonx AI Granite-3-2B Instruct model."
)

# Launch the app
chatbot.launch(server_name="127.0.0.1", server_port=7860)