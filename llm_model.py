import os
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

model_id = 'ibm/granite-3-2b-instruct'

parameters = {
    GenParams.TEMPERATURE: 0.5,
    GenParams.MAX_NEW_TOKENS: 256
}

project_id = os.getenv('WATSONX_PROJECT_ID')
watsonx_url = os.getenv('WATSONX_URL')

# Define model
watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url=watsonx_url,
    project_id=project_id,
    params=parameters,
)

# Function to get response from the model
def get_model_response(prompt):
    response = watsonx_llm.invoke(prompt)
    return response

# Create Gradio interface
chatbot = gr.Interface(
    fn=get_model_response,
    allow_flagging="never",
    inputs=gr.Textbox(lines=5, label="Input", placeholder="Enter your prompt here..."),
    outputs=gr.Textbox(lines=5, label="Output"),
    title="Watsonx AI Chatbot",
    description="Interact with the IBM Watsonx AI Granite-3-2B Instruct model using this interface."
)

# Launch the Gradio app
chatbot.launch(server_name="127.0.0.1", server_port=7860)