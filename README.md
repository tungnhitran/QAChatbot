# Simple chatbot Q&A by Gradio, 

A basic application of Gradio that integrates with IBM Watson AI to create a functional chatbot. This project uses LLM model, Granite 3-2b-instruct.

![Demo GIF](demo.gif)

## Features

- Integration with IBM Watson AI (watsonx)
- Support for LLM models (Granite)
- Built with Gradio

## Prerequisites

- Python 3.x
- pip (Python package manager)

## Setup

### 1. Clone the Repository

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your Watson AI credentials:

```
WATSONX_URL=https://au-syd.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_APIKEY=your_api_key_here
```

### 5. Run the Application

```bash
python llm_model.py
```

The application will be available at `http://127.0.0.1:7860` (the server name and pỏt can be changed)

## Usage

### Run Application

Navigate to `http://127.0.0.1:7860` to access the chatbot where you can interact with the models by entering the prompt at input and get the output after submitting.

## Security

- Credentials are stored in `.env` and excluded from version control
- Never share your `.env` file or commit it to the repository
- Regenerate API keys if they are accidentally exposed
- Use `.env.example` as a template for other developers

## Dependencies

Main dependencies include:
- `gradio` - Interface deployment
- `ibm-watsonx-ai` - IBM Watson AI SDK
- `python-dotenv` - Environment variable management

See `requirements.txt` for a complete list.

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Support

The LLM Models are deployed from IBM Watsonx AI

## Authors

(Alex) Tung Nhi TRAN