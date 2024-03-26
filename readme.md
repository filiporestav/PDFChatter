### Welcome to PDFChatter
This is a simple (free) application which enables you to chat with your PDFs using LLMs and text embedding models of your choice.

![PDFChatter dashboard](https://github.com/filiporestav/PDFChatter/blob/main/PDFChatter.jpg)

## Conifguration
1. Clone the repository to your local machine. 
2. Make sure to have Python installed.
3. Navigate to the folder where the repo is cloned.
4. Create a local environment by typing 'python -m venv .venv' in the terminal inside the folder.
4. Type 'pip install -r requirements.txt' in your terminal inside the folder.
5. Create a free API token from HuggingFace by visiting this page: https://huggingface.co/settings/tokens
6. Insert your token inside the .env file
7. When everything is downloaded and your API token is configured, run the application by typing 'streamlit run app.py' inside the folder.

Note that using free text embedding models running on your local machine is very slow, thus I recommend either using OpenAI's vector embedding models or running them on a third party provider, using more computationally efficient hardware.