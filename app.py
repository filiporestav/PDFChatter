import streamlit as st # For GUI
from dotenv import load_dotenv
from PyPDF2 import PdfReader # For reading PDF
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) # one pdfreader for each pdf
        for page in pdf_reader.pages: # loop through all pages in this pdf
            text += page.extract_text() 
    return text

# Returns a list of text chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    model_name="BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xxl", temperature=0.5, max_length=512)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def submit():
    st.session_state.user_question = st.session_state.input
    st.session_state.input = ""

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, msg in enumerate(st.session_state.chat_history):
        if i%2==0: # User input
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else: # Bot output
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def enableWriting():
    st.session_state.canWriteQuestion = True

def main():
    load_dotenv() # Allow us to access our API keys
    st.set_page_config(page_title="PDF Chatter", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "input" not in st.session_state:
        st.session_state.input = None
    if "canWriteQuestion" not in st.session_state:
        st.session_state.canWriteQuestion = False

    st.header("Chat with your PDFs! :books:")
    st.text_input("Ask a question about your documents:", key="input", on_change=submit(), disabled=not st.session_state.canWriteQuestion)
    if st.session_state.user_question:
        handle_user_input(st.session_state.user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process", on_click=enableWriting()): # True when button is pressed
            with st.spinner("Processing"): # add spinning icon while processing
                # retrieve the entire pdf text
                st.write("Retrieving text...")
                raw_text = get_pdf_text(pdf_docs)

                # divide the pdf text into chunks
                st.write("Dividing texts into chunks...")
                text_chunks = get_text_chunks(raw_text)

                # create vector store with chunk embeddings
                st.write("Creating vectorstore...")
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.write("Creating conversation chain...")
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.write("Ready to go!")

if __name__ == '__main__':
    main()