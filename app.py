import re

import streamlit as st
import docx2txt
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


class ChatbotApp:
    def __init__(self, st_session_state=None, st_text_input=None, docx2txt_process=None):
        self.st_session_state = st_session_state
        self.st_text_input = st_text_input
        self.docx2txt_process = docx2txt_process
        load_dotenv()
    
    def initialize_session_state(self):
        if "chat" not in st.session_state:
            st.session_state.chat = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
    
    def run(self):
        st.header("Chat with multiple Docx :bookmark_tabs:")
        user_question = st.text_input("Ask a question about your documents: ")
        
        if user_question:
            if st.session_state.chat is None:
                st.error('Please upload documents')
                return  # Exit early to prevent further processing
            else:
                self.handle_userinput(user_question)
        
        with st.sidebar:
            st.subheader("Your documents")
            docx_text = st.file_uploader("Upload your Docx here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                if docx_text is None or len(docx_text) == 0:
                    st.warning("No documents uploaded. Please upload documents before processing.")
                else:
                    with st.spinner("Loading"):
                        raw_text = ""
                        for docx_file in docx_text:
                            text = self.get_docx_text(docx_file)
                            raw_text += text
                        
                        text_chunks = self.get_text_chunks(raw_text)
                        vectorstore = self.get_vectorstore(text_chunks)
                        
                        if vectorstore is not None:
                            st.session_state.chat = self.get_chat_chain(vectorstore)
    
    def handle_userinput(self, user_question):
        response = st.session_state.chat({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    @staticmethod
    def get_docx_text(docx_text):
        pre_text = docx2txt.process(docx_text)
        text = ChatbotApp.clean_text(pre_text)
        return text
    
    @staticmethod
    def clean_text(text):
        # Remove "Online Submission Guidelines" section and everything below it
        text = re.sub(r'Current Employees:.+', '', text, flags=re.DOTALL)
        # Remove sentences containing specific phrases
        text = (
            re.sub(r'Cornell\'s Culture of Inclusion and Community Standards.+? organizational success\.', '',
                   text, flags=re.DOTALL))
        # remove about arvix
        text = re.sub(r'About arXiv.+?ambitious project\.', '', text, flags=re.DOTALL)
        
        return text
    
    @staticmethod
    def get_text_chunks(raw_text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks
    
    @staticmethod
    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    
    @staticmethod
    def get_chat_chain(vectorstore):
        llm = ChatOpenAI()
        
        prompt_template = f"""
        You are an AI career advisor affiliated with Cornell University. Your task is to evaluate job postings for recent
        graduates. Follow these guidelines:
        If you don't know the answer, admit it. Do not guess. If the question is off-topic,
        explain your focus on context-related questions.
        Be specific and concise in your responses, unless the user seeks more detail.
        Avoid repeating unless necessary for clarity. Use chat history for summaries.
        Utilize context fields:
            "Job Posting" for job titles.
            "Posted Date" for posting dates.
            "Job Summary" for job details.
            "Minimum Qualifications" and "Preferred Qualifications" for qualifications.
            "Job Title" for university job titles.
            "Job Category" for job families.
            "Level" for job grades.
        Prioritize response based on the provided document. Use {vectorstore} tag when referencing document content
        If there is no document provided response as OpenAI
        """
        
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, template=prompt_template)
        chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 10}),
            memory=memory
        )
        
        return chat_chain


if __name__ == '__main__':
    app = ChatbotApp()
    app.run()

# Include the CSS template
st.write(css, unsafe_allow_html=True)
