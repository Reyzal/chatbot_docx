import streamlit as st
import docx2txt
import re
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_docx_text(docx_text):
    pre_text = docx2txt.process(docx_text)
    text = clean_text(pre_text)
    return text


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


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


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
    Respond only based on the provided document. Use {vectorstore} tag when referencing document content    
    """

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, template=prompt_template)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 10}),
        memory=memory
    )

    return chat_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple Docx",
                       page_icon=":bookmark_tabs:")
    st.write(css, unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple Docx :bookmark_tabs:")
    user_question = st.text_input("Ask a question about your documents: ")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        docx_text = st.file_uploader("Upload your Docx here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Loading"):
                raw_text = ""
                for docx_file in docx_text:
                    text = get_docx_text(docx_file)
                    raw_text += text

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.chat = get_chat_chain(vectorstore)


if __name__ == '__main__':
    main()
