import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import pickle
import faiss
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

if "url_count" not in st.session_state:
    st.session_state.url_count = 1
    st.session_state.urls = [""]  
    st.session_state.model_ready = False  
    st.session_state.ask_question_mode = False  
    st.session_state.answer = None  
    st.session_state.sources = None  
    st.session_state.summary = None  

# function for adding URLs
def add_url_box():
    st.session_state.url_count += 1
    st.session_state.urls.append("")

# this function will load the data from news article, perform splitting, convert the text into embedding vectors and will store them in FAISS database locally.
def process_urls():
    with st.spinner("Preparing the data..."):
        loader = UnstructuredURLLoader(urls=st.session_state.urls)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        
        # storing the FAISS database locally.
        faiss_index = vectorindex_openai.index
        faiss_index_binary = faiss.serialize_index(faiss_index)
        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump(faiss_index_binary, f)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'])
        retriever = vectorindex_openai.as_retriever()
        
        # This prompt template is used for answering user queries
        prompt_template = """Act as a helpful assistant and answer the following question based only on the provided context. Use all the information given by the retriever to create an answer that is easy to understand and read in plain English. If there are multiple points, present the answer using bullet points for clarity. If there are any links present, like the link for registration etc, then make sure to list those links also. If the answer cannot be found in the context, respond with 'Please the official website for this!' without making up information.

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}

        st.session_state.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        
        # This prompt template is used for giving a descriptive summary of the article. If multiple schemes are present, then it will summarize all of them ony by one.
        summary_prompt_template = """Act as an assistant whose job is to provide all the important information about schemes. Based only on the provided context, generate a summary that is easy to read and understand. Note that there may be information about multiple schemes in the context. If so, provide a separate summary for each scheme, covering each scheme one by one.

        For each scheme:

        Start with a brief 3-4 line summary of the scheme.

        Follow with a detailed description covering the following four key topics:

        1) Scheme Benefits - Include all numerical figures present in the context, such as the total amount to be distributed and any financial assistance provided to recipients.
        2) Scheme Application Process- Include any links or URLs available in the context, such as registration links.
        3) Eligibility
        4) Documents Required

        Use bullet points for each topic if there are multiple details. Only use information from the context, and if information on any topic is not available, simply respond with 'Please the official website for this!' without making up details.

        CONTEXT: {context}"""

        SUMMARY_PROMPT = PromptTemplate(template=summary_prompt_template, input_variables=["context"])
        summary_chain_type_kwargs = {"prompt": SUMMARY_PROMPT}

        st.session_state.summary_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=False,
            chain_type_kwargs=summary_chain_type_kwargs
        )

        st.session_state.model_ready = True
    st.success("Processing successful! The model is ready.")

st.title("Scheme Research Tool")
st.write("Get to know about any scheme in detail!")

# side-bar to enter URLS
with st.sidebar:
    st.header("Enter URLs")
    for i in range(st.session_state.url_count):
        st.session_state.urls[i] = st.text_input(f"URL {i + 1}", st.session_state.urls[i], key=f"url_{i}")
    st.button("Add URL", on_click=add_url_box)

    if st.button("Process URLs"):
        process_urls()

if st.session_state.model_ready:
    st.write("Choose an action below:")

    if st.button("Ask a Question"):
        st.session_state.ask_question_mode = True  

    if st.button("Generate Summary"):
        summary_result = st.session_state.summary_chain({"query": "Generate a summary of the provided content"})
        st.session_state.summary = summary_result['result']  
        st.write("Summary:", st.session_state.summary)

    if st.session_state.ask_question_mode:
        question = st.text_input("Enter your question:", key="question_input")
        if st.button("Submit Question"):
            result = st.session_state.chain({"query": question})
            st.session_state.answer = result['result']  
            
            # Extract source documents for URL information
            st.session_state.sources = {doc.metadata['source'] for doc in result['source_documents']}


        if st.session_state.answer:
            st.write("Answer:", st.session_state.answer)
            if st.session_state.sources:
                st.write("Sources:")
                for source in st.session_state.sources:
                    st.write("- ", source)
