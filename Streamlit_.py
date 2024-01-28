import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
 
def main():
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(text)
        query = st.text_input("Ask questions about your PDF file:")
 
        if query:
            embeddings = OpenAIEmbeddings()
            docs = FAISS.from_texts(texts, embeddings).similarity_search(query)
            chain = load_qa_chain(OpenAI(), chain_type="stuff")
            response = chain.run(input_documents=docs, question=query, model_name = "gpt-3.5-turbo-instruct")
            print(response)
            st.write(response)
 
if __name__ == '__main__':
    main()