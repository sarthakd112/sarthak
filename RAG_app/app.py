import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from glob import glob
from tqdm import tqdm
import yaml
import base64 
from link_scraper import LinkScraper
from text_extractor import TextExtractor
from langchain.document_loaders import TextLoader

url_for_scraping = None

def scrape(head_url, max_length, num_of_urls):
    try:
        web_scraper = LinkScraper(head_url, max_length, num_of_urls)
        web_scraper.Links_Extractor()
        print("All the available links are saved...")

        # Text extraction
        docs = TextExtractor().Extract_text()

        print("All the text got scraped")
        acknowledgment = "Web Scraping done..."
        return acknowledgment, docs
    except Exception as e:
        print(f"Error occurres in the main function: {e}")
        return None
        
        
load_dotenv()

def load_config():
    """
    Load configuration from the 'config.yaml' file.

    Returns:
        dict: Configuration settings.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

def load_embeddings(model_name=config["embeddings"]["name"],
                    model_kwargs={'device': config["embeddings"]["device"]}):
    """
    Load HuggingFace embeddings.

    Args:
        model_name (str): The name of the HuggingFace model.
        model_kwargs (dict): Keyword arguments for the model.

    Returns:
        HuggingFaceEmbeddings: Embeddings model.
    """
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

def load_documents():
    """
    Load documents from PDF files in the specified directory or from the provided URL.

    Args:
        url_for_scraping (str): The URL for scraping. If None, load documents from PDF files.

    Returns:
        list: List of loaded documents.
    """
    print("Loading the data")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["TextSplitter"]["chunk_size"],
                                                   chunk_overlap=config["TextSplitter"]["chunk_overlap"])
    documents = []
    
    try:
        doc_content = TextLoader("./Content.txt", encoding="utf-8").load()
        chunked_docs = text_splitter.split_documents(doc_content)
        print("going to add the chunks in documents")
        documents.extend(chunked_docs)

        for item_path in tqdm(glob("data/" + "*.pdf"), desc="Loading PDF files"):
            print("inside the data directory and extracting the all pdf files")
            loader = PyPDFLoader(item_path)
            try:
                documents.extend(loader.load_and_split(text_splitter=text_splitter))
            except Exception as e:
                print(f"Error loading document {item_path}: {e}")
    except Exception as e:
        print(f"Error loading or splitting documents: {e}")

    return documents


def save_uploaded_file(uploaded_file, destination_folder):
    """
    Save the uploaded file to the specified destination folder.

    Args:
        uploaded_file (BytesIO): The uploaded file.
        destination_folder (str): The folder where the file should be saved.

    Returns:
        str: The path to the saved file.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    file_path = os.path.join(destination_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())  
    return file_path

def answer_question(question: str):
    """
    Answer a question using the loaded documents and retriever.

    Args:
        question (str): The user's question.

    Returns:
        tuple: A tuple containing the answer and the retriever.
    """
    embedding_function = load_embeddings()
    documents = load_documents()
    
    db = FAISS.from_documents(documents, embedding_function)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    qa_chain_prompt = PromptTemplate.from_template("Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible. Always say 'thanks for asking!' at the end of the answer. {context} Question: {question} Helpful Answer:")
    
    qa_chain = RetrievalQA.from_chain_type(
        get_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_chain_prompt},
        return_source_documents=True
    )

    output = qa_chain({"query": question})
    return output["result"], retriever 

repo_id = "google/flan-t5-xxl"
    
def get_llm():
    """
    Get the HuggingFace Hub model.

    Returns:
        HuggingFaceHub: The HuggingFace Hub model.
    """
    try:
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 256})
        return llm
    except Exception as e:
        print(f"Error loading language model: {e}")
        return None

def displayPDF(file):
    """
    Display a PDF file in the HTML format.

    Args:
        file (str): The path to the PDF file.

    Returns:
        str: HTML code to embed the PDF.
    """
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = (f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" 'f'type="application/pdf">')
                   
    return pdf_display
    
def main():
    """
    Main function to run the Streamlit application.
    """
 
    # Center-align and change the color of the text
    st.markdown("<h1 style='text-align: center; color: blue;'>DocChat</h1>", unsafe_allow_html=True)

    # Help Section
    st.sidebar.subheader("Help")
    with st.sidebar.expander("How to Use"):
        st.write("1. **Interact with Chatbot:**")
        st.write("   - **Upload a PDF file:** Use the 'Browse files' button to upload a PDF file.")
        st.write("   - **Select a PDF file:** Choose a PDF file from the list or use the uploaded file.")
        st.write("   - **Expand the PDF Viewer:** Click the checkbox to view the contents of the PDF in an expanded viewer.")
        st.write("   - **Ask a question in the Chatbot section:** Enter your question in the text input field.")
        st.write("   - **View the Chatbot response and relevant contexts:** Check the chatbot response and relevant contexts in the text areas below.")
        st.write("2. **Interact with URL:**")
        st.write("   - Paste the URL you want to interact with inside the text area.")
        st.write("   - Click the 'Enter' button to perform web scraping using the provided URL.")
        st.write("   - View the Extracted data in the text area below.")
        st.write("   - Enter your question in the Chatbot.")
        st.write("   - Check the chatbot response and relevant contexts in the text areas below.")
        

    with st.sidebar:
        st.subheader("File handler")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
        # Check if a new file is uploaded
        if uploaded_file:
            st.success("File uploaded successfully!")
            save_uploaded_file(uploaded_file, "data")
            selected_pdf = st.selectbox("Select PDF", glob("data/*.pdf") if glob("data/*.pdf") else [])
        else:
            # Use the previously selected file (if any)
            selected_pdf = st.selectbox("Select PDF", glob("data/*.pdf") if glob("data/*.pdf") else [], key='selectbox_key')
        

    st.markdown("##### PDF Viewer")
    pdf_expander = st.checkbox("View PDF", False)
    if selected_pdf and pdf_expander:
            pdf_display = displayPDF(selected_pdf)
            st.markdown(pdf_display, unsafe_allow_html=True)
    elif pdf_expander:
            st.info("Please select or upload a PDF to display.")
    
    
    st.markdown("##### Chatbot")
    user_question = st.text_input(" ", placeholder = "Enter the question here")
    
    if user_question:
        answer, retriever = answer_question(user_question)
        st.text_area("Chatbot Response", answer)  
        relevant_documents = retriever.get_relevant_documents(query=user_question)
        
        # Concatenate metadata sources into a single string
        Source = "\n".join(set([f"{doc.metadata['source']}" for doc in relevant_documents]))
        print(Source)

        if "data/" in Source: 
        # Concatenate all documents into a single string
            all_contexts = "@@@@@\n\n".join([f"Source: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n\n{doc.page_content}\n{'='*70}" for doc in relevant_documents])
        else:
            all_contexts = "@@@@@\n\n".join([f"Source: {Source}\nContext:\n\n{context}\n{'='*70}" for context in relevant_documents])

        # Display all contexts in a single text area
        st.text_area("Retrieved Contexts", all_contexts)
        
    
    st.markdown("---")
    st.markdown("##### Interact with URL") 
    url_for_scraping  = st.text_input(" ",placeholder = "Paste your URL")
    
    # Trigger web scraping on button click
    if st.button("Enter"):
        # Call your web scraping function with url_for_scraping
        scraped_data = scrape(url_for_scraping, max_length=1000, num_of_urls=100)
        st.text_area("Extracted data ", scraped_data)


if __name__ == "__main__":
    main()