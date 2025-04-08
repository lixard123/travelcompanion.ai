import pandas as pd
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

# Read the conversation flow from Excel (assuming the file is already uploaded or available locally)
def load_conversation_flow(excel_path):
    df = pd.read_excel(excel_path)
    return df

# Initialize the agent with conversation prompts, including a system message
def initialize_agent(df):
    agent_prompts = {}
    system_message = "You are an AI Travel Assistant. Your role is to guide users through the process of planning their travel by offering recommendations, answering questions about destinations, and helping with travel bookings. Please ensure the tone is friendly, professional, and focused on helping the user make decisions for their trip."
    agent_prompts['system_message'] = system_message  # Set system message to guide agent behavior
    
    # Add user/agent conversation flow from the Excel file
    for _, row in df.iterrows():
        step = row['Step']
        user_message = row['User Message']
        agent_prompts[step] = user_message
        
    return agent_prompts

# 🟢 Cache document processing (FAISS embedding)
@st.cache_resource(show_spinner=False)
def load_and_vectorize_pdfs(pdf_folder):
    """Loads and vectorizes PDFs from the specified folder (cached)."""
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)  # Explicit API key
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Main function to display the interface and engage with the user
def main():
    # Initialize Streamlit
    st.set_page_config(page_title="Agentic AI Travel Assistant", page_icon="🌍", layout="centered")
    
    st.title("🌍 Agentic AI Travel Assistant")
    
    # Load and initialize agent prompts from the Excel file
    excel_path = "travel_agent_conversation_flow.xlsx"  # Update the path as necessary
    df = load_conversation_flow(excel_path)
    agent_prompts = initialize_agent(df)
    
    # Displaying the system message first to guide the agent's behavior
    st.write(agent_prompts.get('system_message', "Hello! I'm your AI assistant ready to help you plan your next adventure."))

    # Feature overview
    features = {
        "📍 Places": "Get details about cities, landmarks, and hidden gems.",
        "🌦️ Weather": "Real-time weather forecasts for any location.",
        "🍽️ Cuisines": "Discover local and international food specialties.",
        "🏝️ Destinations": "Explore top tourist attractions and experiences.",
        "🛫 Travel Packages": "Find the best travel deals from our brochures."
    }
    for icon, description in features.items():
        st.markdown(f"- {icon}: {description}")

    # User query input
    user_query = st.text_input("What would you like to know?", "")

    # Load FAISS vectorstore (cached)
    pdf_folder = "brochures"
    vectorstore = load_and_vectorize_pdfs(pdf_folder)
    retriever = vectorstore.as_retriever()

    # Initialize OpenAI LLM
    llm = OpenAI(api_key=openai_api_key)

    # RetrievalQA with explicit chain type for better performance
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    if st.button("Get Information"):
        with st.spinner("Fetching details..."):
            response = None  # Initialize response variable

            # Check if we have a query from the user and engage the assistant
            if user_query:
                # Look for the appropriate stage of conversation based on user query
                response = agent_prompts.get("Destination & Interests Exploration", "Can you tell me what kind of trip you're looking for?")
                # Here, you can adjust the flow based on the user query, maybe using simple keyword matching or more complex NLP logic.
                
                # Step 1: Try FAISS Vector Store First
                response = qa_chain.run(user_query)

                # Step 2: If FAISS returns no answer, use OpenAI LLM
                if response is None or "I don't know" in response:
                    response = llm(user_query)
                    response += "\n\nFor exclusive travel packages to these destinations, contact Margie’s Travel! ✈️🌍"
                
                # Display response
                st.success(f"**{user_query}**: {response}")

if __name__ == "__main__":
    main()
