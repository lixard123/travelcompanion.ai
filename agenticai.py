import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Load the conversation flow from Excel file
def load_conversation_flow(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to load and process PDF brochures
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages

# Define the LLM and Conversation Chain
def create_conversation_chain(pages):
    # Initialize the memory to store conversation history
    memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

    # Set up the LLM (Language Model) for the conversation
    llm = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4")

    # Define the prompt template based on the conversation flow
    prompt_template = """
    You are an assistant named {agent_name}. Your role is {agent_role}.
    You will assist users by responding based on the flow in your conversation data and the provided brochure information.
    Use the user's messages and maintain a friendly, helpful tone. You have a memory of the conversation.
    {conversation_history}
    Brochure Info: {brochure_content}
    User: {user_message}
    Assistant: 
    """

    # Instantiate the prompt with dynamic fields for agent name, role, conversation history, and brochure content
    prompt = PromptTemplate(
        input_variables=["agent_name", "agent_role", "conversation_history", "brochure_content", "user_message"],
        template=prompt_template
    )
    
    # Create the LLM chain with memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return chain

# Main function to handle the app flow
def main():
    # File path to the conversation flow and PDF brochures (assuming they're in the home directory)
    excel_path = "travel_agent_conversation_flow.xlsx"
    pdf_path = "brochure.pdf"  # Path to your brochure PDF

    # Load conversation flow from Excel file
    df = load_conversation_flow(excel_path)
    
    # Load and process PDF brochure
    pages = load_pdf(pdf_path)

    # Initialize conversation chain with brochure content
    brochure_content = " ".join([page.page_content for page in pages])
    conversation_chain = create_conversation_chain(brochure_content)

    # Streamlit app title
    st.title("Travel Companion Agent Chat")

    # Display the instructions to the user
    st.write("Welcome! I am here to help with your travel plans. Let me know what you'd like assistance with.")

    # Display agent information (Olivia the Concierge)
    agent_name = "Olivia"
    agent_role = "Concierge"
    conversation_history = ""

    # Process the conversation flow from the loaded Excel data
    for index, row in df.iterrows():
        # Extract role-specific data
        user_message = row['User Message']
        assistant_message = row['Assistant Message']

        # Append the conversation history for each step
        conversation_history += f"User: {user_message}\n"
        conversation_history += f"Assistant: {assistant_message}\n"

        # Get the agent's response based on the conversation flow and brochure content
        agent_response = conversation_chain.run(agent_name=agent_name, 
                                                agent_role=agent_role, 
                                                conversation_history=conversation_history, 
                                                brochure_content=brochure_content, 
                                                user_message=user_message)

        # Display the agent's response
        st.write(f"**{agent_name} (Concierge):** {agent_response}")

        # Update conversation history with agent's response
        conversation_history += f"Assistant: {agent_response}\n"

if __name__ == "__main__":
    main()
