import os
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import openai
from langchain.document_loaders import PyPDFLoader

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Load the conversation flow from Excel file
def load_conversation_flow(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to load and process PDF brochures from the brochures folder
def load_pdf_from_folder(pdf_folder_path):
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    all_pages_content = {}
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        all_pages_content[pdf_file] = " ".join([page.page_content for page in pages])
    
    return all_pages_content

# Define the LLM and Conversation Chain
def create_conversation_chain(brochure_content):
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
    # File path to the conversation flow (assuming it's in the home directory)
    excel_path = "travel_agent_conversation_flow.xlsx"
    
    # Load conversation flow from Excel file
    df = load_conversation_flow(excel_path)

    # Streamlit app title
    st.title("Travel Companion Agent Chat")

    # Load PDF brochures from the "brochures" folder
    brochure_folder_path = "brochures"
    all_brochures_content = load_pdf_from_folder(brochure_folder_path)

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

        # Get brochure content for this conversation step (assuming we're using the first brochure file here)
        brochure_content = " ".join(all_brochures_content.values())  # You can modify to select a specific brochure
        
        # Ensure the correct parameters are passed to the LLMChain run method
        try:
            # Print debug information to verify parameters
            st.write(f"Conversation history so far: {conversation_history}")
            st.write(f"Brochure content: {brochure_content[:200]}...")  # Truncate for display

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
        except Exception as e:
            # Catch and display any errors
            st.error(f"Error occurred during agent response generation: {e}")

if __name__ == "__main__":
    main()
