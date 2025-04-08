import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai

# OpenAI API Key (set up in Streamlit secrets)
openai.api_key = "open-ai-key"  # Replace this with your actual OpenAI API key or use Streamlit secrets

# Load the conversation flow from Excel file
def load_conversation_flow(file_path):
    df = pd.read_excel(file_path)
    return df

# Define the LLM and Conversation Chain
def create_conversation_chain():
    # Initialize the memory to store conversation history
    memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

    # Set up the LLM (Language Model) for the conversation
    llm = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-3.5-turbo")

    # Define the prompt template based on the conversation flow
    prompt_template = """
    You are an assistant named {agent_name}. Your role is {agent_role}.
    You will assist users by responding based on the flow in your conversation data.
    Use the user's messages and maintain a friendly, helpful tone. You have a memory of the conversation.
    {conversation_history}
    User: {user_message}
    Assistant: 
    """

    # Instantiate the prompt with dynamic fields for agent name, role, and conversation history
    prompt = PromptTemplate(
        input_variables=["agent_name", "agent_role", "conversation_history", "user_message"],
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

    # Display the instructions to the user
    st.write("Welcome! I am here to help with your travel plans. Let me know what you'd like assistance with.")

    # Initialize conversation chain
    conversation_chain = create_conversation_chain()

    # Display agent information (Olivia the Concierge)
    agent_name = "Olivia"
    agent_role = "Concierge"
    conversation_history = ""

    # Get user input (chat with the agent)
    user_input = st.text_input("Ask me anything about your travel plans:")

    if user_input:
        # Add user input to the conversation history
        conversation_history += f"User: {user_input}\n"

        # Get the agent's response based on the conversation flow
        agent_response = conversation_chain.run(agent_name=agent_name, 
                                                agent_role=agent_role, 
                                                conversation_history=conversation_history, 
                                                user_message=user_input)

        # Display the agent's response
        st.write(f"**{agent_name} (Concierge):** {agent_response}")

        # Update conversation history with agent's response
        conversation_history += f"Assistant: {agent_response}\n"

if __name__ == "__main__":
    main()
