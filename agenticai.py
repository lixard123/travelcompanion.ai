import streamlit as st
import pandas as pd
import time
from langchain_openai import ChatOpenAI  # Updated import
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain  # Use LLMChain instead of ConversationChain, and RunnableWithMessageHistory is the newest replacement.

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Define the Excel file path for conversation flow
excel_file_path = 'conversation_flow.xlsx'

# Load the conversation flow from the Excel file
def load_conversation_flow():
    try:
        df = pd.read_excel(excel_file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{excel_file_path}' not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the Excel file: {e}")
        return None

# Create the conversation chain with memory and prompt
def create_conversation_chain():
    # Define the template for conversation prompts
    prompt_template = """
    You are a helpful assistant. Here is the conversation history:
    {conversation_history}
    User's message: {user_message}
    Respond accordingly.
    """
    prompt = PromptTemplate(
        input_variables=["conversation_history", "user_message"],
        template=prompt_template
    )
    
    # Initialize the OpenAI chat model (GPT-4)
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
    except Exception as e:
        st.error(f"Error initializing ChatOpenAI: {e}")
        return None

    # Set up memory to store conversation history
    memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

    # Create conversation chain with memory and prompt
    conversation_chain = LLMChain(
        llm=llm,
        memory=memory,
        prompt=prompt
    )
    return conversation_chain

# Main function to handle the conversation loop
def main():
    # Load the conversation flow data
    df = load_conversation_flow()

    if df is None:
        return  # Exit if Excel loading failed

    # Create the conversation chain
    conversation_chain = create_conversation_chain()

    if conversation_chain is None:
        return #Exit if chain creation failed.

    # Initialize conversation history
    conversation_history = ""

    # Start the conversation loop based on the flow from the Excel file
    for index, row in df.iterrows():
        # Display the conversation step
        user_message = row['User Message']  # Assuming 'User Message' column exists in your Excel

        st.write(f"User: {user_message}")
        
        # Append the user's message to the conversation history
        conversation_history += f"User: {user_message}\n"
        
        # Run the conversation chain and get the agent's response
        try:
            agent_response = conversation_chain.run(user_message)
        except Exception as e:
            st.error(f"Error during conversation chain run: {e}")
            return

        # Display the agent's response
        st.write(f"Assistant: {agent_response}")
        
        # Update the conversation history with the agent's response
        conversation_history += f"Assistant: {agent_response}\n"
        
        # Optionally, pause before proceeding to the next iteration for better user experience
        st.write("**Waiting for next input...**\n")
        
        # Add a brief delay (use time.sleep instead of st.time.sleep)
        time.sleep(2)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
