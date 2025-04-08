import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain  # Use ConversationChain instead

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Define the Excel file path for conversation flow
excel_file_path = 'conversation_flow.xlsx'

# Load the conversation flow from the Excel file
def load_conversation_flow():
    df = pd.read_excel(excel_file_path)
    return df

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
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")

    # Set up memory to store conversation history
    memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

    # Create conversation chain with memory and prompt
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt
    )
    return conversation_chain

# Main function to handle the conversation loop
def main():
    # Load the conversation flow data
    df = load_conversation_flow()

    # Create the conversation chain
    conversation_chain = create_conversation_chain()

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
        agent_response = conversation_chain.run(
            {"conversation_history": conversation_history, "user_message": user_message}
        )
        
        # Display the agent's response
        st.write(f"Assistant: {agent_response}")
        
        # Update the conversation history with the agent's response
        conversation_history += f"Assistant: {agent_response}\n"
        
        # Optionally, pause before proceeding to the next iteration for better user experience
        st.write("**Waiting for next input...**\n")
        
        # Add a brief delay (optional)
        st.time.sleep(2)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
