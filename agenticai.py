import pandas as pd
import openai
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Load the conversation flow from the Excel file
def load_conversation_flow():
    # Path to the Excel file in the main directory
    excel_file_path = 'conversation_flow.xlsx'  # Assuming the file is in the main directory
    df = pd.read_excel(excel_file_path)
    return df

# Create a RunnableWithMessageHistory based on the loaded conversation flow
def create_conversation_chain():
    # Set up the LLM (GPT-4 model) with OpenAI API key
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")
    
    # Set up memory to store conversation history
    memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)
    
    # Set up prompt template (define the format of the conversation prompt)
    prompt = PromptTemplate(
        input_variables=["conversation_history", "user_message"],
        template="The following is a conversation with an AI assistant:\n{conversation_history}User: {user_message}\nAssistant:"
    )
    
    # Create a RunnableWithMessageHistory with memory and prompt
    conversation_chain = RunnableWithMessageHistory(
        llm=llm,
        memory=memory,
        prompt=prompt
    )
    return conversation_chain

# Main function to load data, process conversation flow, and return responses
def main():
    # Load conversation flow from the Excel file
    df = load_conversation_flow()

    # Initialize the conversation chain
    conversation_chain = create_conversation_chain()

    # Start the conversation loop
    conversation_history = ""
    for index, row in df.iterrows():
        # Extract user and assistant messages from the Excel file
        user_message = row['User Message']
        assistant_message = row['Assistant Message']

        # Append the conversation history for each step
        conversation_history += f"User: {user_message}\n"
        conversation_history += f"Assistant: {assistant_message}\n"

        # Get the agent's response based on the conversation flow
        agent_response = conversation_chain.run(
            conversation_history=conversation_history, 
            user_message=user_message
        )
        
        # Display the agent's response
        st.write(f"User: {user_message}")
        st.write(f"Assistant: {assistant_message}")
        st.write(f"Agent's Response: {agent_response}")

# Run the main function
if __name__ == "__main__":
    main()
