import pandas as pd
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load API key securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Check if the API key is available
if not openai_api_key:
    st.error("OpenAI API key not found in secrets.")

def load_conversation_flow():
    # Load conversation data from the updated Excel file
    excel_file_path = '/path/to/conversation_flow.xlsx'  # Update with the path to your Excel file
    df = pd.read_excel(excel_file_path)
    return df

def create_conversation_chain():
    # Initialize the LLM (assuming the use of GPT-4 model)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")

    # Create a memory buffer to hold the conversation context
    memory = ConversationBufferMemory()

    # Define a basic prompt template for the assistant
    prompt = PromptTemplate(
        input_variables=["user_message", "conversation_history"],
        template="Assistant, your goal is to assist the user based on the following conversation history:\n{conversation_history}\nUser: {user_message}\nAssistant:"
    )

    # Initialize the LLM chain
    conversation_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return conversation_chain

def process_conversation(conversation_chain, df):
    conversation_history = ""  # Initialize an empty conversation history

    for index, row in df.iterrows():
        # Extract user message and assistant message from the flow
        user_message = row['User Message']
        assistant_message = row['Assistant Message']

        # Append the user and assistant messages to the conversation history
        conversation_history += f"User: {user_message}\n"
        conversation_history += f"Assistant: {assistant_message}\n"

        # Get the agent's response using the conversation chain
        agent_response = conversation_chain.run(
            user_message=user_message,
            conversation_history=conversation_history
        )

        # Display the assistant's response (for testing)
        print(f"Assistant Response: {agent_response}\n")

def main():
    # Ensure the API key is available before proceeding
    if not openai_api_key:
        return

    # Load conversation flow from Excel file
    df = load_conversation_flow()

    # Create conversation chain
    conversation_chain = create_conversation_chain()

    # Process the conversation and generate responses
    process_conversation(conversation_chain, df)

# Run the main function
if __name__ == "__main__":
    main()
