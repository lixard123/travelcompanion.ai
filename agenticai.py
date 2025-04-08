import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import time

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Define the CSV file path for conversation flow
csv_file_path = 'conversation.flow.csv'  # Changed to conversation.flow.csv

# Load conversation flow from CSV
def load_conversation_flow_from_csv():
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{csv_file_path}' not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading conversation flow from CSV: {e}")
        return None

# Create the conversation chain with memory and prompt
def create_conversation_chain():
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

    try:
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
    except Exception as e:
        st.error(f"Error initializing ChatOpenAI: {e}")
        return None

    memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

    conversation_chain = LLMChain(
        llm=llm,
        memory=memory,
        prompt=prompt
    )
    return conversation_chain

# Main function to handle the conversation loop
def main():
    df = load_conversation_flow_from_csv()

    if df is None:
        return

    conversation_chain = create_conversation_chain()

    if conversation_chain is None:
        return

    conversation_history = ""

    # Start the conversation loop based on the flow from the CSV file
    for index, row in df.iterrows():
        # Display the conversation step
        user_message = row['Message']  # Assuming 'Message' column exists in your CSV

        st.write(f"User: {user_message}")
        conversation_history += f"User: {user_message}\n"

        try:
            agent_response = conversation_chain.run(user_message)
        except Exception as e:
            st.error(f"Error during conversation chain run: {e}")
            return

        st.write(f"Assistant: {agent_response}")
        conversation_history += f"Assistant: {agent_response}\n"

        st.write("**Waiting for next input...**\n")
        time.sleep(2)

if __name__ == "__main__":
    main()
