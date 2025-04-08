import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from openai import OpenAIError  # Import OpenAIError from the correct location

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("open-ai-key", "")

# Define the CSV file path for conversation flow
csv_file_path = 'conversation_flow.csv'

# Load conversation flow from CSV
def load_conversation_flow_from_csv():
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{csv_file_path}' not found.  Please make sure the file is uploaded and the path is correct.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading conversation flow from CSV: {e}")
        return None

# Create the conversation chain with memory and prompt
def create_conversation_chain():
    prompt_template = """
    You are a helpful assistant guiding a user through a travel planning conversation. Here is the conversation history:
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
        return llm  # Return the LLM instance
    except OpenAIError as e:
        st.error(f"Error initializing ChatOpenAI. Please check your OpenAI API key and connection. Error details: {e}")
        return None  # Return None in case of an error
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

    

# Main function to handle the conversation loop
def main():
    df = load_conversation_flow_from_csv()

    if df is None:
        return

    llm = create_conversation_chain()  # Get the LLM instance
    if llm is None:
        return

    conversation_chain = LLMChain(
        llm=llm,
        memory=ConversationBufferMemory(memory_key="conversation_history", return_messages=True),
        prompt=PromptTemplate(
            input_variables=["conversation_history", "user_message"],
            template="""You are a helpful assistant guiding a user through a travel planning conversation. Here is the conversation history:
            {conversation_history}
            User's message: {user_message}
            Respond accordingly.
            """
        )
    )

    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = ""
        st.session_state.step_index = 0  # Start at the first step

    # Display previous conversation
    for message in st.session_state.conversation_history.split("\n"):
        if message:
            st.text(message)

    # Get the current step's message from the CSV
    if st.session_state.step_index < len(df):
        current_step_message = df.iloc[st.session_state.step_index]['Message']
        # Check if 'Role' column exists before accessing it
        if 'Role' in df.columns:
            role = df.iloc[st.session_state.step_index]['Role']  # get role
        else:
            role = "Assistant"  # Default role if 'Role' column is missing

        if role == "Assistant":
            st.text(f"Assistant: {current_step_message}")  # Show the assistant's message from CSV
            user_message = st.text_input("Your response:", key="user_input", value="")
        elif role == "User":
            user_message = st.text_input(f"User: {current_step_message}", key="user_input", value="")
        else:
            user_message = st.text_input("Your response:", key="user_input", value="")

        if user_message:
            st.write(f"User: {user_message}")
            conversation_history = st.session_state.conversation_history + f"User: {user_message}\n"
            st.session_state.conversation_history = conversation_history

            try:
                agent_response = conversation_chain.run(user_message)
                st.write(f"Assistant: {agent_response}")
                conversation_history = st.session_state.conversation_history + f"Assistant: {agent_response}\n"
                st.session_state.step_index += 1
                st.rerun()

            except OpenAIError as e:
                st.error(
                    "I encountered an issue processing your request.  Could you please rephrase your input?"
                )
                st.session_state.conversation_history = conversation_history  # Keep history
                st.rerun()  # Force a refresh to show the error and input again
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.step_index += 1
                st.rerun()

    else:
        st.write("End of conversation flow.")
        st.button("Restart", on_click=reset_conversation)


def reset_conversation():
    st.session_state.conversation_history = ""
    st.session_state.step_index = 0
    st.rerun()


if __name__ == "__main__":
    main()
