# travelcompanion.ai
Travel Companion Agent Chat
Overview
This app is a conversational assistant designed to help users with their travel plans. The app is powered by OpenAI's GPT-3.5 model and uses Streamlit for the web interface. The assistant (currently Olivia the Concierge) guides users through planning trips, providing destination recommendations, itinerary suggestions, and general travel-related assistance. The conversation flow is dynamic and based on a pre-defined structure in an Excel file.

Features
Conversational Agent: Olivia the Concierge serves as the primary agent to assist users with travel planning.

User-Friendly Interface: Powered by Streamlit, allowing users to easily interact with the assistant via a simple text input.

Memory-Based Conversations: The app remembers the history of the conversation, making the interaction feel natural and flowing.

Dynamic Responses: The assistant’s responses are generated based on user input and conversation history, providing personalized assistance.

How It Works
Conversation Flow: The assistant follows a structured conversation flow, loaded from an Excel file (travel_agent_conversation_flow.xlsx). The flow outlines how the assistant engages with the user and provides travel-related responses.

Role and Memory: Olivia, the concierge, uses a memory buffer to keep track of the conversation, ensuring continuity in the dialogue.

OpenAI API: The app uses OpenAI’s GPT-3.5 model for generating responses, which are tailored based on the user’s input and the conversation history.

Installation Instructions
Prerequisites
Ensure you have the following installed on your machine:

Python 3.x

pip (Python package installer)

Step-by-Step Installation
Clone or download this repository to your local machine.

Install Required Libraries:

Use pip to install the necessary libraries:

bash
Copy
Edit
pip install streamlit pandas langchain openai
Set Up OpenAI API Key:

Add your OpenAI API key to the app. You can either:

Set it in your Streamlit secrets (recommended for production).

Replace the placeholder open-ai-key in the code with your actual key.

Prepare the Conversation Flow File:

Ensure the Excel file travel_agent_conversation_flow.xlsx is correctly formatted and located in the same directory as the app.

The file should include the conversation flow with the agent persona (Olivia), structured as described in the original code.

Run the App:

In your terminal, navigate to the directory where the script is located and run:

bash
Copy
Edit
streamlit run travel_agent_app.py
This will launch the app in your browser, typically at http://localhost:8501.

How to Use the App
Upon loading, you will be greeted by Olivia the Concierge, the primary agent who will assist you with your travel plans.

You can ask Olivia anything related to your trip, such as:

Recommendations for destinations.

Help with itineraries and bookings.

Travel advice, tips, and more.

Olivia will respond based on your input, using the conversation flow defined in the Excel file.

Conversation Flow Structure
The conversation flow is loaded from the Excel file (travel_agent_conversation_flow.xlsx), which contains:

User Messages: Example prompts or questions that the user might ask.

Role Messages: The expected response from the agent (Olivia).

Conversation Steps: How the conversation progresses, based on different user inputs.

Example Flow:
Agent Name	Agent Role	User Message	Role Message
Olivia	Concierge	"Hello, I need help planning a trip."	"I'd be happy to help! What kind of trip are you thinking about?"
Olivia	Concierge	"I’m looking for a beach vacation."	"Great choice! I can recommend some beautiful beach destinations. Do you have any specific places in mind?"
App Customization
Changing the Agent Persona:
If you wish to change the agent persona (e.g., Olivia to a Chauffeur or Visa Advisor), you can modify the agent name and role in the code, as well as update the conversation flow accordingly in the Excel file.

Adding More Agents:
If you want to add multiple agents in the future, you can expand the conversation flow and set up different agent personas (e.g., Chauffeur, Visa Advisor, Local Guide), and adapt the flow to switch between them based on user input.

Adjusting the Flow:
Modify the conversation steps and messages in the Excel file to suit your specific use case and keep the interaction engaging.

Troubleshooting
Error: Missing openai module: Ensure that you've installed all dependencies using pip install streamlit pandas langchain openai.

Error: Conversation History Issue: If the conversation history isn't saved correctly, ensure the ConversationBufferMemory is set up properly in the code.

Contact
For issues or enhancements, feel free to open an issue or contribute to this project.

GitHub Repository: [Link to your repository]

Email: [Your contact email]

This README provides an overview of the app's purpose, installation steps, and how to use and customize it. If you'd like to make any changes to the flow or agent behavior, you can easily modify the code or the conversation flow in the Excel file. Let me know if you need further modifications or clarifications!
