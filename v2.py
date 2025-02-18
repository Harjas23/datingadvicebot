
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as BlenderbotForConditionalGeneration


# Load BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Ensure conversation history is a list in session state
if "conversation_history" not in st.session_state or not isinstance(st.session_state.conversation_history, list):
    st.session_state.conversation_history = []

# Define the initial system prompt
INITIAL_PROMPT = (
    "Act as a dating advisor for Gen Z in 2025. Give advice on ghosting, red flags, and when to move on. Be Rough and honest\n"
)

def get_blenderbot_response(user_input):
    """Generate a response from BlenderBot while maintaining conversation history."""
    
    # Append user input to conversation history
    st.session_state.conversation_history.append(f"User: {user_input}")

    # Keep only the last few exchanges (to prevent exceeding token limits)
    truncated_history = st.session_state.conversation_history[-5:]  # Keep last 5 messages

    # Formulate the input prompt
    full_prompt = INITIAL_PROMPT + "\n".join(truncated_history) + "\nBot:"

    # Encode while ensuring it fits within token limits
    inputs = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=128)

    # Generate response
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    bot_response = response.split("Bot:")[-1].strip()

    # Append bot response to history
    st.session_state.conversation_history.append(f"Bot: {bot_response}")

    return bot_response

# Streamlit Interface
st.title('Dating Advice Chatbot')

# User input text area
user_input = st.text_area("Your Message", placeholder="Ask for dating advice...")

# Generate response when button is clicked
if st.button('Get Advice'):
    if user_input:
        advice = get_blenderbot_response(user_input)
        st.write("**Advice:**")
        st.write(advice)
    else:
        st.write("Please enter a message to get advice.")
