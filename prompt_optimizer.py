import streamlit as st
import openai
from transformers import pipeline

# Load a pre-trained model for generating sentence embeddings
embedder = pipeline('feature-extraction', model='sentence-transformers/all-MiniLM-L6-v2')

def get_response(prompt, api_key):
    """ Get a response from GPT-4 based on the given prompt using the specified API key. """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred: {str(e)}"

def analyze_prompt(prompt):
    """ Analyze the prompt to classify its type using heuristics. """
    if any(word in prompt.lower() for word in ['how', 'why', 'cause']):
        return "causal"
    elif any(word in prompt.lower() for word in ['compare', 'versus', 'vs']):
        return "comparative"
    elif any(word in prompt.lower() for word in ['step', 'how to', 'guide']):
        return "procedural"
    elif any(word in prompt.lower() for word in ['what will happen', 'predict', 'forecast']):
        return "predictive"
    elif any(word in prompt.lower() for word in ['describe', 'explain', 'outline']):
        return "descriptive"
    elif any(word in prompt.lower() for word in ['evaluate', 'assessment', 'judge']):
        return "evaluative"
    else:
        return "general"

def optimize_prompt(prompt):
    """ Apply tailored optimizations based on the prompt type. """
    prompt_type = analyze_prompt(prompt)
    explanation_details = ""
    if prompt_type == "causal":
        enhanced_prompt = f"{prompt} - Please explain the causes and their effects in detail."
        explanation_details = "Optimized to explore deeper causal relationships, providing a basis for targeted interventions."
    elif prompt_type == "comparative":
        enhanced_prompt = f"{prompt} - Include detailed comparisons of each option's efficiency, cost, and environmental impact."
        explanation_details = "Optimized for detailed comparison to aid strategic decisions in product development or marketing."
    elif prompt_type == "procedural":
        enhanced_prompt = f"{prompt} - Outline the steps involved in detail."
        explanation_details = "Enhanced to provide a clear, actionable guide, reducing execution risk and enhancing operational clarity."
    elif prompt_type == "predictive":
        enhanced_prompt = f"{prompt} - What predictions can be made based on current trends?"
        explanation_details = "Optimized to incorporate forecasting, helping in strategic planning and risk management."
    elif prompt_type == "descriptive":
        enhanced_prompt = f"{prompt} - Provide a comprehensive description including key stages and their significance."
        explanation_details = "Enhanced to provide a thorough understanding, foundational for informed decision-making."
    elif prompt_type == "evaluative":
        enhanced_prompt = f"{prompt} - Evaluate and provide reasoning for your judgment."
        explanation_details = "Optimized to include evaluation with justifications, crucial for validating business strategies."
    else:
        enhanced_prompt = f"{prompt} - Could you elaborate more on this topic?"
        explanation_details = "General optimization for more detailed information, promoting comprehensive understanding."

    return enhanced_prompt, explanation_details, prompt_type

# Streamlit application setup
st.title('AI-based Prompt Optimizer')
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

user_prompt = st.text_area("Enter your business-related query:", height=150)

if st.button('Optimize Prompt'):
    if api_key:
        optimized_prompt, explanation_details, prompt_type = optimize_prompt(user_prompt)
        original_response = get_response(user_prompt, api_key)
        optimized_response = get_response(optimized_prompt, api_key)

        st.write("### Suggested Optimized Prompt")
        st.write(optimized_prompt)
        st.write (f"Prompt type is {prompt_type}")

        st.write("### Response Comparison")
        st.markdown(
            f"<style>td, th {{vertical-align: top;}}</style><table style='width:100%;'><tr><td><strong>Original Prompt Response</strong></td><td><strong>Suggested Prompt Response</strong></td></tr><tr><td>{original_response}</td><td>{optimized_response}</td></tr></table>",
            unsafe_allow_html=True
        )

        st.subheader('Why is this Prompt Better?')
        st.write(explanation_details)

    else:
        st.error("Please enter a valid API key in the sidebar.")
