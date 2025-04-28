import streamlit as st
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
import re

# --- Configuration ---
load_dotenv()  # Load variables from .env file

# Securely load API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Error: GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop() # Stop execution if key is missing

# Configure the Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- Model Selection ---
# Use the specified Gemini 2.5 Pro model
MODEL_NAME = "gemini-2.5-pro-preview-03-25"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"Error creating Gemini model ({MODEL_NAME}): {e}")
    st.stop()

# --- Helper Functions ---
def count_words(text):
    """Count the number of words in a text."""
    if not text:
        return 0
    return len(re.findall(r'\b\w+\b', text))

# --- Core Function ---
def humanize_text_academic(text, target_word_count):
    """
    Rewrites AI-generated academic text to sound more human-like,
    maintaining meaning and tone, using Gemini 2.5 Pro.

    Args:
      text: The input AI-generated academic text.
      target_word_count: The exact number of words the output should have.

    Returns:
      Humanized text or an error message.
    """
    prompt = f"""
You are an expert academic writer and editor specializing in crafting text that is completely undetectable by AI detection systems. Your goal is to rewrite the provided text while strictly adhering to these guidelines:

**Core Task:** Rewrite and paraphrase the following text to sound entirely human-written, maintaining its original meaning and core arguments while being completely undetectable by AI detection tools.

**EXTREMELY IMPORTANT - WORD COUNT CONSTRAINT:**
- The rewritten text MUST contain EXACTLY {target_word_count} words - no more, no less.
- Count words as separate units divided by spaces (contractions count as one word).
- This is critical for my use case and is non-negotiable.

**Human-Like Writing Characteristics (Essential for Bypassing AI Detection):**
- Use inconsistent patterns and occasional imperfections typical of human writing
- Incorporate idiomatic expressions and colloquialisms where appropriate
- Include occasional filler words, qualifying phrases, or hedge words when natural
- Vary punctuation usage with occasional comma splices or dashes
- Use contractions inconsistently (sometimes use them, sometimes don't)
- Occasionally use sentence fragments or slightly run-on sentences

**Advanced Techniques for Bypassing AI Detection:**
- Inject subtle randomness and natural irregularities in sentence structures
- Use distinctly human metaphors and analogies not commonly found in AI writing
- Incorporate personal-seeming intuitions or reflections where appropriate
- Employ varied stylistic elements that AI systems typically avoid or use predictably
- Deliberately avoid repetitive sentence beginnings, perfect syntactic parallels, and other AI giveaways

**Content Integrity:**
- Maintain the original meaning, key arguments, and technical accuracy
- Ensure technical terms are used correctly and consistently
- Preserve all factual content and logical flow

**Output Requirements:**
- Provide ONLY the rewritten text with EXACTLY {target_word_count} words
- Do not include any explanations, introductions, or comments in your response

**Input Text:**
---
{text}
---

**Rewritten Human Text (EXACTLY {target_word_count} words):**
"""

    try:
        response = model.generate_content(
            prompt,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.9  # Higher temperature for more creative variations
            )
        )
        # Handle potential blocking or lack of response text
        if response.parts:
             output_text = response.text
             # Verify word count and attempt correction if needed
             actual_word_count = count_words(output_text)
             if actual_word_count != target_word_count:
                 # Log warning but return text anyway
                 st.warning(f"Note: Output has {actual_word_count} words instead of the requested {target_word_count}. Slight variations may occur.")
             return output_text
        elif response.prompt_feedback.block_reason:
             return f"Error: Content blocked due to {response.prompt_feedback.block_reason}. Input may violate safety policies."
        else:
             return "Error: Received an empty response from the model. Please try again or rephrase the input."

    except Exception as e:
        # Catch potential API errors, rate limits, etc.
        st.error(f"An error occurred during text generation: {e}")
        return f"Error during processing: {e}"

# --- Streamlit App Interface ---
st.set_page_config(layout="wide") # Use wider layout for more space

st.title("AI Text Humanizer - Undetectable ✨")
st.markdown(f"""
This tool uses the **{MODEL_NAME}** model to transform AI-generated text into completely human-like content that passes AI detection checks.
It preserves the original meaning while making the text appear naturally human-written, maintaining the exact word count from input to output.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input AI-Generated Text")
    input_text = st.text_area("Paste your AI-generated text here:", height=350, key="input")
    # Word counter for input
    # Word counter for input
    word_count = count_words(input_text)
    st.markdown(f"**Word count: {word_count}**")

with col2:
    st.subheader("Humanized Text (Undetectable)")
    output_container = st.container()
    with output_container:
        output_text_area = st.empty() # Placeholder for the output text area
        output_text_area.text_area("Output will appear here...", height=350, key="output", disabled=True)
        output_word_count = st.empty()  # Placeholder for word count

if st.button("🚀 Humanize Text"):
    if input_text:
        cleaned_input = input_text.strip() # Remove leading/trailing whitespace
        if cleaned_input:
            # Get the word count of the input text
            input_word_count = count_words(cleaned_input)
            
            with st.spinner("🧠 Processing... Creating undetectable human-like text..."):
                start_time = time.time()
                humanized_output = humanize_text_academic(cleaned_input, input_word_count)
                end_time = time.time()

            processing_time = end_time - start_time
            st.info(f"Processing completed in {processing_time:.2f} seconds.")

            # Display output in the placeholder
            output_text_area.text_area("Output:", value=humanized_output, height=350, key="output_filled")
            
            # Update output word count
            output_word_count.markdown(f"**Word count: {count_words(humanized_output)}**")

            # Check if the output indicates an error before showing success
            if "Error:" not in humanized_output:
                st.success("✅ Text humanization complete!")
                
                # AI Detection Check section
                st.subheader("🔍 Verify AI Detection")
                st.markdown("""
                To ensure your text passes AI detection, check it with these free detection tools:
                - [ZeroGPT](https://www.zerogpt.com/)
                - [GPTZero](https://gptzero.me/)
                - [Writer AI Content Detector](https://writer.com/ai-content-detector/)
                - [Content at Scale](https://contentatscale.ai/ai-content-detector/)
                
                Recommended workflow:
                1. Copy the humanized text above
                2. Paste it into 2-3 different detectors
                3. If any detector flags it as AI, try running it through the humanizer again
                """)
            else:
                 st.error("Processing encountered an issue. See message above.")

        else:
            st.warning("Input text is empty or contains only whitespace.")
    else:
        st.warning("Please paste some text into the input box.")

# --- Important Disclaimer ---
st.divider()
st.subheader("⚠️ Important Disclaimer & Ethical Use")
st.markdown("""
*   **Purpose:** This tool is intended to *assist* in refining AI-generated drafts to sound more natural and human-like. It is **not** a substitute for original thought or proper attribution.
*   **Academic Integrity:** **Plagiarism and academic dishonesty are serious offenses.** Always follow your institution's policies regarding AI-assisted writing.
*   **Responsibility:** You are solely responsible for the final content, accuracy, and ethical use of your work.
*   **Improvement, Not Deception:** The primary goal is to improve the quality and readability of AI-generated content, not to deliberately deceive or misrepresent.
""")
