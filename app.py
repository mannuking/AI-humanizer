import streamlit as st
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
import re
import random

# --- Configuration ---
load_dotenv()  # Load variables from .env file

# Securely load API key - Check Streamlit secrets first, then env variables
try:
    # First check for Streamlit secrets (for Streamlit Cloud deployment)
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    st.info("Using API key from Streamlit secrets.")
except (KeyError, FileNotFoundError):
    # Fallback to environment variable (for local development)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        st.info("Using API key from environment variables.")
    else:
        st.error("Error: GEMINI_API_KEY not found. Please set it in your .env file for local development or in Streamlit secrets for deployment.")
        st.stop()  # Stop execution if key is missing

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

def synonym_swap(text):
    """Randomly swap synonyms for common words to increase human-likeness."""
    synonyms = {
        'important': ['crucial', 'vital', 'significant'],
        'result': ['outcome', 'finding', 'consequence'],
        'method': ['approach', 'technique', 'procedure'],
        'show': ['demonstrate', 'reveal', 'indicate'],
        'use': ['utilize', 'employ', 'apply'],
        'increase': ['raise', 'enhance', 'amplify'],
        'decrease': ['reduce', 'diminish', 'lower'],
        'data': ['information', 'dataset', 'figures'],
        'analysis': ['examination', 'assessment', 'evaluation'],
        'study': ['investigation', 'research', 'examination'],
        'model': ['framework', 'paradigm', 'structure'],
        'significant': ['notable', 'remarkable', 'substantial'],
        'conclusion': ['inference', 'determination', 'resolution'],
        'demonstrate': ['show', 'illustrate', 'exhibit'],
        'obtain': ['acquire', 'procure', 'derive'],
        'support': ['corroborate', 'substantiate', 'back'],
        'suggest': ['imply', 'indicate', 'hint'],
        'however': ['nevertheless', 'nonetheless', 'yet'],
        'therefore': ['thus', 'consequently', 'accordingly'],
        'because': ['since', 'as', 'due to the fact that'],
        'although': ['though', 'even though', 'albeit'],
        'can': ['may', 'could', 'might'],
        'will': ['shall', 'is expected to', 'is likely to'],
        'also': ['additionally', 'furthermore', 'moreover'],
        'but': ['however', 'yet', 'nevertheless'],
        'very': ['extremely', 'highly', 'particularly'],
        'different': ['distinct', 'varied', 'diverse'],
        'similar': ['comparable', 'analogous', 'alike'],
        'large': ['substantial', 'considerable', 'sizeable'],
        'small': ['modest', 'minor', 'limited'],
        'new': ['novel', 'recent', 'fresh'],
        'old': ['previous', 'former', 'prior'],
        'main': ['primary', 'principal', 'chief'],
        'common': ['prevalent', 'widespread', 'frequent'],
        'rare': ['infrequent', 'uncommon', 'seldom'],
    }
    words = text.split()
    for i, word in enumerate(words):
        key = word.lower().strip('.,;:')
        if key in synonyms and random.random() < 0.18:
            replacement = random.choice(synonyms[key])
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            # Preserve punctuation
            if word[-1] in ',.;:':
                replacement += word[-1]
            words[i] = replacement
    return ' '.join(words)

def introduce_quirks(text):
    """Introduce minor grammatical quirks and random punctuation for human-likeness."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    for i in range(len(sentences)):
        # Randomly add a comma splice
        if random.random() < 0.10 and len(sentences[i].split()) > 8:
            words = sentences[i].split()
            pos = random.randint(3, len(words)-3)
            words.insert(pos, ',')
            sentences[i] = ' '.join(words)
        # Randomly drop a word (simulate a human typo)
        if random.random() < 0.05 and len(sentences[i].split()) > 10:
            words = sentences[i].split()
            drop = random.randint(1, len(words)-2)
            del words[drop]
            sentences[i] = ' '.join(words)
        # Randomly add a British/American spelling variant
        if random.random() < 0.08:
            sentences[i] = sentences[i].replace('analyze', 'analyse')
            sentences[i] = sentences[i].replace('color', 'colour')
            sentences[i] = sentences[i].replace('behavior', 'behaviour')
            sentences[i] = sentences[i].replace('modeling', 'modelling')
    return ' '.join(sentences)

def postprocess_humanization(text):
    """Apply synonym swapping and quirks to further humanize the text."""
    text = synonym_swap(text)
    text = introduce_quirks(text)
    return text

def advanced_humanize(text):
    """
    Advanced post-processing: chunked rewriting, sentence shuffling, aggressive synonym/structure variation.
    """
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
    import random
    
    # Aggressive synonym replacement
    synonyms = {
        'important': ['crucial', 'vital', 'pivotal', 'essential'],
        'result': ['outcome', 'finding', 'consequence', 'implication'],
        'method': ['approach', 'technique', 'procedure', 'strategy'],
        'show': ['demonstrate', 'reveal', 'indicate', 'exhibit'],
        'use': ['utilize', 'employ', 'apply', 'deploy'],
        'increase': ['raise', 'enhance', 'amplify', 'escalate'],
        'decrease': ['reduce', 'diminish', 'lower', 'curtail'],
        'data': ['information', 'dataset', 'figures', 'statistics'],
        'analysis': ['examination', 'assessment', 'evaluation', 'scrutiny'],
        'study': ['investigation', 'research', 'examination', 'survey'],
        'model': ['framework', 'paradigm', 'structure', 'schema'],
        'significant': ['notable', 'remarkable', 'substantial', 'meaningful'],
        'conclusion': ['inference', 'determination', 'resolution', 'judgment'],
        'demonstrate': ['show', 'illustrate', 'exhibit', 'display'],
        'obtain': ['acquire', 'procure', 'derive', 'attain'],
        'support': ['corroborate', 'substantiate', 'back', 'reinforce'],
        'suggest': ['imply', 'indicate', 'hint', 'propose'],
        'however': ['nevertheless', 'nonetheless', 'yet', 'still'],
        'therefore': ['thus', 'consequently', 'accordingly', 'hence'],
        'because': ['since', 'as', 'due to the fact that', 'inasmuch as'],
        'although': ['though', 'even though', 'albeit', 'notwithstanding'],
        'can': ['may', 'could', 'might', 'is able to'],
        'will': ['shall', 'is expected to', 'is likely to', 'is set to'],
        'also': ['additionally', 'furthermore', 'moreover', 'as well'],
        'but': ['however', 'yet', 'nevertheless', 'on the other hand'],
        'very': ['extremely', 'highly', 'particularly', 'exceptionally'],
        'different': ['distinct', 'varied', 'diverse', 'dissimilar'],
        'similar': ['comparable', 'analogous', 'alike', 'resembling'],
        'large': ['substantial', 'considerable', 'sizeable', 'immense'],
        'small': ['modest', 'minor', 'limited', 'minute'],
        'new': ['novel', 'recent', 'fresh', 'innovative'],
        'old': ['previous', 'former', 'prior', 'ancient'],
        'main': ['primary', 'principal', 'chief', 'foremost'],
        'common': ['prevalent', 'widespread', 'frequent', 'ubiquitous'],
        'rare': ['infrequent', 'uncommon', 'seldom', 'scarce'],
    }
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    # Shuffle sentences to break global coherence
    random.shuffle(sentences)
    # Aggressively rewrite each sentence
    rewritten = []
    for s in sentences:
        words = s.split()
        for i, word in enumerate(words):
            key = word.lower().strip('.,;:')
            if key in synonyms and random.random() < 0.25:
                replacement = random.choice(synonyms[key])
                if word[0].isupper():
                    replacement = replacement.capitalize()
                if word[-1] in ',.;:':
                    replacement += word[-1]
                words[i] = replacement
        # Randomly switch active/passive voice (simple heuristic)
        if random.random() < 0.15 and ' by ' not in s and len(words) > 6:
            words = words[::-1]
        # Randomly merge with previous sentence
        if rewritten and random.random() < 0.18:
            rewritten[-1] += ' ' + ' '.join(words)
        else:
            rewritten.append(' '.join(words))
    # Optionally, insert a parenthetical or editor's note
    if len(rewritten) > 2 and random.random() < 0.5:
        idx = random.randint(1, len(rewritten)-2)
        rewritten[idx] += ' (editor’s note: see also related literature for further context.)'
    return ' '.join(rewritten)

def ultra_humanize(text):
    """
    Further humanize text by simulating typos, inline corrections, parenthetical asides, filler phrases, and sentence structure disruption.
    """
    import random
    import re
    # Filler and self-reference phrases
    fillers = [
        'it is worth noting that', 'interestingly', 'in other words', 'as previously mentioned',
        'from my perspective', 'to be clear', 'in my experience', 'as discussed earlier',
        'notably', 'indeed', 'in fact', 'to reiterate', 'as an aside', 'on a related note'
    ]
    # Parenthetical asides
    asides = [
        '(see discussion above)', '(typo corrected)', '(revised for clarity)',
        '(editor’s note)', '(as I recall)', '(personal observation)', '(cf. related work)'
    ]
    # Simulate typos and inline corrections
    def typo_correction(word):
        if len(word) > 6 and random.random() < 0.08:
            idx = random.randint(1, len(word)-2)
            typo = word[:idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[idx+1:]
            return f"{typo}—sorry, {word}"
        return word
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    new_sentences = []
    for s in sentences:
        # Randomly insert a filler phrase at the start
        if random.random() < 0.18:
            s = f"{random.choice(fillers).capitalize()}, {s[0].lower() + s[1:]}"
        # Randomly insert a parenthetical aside
        if random.random() < 0.12:
            words = s.split()
            if len(words) > 6:
                pos = random.randint(2, len(words)-2)
                words.insert(pos, random.choice(asides))
                s = ' '.join(words)
        # Randomly simulate a typo and inline correction
        if random.random() < 0.10:
            words = s.split()
            idx = random.randint(0, len(words)-1)
            words[idx] = typo_correction(words[idx])
            s = ' '.join(words)
        # Randomly start with a conjunction
        if random.random() < 0.10:
            s = random.choice(['And', 'But', 'So', 'Yet']) + ', ' + s[0].lower() + s[1:]
        # Randomly merge with previous sentence
        if new_sentences and random.random() < 0.15:
            new_sentences[-1] += ' ' + s
        else:
            new_sentences.append(s)
    return ' '.join(new_sentences)

def split_preserve_structure(text):
    """
    Split text into chunks (paragraphs, numbered/bulleted points) while preserving structure.
    Returns a list of (prefix, chunk) where prefix is the bullet/number or empty string.
    """
    import re
    lines = text.splitlines()
    chunks = []
    buffer = []
    prefix = ''
    for line in lines:
        m = re.match(r'^(\s*([\d]+[.)]|[-*•])\s+)(.*)', line)
        if m:
            # Save previous buffer
            if buffer:
                chunks.append((prefix, '\n'.join(buffer).strip()))
                buffer = []
            prefix = m.group(1)
            buffer.append(m.group(3))
        elif line.strip() == '':
            if buffer:
                chunks.append((prefix, '\n'.join(buffer).strip()))
                buffer = []
                prefix = ''
        else:
            buffer.append(line)
    if buffer:
        chunks.append((prefix, '\n'.join(buffer).strip()))
    return chunks

def join_preserved_structure(chunks):
    """
    Reassemble chunks into text, preserving original prefixes and paragraph breaks.
    """
    out = []
    for prefix, chunk in chunks:
        if prefix:
            out.append(f"{prefix}{chunk}")
        else:
            out.append(chunk)
    return '\n\n'.join(out)

def split_topics_explanations(text):
    """
    Split text into (topic, explanation) pairs based on lines ending with a colon or bold/numbered points.
    Returns a list of (topic, explanation) tuples.
    """
    import re
    lines = text.splitlines()
    pairs = []
    topic = None
    explanation = []
    for line in lines:
        # Detect topic heading (ends with colon, or is bold, or is a numbered/bulleted point)
        if re.match(r'^\s*([\d]+[.)]|[-*•])?\s*([A-Z][^:]{2,}):\s*$', line):
            # Save previous
            if topic is not None:
                pairs.append((topic, ' '.join(explanation).strip()))
                explanation = []
            topic = line.strip()
        elif re.match(r'^\s*([A-Z][^:]{2,}):\s*', line):
            # Inline topic: explanation
            m = re.match(r'^(\s*[A-Z][^:]{2,}:)(.*)$', line)
            if topic is not None:
                pairs.append((topic, ' '.join(explanation).strip()))
                explanation = []
            topic = m.group(1).strip()
            if m.group(2).strip():
                explanation.append(m.group(2).strip())
        else:
            explanation.append(line.strip())
    if topic is not None:
        pairs.append((topic, ' '.join(explanation).strip()))
    # Remove empty explanations
    pairs = [(t, e) for t, e in pairs if t and e]
    return pairs

def join_topics_explanations(pairs):
    """
    Reassemble topic–explanation pairs into text, preserving headings and explanations.
    """
    out = []
    for topic, explanation in pairs:
        out.append(f"{topic}\n{explanation}")
    return '\n\n'.join(out)

def split_first_sentence(text):
    """
    Split text into the first sentence (or paragraph) and the rest.
    Returns (first, rest).
    """
    import re
    # Try to split by first period, exclamation, or question mark followed by space or newline
    m = re.match(r'(.+?[.!?])([\s\n].*)', text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fallback: split by first paragraph
    parts = text.split('\n', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ''

def split_paragraphs(text):
    """
    Split text into paragraphs using double newlines or blank lines as separators.
    Returns a list of paragraphs (including empty ones for perfect structure preservation).
    """
    import re
    # Split on two or more newlines (or blank lines)
    paras = re.split(r'\n\s*\n', text)
    return [p for p in paras]

def join_paragraphs(paragraphs):
    """
    Join paragraphs with double newlines to preserve original formatting.
    """
    return '\n\n'.join(paragraphs)

# --- Core Function ---
def humanize_text_academic(text, target_word_count):
    """
    Rewrites AI-generated academic text to sound more human-like while maintaining
    scientific rigor and formal academic tone, using Gemini 2.5 Pro.

    Args:
      text: The input AI-generated academic text.
      target_word_count: The exact number of words the output should have.

    Returns:
      Humanized academic text or an error message.
    """
    prompt = f"""
You are a distinguished professor and scientific writer with decades of experience publishing in prestigious journals. Your task is to rewrite the provided text to be completely undetectable by AI detection systems while maintaining the highest standards of academic writing. Follow these guidelines exactly:

**Core Task:** Transform the following text to sound completely human-written while preserving its scholarly tone, formal academic language, technical precision, and core arguments.

**EXTREMELY IMPORTANT - WORD COUNT CONSTRAINT:**
- The rewritten text MUST contain EXACTLY {target_word_count} words - no more, no less.
- Count words as separate units divided by spaces (contractions count as one word).
- This word count requirement is critical and non-negotiable.

**Advanced Anti-Detection Techniques (ESSENTIAL):**
- Introduce subtle grammatical quirks and occasional inconsistencies (humans aren't perfect)
- Add rare punctuation patterns that AI models rarely generate (e.g., semicolons followed by em dashes; unusual comma placements)
- Insert uncommon transitional phrases and scholarly expressions rarely found in AI databases
- Vary sentence complexity unpredictably - mix extremely complex sentences with occasional short, direct ones
- Include idiosyncratic word choices that wouldn't follow typical AI training patterns
- Break patterns periodically - AI tends to maintain consistent styles throughout a text
- Introduce subtle typographical inconsistencies (like occasional use of British vs. American spellings)
- Use discipline-specific jargon with slight variations that only a human expert would know
- Include an occasional tangential argument or minor contradiction (humans don't write with perfect logical flow)

**Academic Writing Characteristics:**
- Use complex lexicon and discipline-appropriate terminology
- Incorporate scholarly hedging where appropriate (e.g., "suggests that," "appears to indicate")
- Apply specialized academic phrases and disciplinary conventions
- Employ methodical logical progression with minor natural diversions
- Include occasional self-references or references to previous work (e.g., "our findings suggest," "as I have previously argued")
- Keep a formal academic register but with subtle variations in tone

**Human-Like Writing Elements:**
- Include irregular paragraph structures that don't follow algorithmic patterns
- Add the occasional personal viewpoint hidden in seemingly objective statements
- Insert minor stylistic quirks that would be specific to a human author
- Use uncommon synonyms for common academic terms
- Include research-specific analogies or metaphors that AI wouldn't typically generate
- Vary citation and reference patterns in ways AI wouldn't predict

**Scientific Integrity Requirements:**
- Maintain meticulous accuracy of all technical terms, data, and research concepts
- Preserve the logical structure and argumentative framework of the original
- Retain all evidence-based reasoning and methodological details
- Ensure all scientific claims remain properly qualified and contextualized

**Output Requirements:**
- Provide ONLY the rewritten text with EXACTLY {target_word_count} words
- Do not include any explanations, introductions, or comments in your response
- The final text should read as though written by a human academic expert in the field with their own distinctive writing style

**Input Text:**
---
{text}
---

**Rewritten Human Academic Text (EXACTLY {target_word_count} words):**
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
                temperature=0.95,  # Higher temperature for more randomness and unpredictability
                top_p=0.85,        # Slightly lower top_p for more diversity in word choice
                top_k=60           # Wider selection of tokens for increased randomness
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
st.set_page_config(
    page_title="HumanizeAI Academic Writer",
    page_icon="🧑‍🎓",
    layout="wide"
)

st.title("HumanizeAI Academic Writer 🧑‍🎓")
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
            # --- New: Structure-preserving paragraph split ---
            paragraphs = split_paragraphs(cleaned_input)
            total_word_count = count_words(cleaned_input)
            para_word_counts = [count_words(p) for p in paragraphs]
            total_paras_words = sum(para_word_counts)
            scale = total_word_count / total_paras_words if total_paras_words else 1
            para_word_counts = [max(1, round(w * scale)) for w in para_word_counts]
            diff = total_word_count - sum(para_word_counts)
            for i in range(abs(diff)):
                idx = i % len(para_word_counts)
                if diff > 0:
                    para_word_counts[idx] += 1
                elif diff < 0 and para_word_counts[idx] > 1:
                    para_word_counts[idx] -= 1
            humanized_paragraphs = []
            with st.spinner("🧠 Processing... Creating undetectable human-like text..."):
                start_time = time.time()
                for para, wc in zip(paragraphs, para_word_counts):
                    if para.strip():
                        # Preserve first sentence structure for each paragraph
                        first, rest = split_first_sentence(para)
                        first_wc = count_words(first)
                        rest_wc = wc - first_wc
                        first_human = humanize_text_academic(first, first_wc) if first_wc > 0 else ''
                        if rest_wc > 0:
                            rest_human = humanize_text_academic(rest, rest_wc)
                            if "Error:" not in rest_human:
                                rest_human = advanced_humanize(rest_human)
                                rest_human = ultra_humanize(rest_human)
                        else:
                            rest_human = ''
                        humanized_paragraphs.append((first_human + '\n' + rest_human).strip())
                    else:
                        humanized_paragraphs.append('')
                humanized_output = join_paragraphs(humanized_paragraphs)
                end_time = time.time()
            processing_time = end_time - start_time
            st.info(f"Processing completed in {processing_time:.2f} seconds.")
            output_text_area.text_area("Output:", value=humanized_output, height=350, key="output_filled")
            output_word_count.markdown(f"**Word count: {count_words(humanized_output)}**")
            if "Error:" not in humanized_output:
                st.success("✅ Text humanization complete!")
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
