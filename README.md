# AI Text Humanizer - Undetectable ✨

A Streamlit application that transforms AI-generated text into human-like content that can pass AI detection tools, while preserving the original meaning, formatting, and maintaining the exact word count.


## Features

- **Humanize AI Text**: Transform AI-generated content to sound naturally human-written
- **Maintain Word Count**: Preserves the exact word count from input to output
- **Preserve Structure & Formatting**: Maintains original paragraphing, bullet points, numbered lists, and topic–explanation pairs
- **Advanced Anti-Detection Techniques**: Uses sentence shuffling, synonym swaps, simulated typos, filler phrases, and more to evade AI detectors
- **Academic/Research Paper Tone**: Optionally produces output in a formal, research-paper style
- **Copy Functionality**: Easy-to-use copy buttons for both input and output text
- **Word Counter**: Displays word count for input and output text
- **Detection Avoidance**: Crafted to bypass common AI detection tools
- **User-Friendly Interface**: Clean, intuitive design with side-by-side comparison

## How It Works

This application leverages Google's Gemini 2.5 Pro model and advanced post-processing to rewrite AI-generated text with human-like characteristics:

1. **Input**: Paste your AI-generated text in the left panel (supports paragraphs, lists, and topic–explanation formats)
2. **Processing**: The app analyzes the structure, splits the text into logical sections, and rewrites each using advanced humanization techniques
3. **Output**: Get naturally human-sounding text with the same word count, meaning, and formatting as your input

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ai-humanizer.git
   cd ai-humanizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with your Google API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Paste AI-generated text** in the input field on the left (supports lists, topics, and paragraphs)
2. **Click "Humanize Text"** to process the text
3. **Review the output** on the right panel (structure and formatting will be preserved)
4. **Copy the result** using the convenient copy button
5. **Verify with detection tools** using the provided links (optional)

## Use Cases

- Academic writing assistance (research papers, essays, reports)
- Content creation and marketing materials
- Report and documentation preparation
- Creative writing enhancement

## Advanced Techniques

- **Structure Preservation**: The app detects and preserves numbered/bulleted lists, topic–explanation pairs, and paragraph breaks.
- **Chunked Humanization**: Each logical section is humanized separately to maintain clarity and flow.
- **Aggressive Anti-Detection**: Post-processing includes synonym swaps, sentence shuffling, simulated typos, filler/self-reference phrases, and more.
- **Academic Tone**: Output can be tailored to match the style of top-level research papers.

## Ethical Considerations

This tool is designed to improve the quality and readability of AI-generated content, not to enable deception or plagiarism. Users are responsible for:

- Following academic integrity policies
- Proper attribution of sources
- Ethical use of the tool for legitimate purposes

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python library
- Python-dotenv
- nltk (for advanced sentence processing)

## License

[MIT License](LICENSE)

## Acknowledgements

- Powered by Google's Gemini 2.5 Pro model
- Built with Streamlit framework

---

Created with ❤️ by JAI
