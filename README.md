# AI Text Humanizer - Undetectable ✨

A Streamlit application that transforms AI-generated text into human-like content that can pass AI detection tools, while preserving the original meaning and maintaining the exact word count.

![AI Humanizer App Screenshot](https://via.placeholder.com/800x400?text=AI+Humanizer+App)

## Features

- **Humanize AI Text**: Transform AI-generated content to sound naturally human-written
- **Maintain Word Count**: Preserves the exact word count from input to output
- **Copy Functionality**: Easy-to-use copy buttons for both input and output text
- **Word Counter**: Displays word count for input and output text
- **Detection Avoidance**: Crafted to bypass common AI detection tools
- **User-Friendly Interface**: Clean, intuitive design with side-by-side comparison

## How It Works

This application leverages Google's Gemini 2.5 Pro model to rewrite AI-generated text with human-like characteristics:

1. **Input**: Paste your AI-generated text in the left panel
2. **Processing**: The app analyzes the text and rewrites it using advanced humanization techniques
3. **Output**: Get naturally human-sounding text with the same word count and meaning

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

1. **Paste AI-generated text** in the input field on the left
2. **Click "Humanize Text"** to process the text
3. **Review the output** on the right panel
4. **Copy the result** using the convenient copy button
5. **Verify with detection tools** using the provided links (optional)

## Use Cases

- Academic writing assistance
- Content creation and marketing materials
- Report and documentation preparation
- Creative writing enhancement

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

## License

[MIT License](LICENSE)

## Acknowledgements

- Powered by Google's Gemini 2.5 Pro model
- Built with Streamlit framework

---

Created with ❤️ by [Your Name]
