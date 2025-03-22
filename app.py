import streamlit as st
from deep_translator import GoogleTranslator as translate
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from gtts import gTTS
import os
import nltk
import pytesseract
from PIL import Image
import tempfile

# Download NLTK data for tokenization
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

# Function to summarize text
def summarize_text(text, sentences_count=3):
    """Summarizes the given text using LSA summarizer."""
    try:
        if not text.strip():
            return "No text to summarize."
        
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        
        if not summary:
            return "Couldn't generate summary. Text might be too short or complex."
            
        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return "Error occurred during summarization."

# Function to convert text to speech
def text_to_speech(text, lang_code):
    """Converts text to speech and returns the file path."""
    try:
        if not text.strip():
            return None
            
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            audio_path = temp_audio.name
            
        tts = gTTS(text=text, lang=lang_code)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Function to extract text from an image
def extract_text_from_image(image):
    """Extracts text from an uploaded image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(image)
        if not text.strip():
            return "No text detected in the image."
        return text
    except Exception as e:
        st.error(f"OCR error: {str(e)}")
        return "Error occurred during text extraction."

# Main Streamlit App
def main():
    st.title("File & Image Translator with Summarization & Speech")
    st.write("Upload a text file or an image, translate it, summarize the translated text, and listen to it.")

    # Add a sidebar with app information
    with st.sidebar:
        st.header("About")
        st.info(
            "This app allows you to:\n"
            "- Upload text files for translation\n"
            "- Upload images containing text for OCR extraction\n"
            "- Translate text to multiple Indian languages\n"
            "- Summarize translated content\n"
            "- Listen to summaries with text-to-speech\n"
            "- Download all processed content"
        )

    # Language options for Indian languages
    language_map = {
        "Bengali": "bn",
        "Gujarati": "gu",
        "Hindi": "hi",
        "Malayalam": "ml",
        "Marathi": "mr",
        "Punjabi": "pa",
        "Tamil": "ta",
        "Telugu": "te",
        "Urdu": "ur"
    }

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Text File", "Image"])

    with tab1:
        st.header("Upload Text File")
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"], key="text_uploader")
        
        if uploaded_file:
            try:
                # Read the file content
                file_contents = uploaded_file.read().decode("utf-8")
                st.subheader("Original Text")
                st.text_area("", file_contents, height=150)
                
                process_text(file_contents, language_map)
            except Exception as e:
                st.error(f"Error processing text file: {str(e)}")

    with tab2:
        st.header("Upload Image with Text")
        uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], key="image_uploader")
        
        if uploaded_image:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Extract text from image
                with st.spinner("Extracting text from image..."):
                    extracted_text = extract_text_from_image(image)
                
                if extracted_text and extracted_text != "No text detected in the image." and extracted_text != "Error occurred during text extraction.":
                    st.subheader("Extracted Text from Image")
                    st.text_area("", extracted_text, height=150)
                    
                    # Process the extracted text
                    process_text(extracted_text, language_map, source_is_image=True)
                else:
                    st.warning(extracted_text)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

def process_text(text, language_map, source_is_image=False):
    """Process the input text for translation, summarization, and speech."""
    # Select target language
    selected_language = st.selectbox(
        "Select target language", 
        list(language_map.keys()),
        key=f"lang_select_{'image' if source_is_image else 'text'}"
    )
    
    if not selected_language:
        return
        
    lang_code = language_map[selected_language]
    
    # Add a translate button
    translate_btn = st.button(
        "Translate", 
        key=f"translate_btn_{'image' if source_is_image else 'text'}"
    )
    
    if translate_btn:
        with st.spinner("Translating..."):
            try:
                # Translate the content
                translated_text = translate(source='auto', target=lang_code).translate(text)
                
                # Display the translated text
                st.subheader("Translated Text")
                st.text_area("", translated_text, height=200, key=f"translated_{'image' if source_is_image else 'text'}")
                
                # Summarize the translated text
                with st.spinner("Summarizing..."):
                    summary = summarize_text(translated_text)
                
                st.subheader("Summarized Text")
                st.text_area("", summary, height=150, key=f"summary_{'image' if source_is_image else 'text'}")
                
                # Convert to speech
                with st.spinner("Generating audio..."):
                    audio_path = text_to_speech(summary, lang_code)
                
                if audio_path:
                    # Provide audio playback
                    st.subheader("Listen to the Summary")
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                    
                    # Clean up the temporary file
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                
                # Download buttons
                st.subheader("Download Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    if source_is_image:
                        st.download_button(
                            "Download Extracted Text", 
                            text, 
                            file_name="extracted_text.txt", 
                            mime="text/plain"
                        )
                    
                    st.download_button(
                        "Download Translated Text", 
                        translated_text, 
                        file_name=f"translated_{selected_language}.txt", 
                        mime="text/plain"
                    )
                
                with col2:
                    st.download_button(
                        "Download Summary", 
                        summary, 
                        file_name=f"summarized_{selected_language}.txt", 
                        mime="text/plain"
                    )
                    
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
