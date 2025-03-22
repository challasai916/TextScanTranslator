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

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\chall\Desktop\tesseract-ocr-w64-setup-5.5.0.20241111.exe"

nltk.download('punkt')

# Function to summarize text
def summarize_text(text, sentences_count=3):
    """Summarizes the given text using LSA summarizer."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# Function to convert text to speech
def text_to_speech(text, lang_code):
    """Converts text to speech and returns the file path."""
    tts = gTTS(text=text, lang=lang_code)
    audio_path = "translated_speech.mp3"
    tts.save(audio_path)
    return audio_path

# Function to extract text from an image
def extract_text_from_image(image):
    """Extracts text from an uploaded image using Tesseract OCR."""
    return pytesseract.image_to_string(image)

# Main Streamlit App
def main():
    st.title("File & Image Translator with Summarization & Speech")
    st.write("Upload a text file or an image, translate it, summarize the translated text, and listen to it.")

    # File uploader for text files
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    # Image uploader
    uploaded_image = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

    # Language options
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

    selected_language = st.selectbox("Select target language", list(language_map.keys()))

    if selected_language:
        lang_code = language_map[selected_language]

        if uploaded_file:
            # Read the file content
            file_contents = uploaded_file.read().decode("utf-8")

            # Translate the content
            translated_text = translate(source='auto', target=lang_code).translate(file_contents)

            # Display the translated text
            st.subheader("Translated Text")
            st.text_area("", translated_text, height=200)

            # Summarize the translated text
            summary = summarize_text(translated_text)
            st.subheader("Summarized Text")
            st.text_area("", summary, height=150)

            # Convert to speech
            audio_path = text_to_speech(summary, lang_code)

            # Provide audio playback
            st.subheader("Listen to the Summary")
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")

            # Download buttons
            st.download_button("Download Translated File", translated_text, file_name=f"translated_{selected_language}.txt", mime="text/plain")
            st.download_button("Download Summarized File", summary, file_name=f"summarized_{selected_language}.txt", mime="text/plain")

        elif uploaded_image:
            # Open and process the image
            image = Image.open(uploaded_image)
            extracted_text = extract_text_from_image(image)

            if extracted_text.strip():
                st.subheader("Extracted Text from Image")
                st.text_area("", extracted_text, height=150)

                # Translate the extracted text
                translated_text = translate(source='auto', target=lang_code).translate(extracted_text)

                # Display the translated text
                st.subheader("Translated Text")
                st.text_area("", translated_text, height=200)

                # Summarize the translated text
                summary = summarize_text(translated_text)
                st.subheader("Summarized Text")
                st.text_area("", summary, height=150)

                # Convert to speech
                audio_path = text_to_speech(summary, lang_code)

                # Provide audio playback
                st.subheader("Listen to the Summary")
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")

                # Download buttons
                st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.txt", mime="text/plain")
                st.download_button("Download Translated File", translated_text, file_name=f"translated_{selected_language}.txt", mime="text/plain")
                st.download_button("Download Summarized File", summary, file_name=f"summarized_{selected_language}.txt", mime="text/plain")

            else:
                st.error("No text detected in the image. Please upload a clear image with readable text.")

if __name__ == "__main__":
    main()
