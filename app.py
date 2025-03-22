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
import uuid
import datetime
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Download NLTK data for tokenization
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

# Initialize Firebase
@st.cache_resource
def initialize_firebase():
    try:
        # Check if the app is already initialized
        if not firebase_admin._apps:
            # Get Firebase credentials from environment variables
            firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
            
            if firebase_creds:
                # Load credentials from environment variable (JSON string)
                cred_dict = json.loads(firebase_creds)
                cred = credentials.Certificate(cred_dict)
            else:
                # Fallback to file if environment variable is not set
                cred = credentials.Certificate("firebase-credentials.json")
                
            firebase_admin.initialize_app(cred)
            
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase initialization error: {str(e)}")
        return None

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
        # Tesseract is installed system-wide on Replit
        # No need to set the path explicitly as in Windows
        text = pytesseract.image_to_string(image)
        if not text.strip():
            return "No text detected in the image."
        return text
    except Exception as e:
        st.error(f"OCR error: {str(e)}")
        return "Error occurred during text extraction."

# Function to save data to Firebase
def save_to_firebase(data):
    """Saves the processed data to Firebase Firestore."""
    try:
        # Initialize Firebase
        db = initialize_firebase()
        if not db:
            st.warning("Firebase connection not available. Data will not be saved.")
            return False
            
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Add timestamp
        data['timestamp'] = datetime.datetime.now().isoformat()
        
        # Save to Firestore
        db.collection('translations').document(doc_id).set(data)
        
        return True
    except Exception as e:
        st.error(f"Firebase save error: {str(e)}")
        return False

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
    tab1, tab2, tab3 = st.tabs(["Text File", "Image", "History"])

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
                
    with tab3:
        st.header("Translation History")
        st.write("View your past translations stored in the database.")
        
        # Get Firebase client
        db = initialize_firebase()
        
        if db:
            try:
                # Create filter options
                st.subheader("Filter Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    filter_type = st.selectbox(
                        "Filter by source type",
                        ["All", "Text File", "Image"],
                        key="history_filter_type"
                    )
                
                with col2:
                    filter_language = st.selectbox(
                        "Filter by language",
                        ["All"] + list(language_map.keys()),
                        key="history_filter_language"
                    )
                
                # Add refresh button
                if st.button("Refresh History", key="refresh_history"):
                    st.rerun()
                
                # Query Firestore
                with st.spinner("Loading history..."):
                    # Get collection reference
                    translations_ref = db.collection('translations')
                    
                    # Apply filters if selected
                    if filter_type != "All" and filter_language != "All":
                        query = translations_ref.where(
                            "source_type", "==", "text_file" if filter_type == "Text File" else "image"
                        ).where("target_language", "==", filter_language)
                    elif filter_type != "All":
                        query = translations_ref.where(
                            "source_type", "==", "text_file" if filter_type == "Text File" else "image"
                        )
                    elif filter_language != "All":
                        query = translations_ref.where("target_language", "==", filter_language)
                    else:
                        query = translations_ref
                    
                    # Order by timestamp (most recent first)
                    translations = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).get()
                    
                    if not translations:
                        st.info("No translation history found. Translate some text to see it here!")
                    else:
                        # Display results
                        for i, doc in enumerate(translations):
                            data = doc.to_dict()
                            
                            # Create an expander for each record
                            with st.expander(f"Translation {i+1} - {data.get('target_language', 'Unknown')} ({data.get('timestamp', 'Unknown date')})"):
                                st.markdown(f"**Source Type:** {'Image' if data.get('source_type') == 'image' else 'Text File'}")
                                st.markdown(f"**Target Language:** {data.get('target_language', 'Unknown')}")
                                
                                # Original text
                                st.subheader("Original Text")
                                st.text_area("", data.get("original_text", ""), height=100, key=f"history_original_{i}")
                                
                                # Translated text
                                st.subheader("Translated Text")
                                st.text_area("", data.get("translated_text", ""), height=100, key=f"history_translated_{i}")
                                
                                # Summary
                                st.subheader("Summary")
                                st.text_area("", data.get("summary", ""), height=100, key=f"history_summary_{i}")
                                
                                # Download buttons
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.download_button(
                                        "Download Original", 
                                        data.get("original_text", ""), 
                                        file_name=f"original_{i+1}.txt", 
                                        mime="text/plain",
                                        key=f"dl_orig_{i}"
                                    )
                                
                                with col2:
                                    st.download_button(
                                        "Download Translation", 
                                        data.get("translated_text", ""), 
                                        file_name=f"translated_{i+1}.txt", 
                                        mime="text/plain",
                                        key=f"dl_trans_{i}"
                                    )
                                
                                with col3:
                                    st.download_button(
                                        "Download Summary", 
                                        data.get("summary", ""), 
                                        file_name=f"summary_{i+1}.txt", 
                                        mime="text/plain",
                                        key=f"dl_summary_{i}"
                                    )
            except Exception as e:
                st.error(f"Error loading history: {str(e)}")
        else:
            st.warning("Firebase connection not available. Unable to display history.")

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
                
                # Store data in Firebase
                with st.spinner("Saving to database..."):
                    # Prepare data for Firebase
                    firebase_data = {
                        "original_text": text,
                        "source_type": "image" if source_is_image else "text_file",
                        "target_language": selected_language,
                        "target_language_code": lang_code,
                        "translated_text": translated_text,
                        "summary": summary,
                        "has_audio": audio_path is not None
                    }
                    
                    # Save to Firebase
                    saved = save_to_firebase(firebase_data)
                    if saved:
                        st.success("Data saved to Firebase successfully!")
                    
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
