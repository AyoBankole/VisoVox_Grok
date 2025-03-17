import os
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from gtts import gTTS
import streamlit as st
import sounddevice as sd
import numpy as np
import speech_recognition as sr

# NEW: Voice command listener using sounddevice (defined at the top so it is available)
def listen_voice_command():
    """Listens for a voice command using sounddevice and returns the recognized text."""
    duration = 5  # seconds to record
    fs = 44100   # sample rate
    st.info("Listening for voice command (using sounddevice)...")
    try:
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        audio_bytes = audio_data.tobytes()
        # Create an AudioData instance compatible with speech_recognition
        audio = sr.AudioData(audio_bytes, fs, 2)
        recognizer = sr.Recognizer()
        command = recognizer.recognize_google(audio)
        st.success(f"Voice command: {command}")
        return command.lower()
    except Exception as e:
        st.error("Voice command not recognized.")
        return ""

# NEW: Function to trigger voice-controlled actions
def handle_voice_control():
    """Handles voice commands to control the app."""
    command = listen_voice_command()
    # Control webcam vs. upload
    if "capture" in command or "take picture" in command:
        st.session_state["voice_capture"] = True
        st.success("Voice command: Using webcam for image capture.")
    elif "upload" in command:
        st.session_state["voice_capture"] = False
        st.success("Voice command: Set to upload an image.")
    # Voice question handling
    elif "ask" in command or "question" in command:
        st.session_state["voice_question"] = True
        st.success("Voice command: Ready to ask your question.")
    # Read out caption immediately if commanded
    elif "read caption" in command or "read description" in command:
        if "image" in st.session_state:
            cap = get_caption(st.session_state["image"])
            audio_bytes = text_to_speech(cap)
            st.audio(audio_bytes, format="audio/mp3")
            if audio_bytes:
                st.download_button("Download Caption Audio", data=audio_bytes, file_name="caption.mp3", mime="audio/mp3")
            st.success("Caption read out loud.")
        else:
            st.error("No image found to read caption from.")
    # Additional commands can be added here

# -------------------------------------------------------------------
# Load the BLIP VQA model and processor (cached for efficiency)
@st.cache_resource
def initialize_vqa_model():
    model_cache_dir = "Model_VQA"
    os.makedirs(model_cache_dir, exist_ok=True)
    try:
        processor_vqa = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base", cache_dir=model_cache_dir)
        model_vqa = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", cache_dir=model_cache_dir)
    except Exception as e:
        st.error(f"Error loading VQA model: {e}")
        raise e
    return processor_vqa, model_vqa

processor_vqa, model_vqa = initialize_vqa_model()

def get_vqa_answer(question, image):
    """
    Uses the BLIP VQA model to generate an answer based on the input image and question.
    """
    if image is None:
        return "No image available. Please upload or capture an image first."
    inputs = processor_vqa(image, question, return_tensors="pt")
    out = model_vqa.generate(**inputs)
    answer = processor_vqa.decode(out[0], skip_special_tokens=True)
    return answer

# -------------------------------------------------------------------
# Load the BLIP captioning model and processor (cached for efficiency)
@st.cache_resource
def initialize_caption_model():
    model_cache_dir = "Model_Caption"
    os.makedirs(model_cache_dir, exist_ok=True)
    try:
        processor_caption = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=model_cache_dir)
        model_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=model_cache_dir)
    except Exception as e:
        st.error(f"Error loading caption model: {e}")
        raise e
    return processor_caption, model_caption

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
processor_caption, model_caption = initialize_caption_model()

def get_caption(image):
    """Generates a caption for the input image using the BLIP captioning model."""
    inputs = processor_caption(image, return_tensors="pt")
    out = model_caption.generate(**inputs)
    caption = processor_caption.decode(out[0], skip_special_tokens=True)
    return caption

def text_to_speech(text):
    """Converts the given text to speech using gTTS and returns audio bytes."""
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.getvalue()

def process_image(image):
    """Processes the image: generates a caption and converts it to speech."""
    caption = get_caption(image)
    audio_bytes = text_to_speech(caption)
    return caption, audio_bytes

# -------------------------------------------------------------------
# Custom CSS for a creative, modern look
st.markdown(
    """
    <style>
    /* Set a gradient background */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #1F509A 100%);
    }
    /* Style the title */
    .title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px #000000;
    }
    /* Style for subtitles */
    .subtitle {
        font-size: 1.5rem;
        color: #ffffff;
    }
    /* Style the sidebar text */
    .sidebar .sidebar-content {
        font-size: 1.1rem;
    }
    /* Style emergency call links */
    .emergency-link {
        font-size: 1.3rem;
        color: red;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Side bar logo
st.sidebar.image("vivox.png", width=100)

# Sidebar: User Onboarding, Emergency Contacts, and About
st.sidebar.header("VisoVox AI")

# --- About VisoVox AI ---
st.sidebar.info(
    """
    **VisoVox AI** empowers visually impaired users by converting images into descriptive captions and providing both audio feedback and interactive Q&A about the image.
    Upload or capture an image and then ask any question about it!
    """
)

# --- User Onboarding ---
user_name = st.sidebar.text_input("Enter your name:", key="user_name")
if user_name and user_name.strip():
    st.sidebar.markdown(f"### Welcome, {user_name}!", unsafe_allow_html=True)

# Sidebar Conversation UI for Image Q&A
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

st.sidebar.subheader("VisoVox Image Q&A Chat")
user_question = st.sidebar.text_input("Ask a question about the image:")

# NEW: Sidebar button to record a voice question
if st.sidebar.button("Record Voice Question"):
    voice_question = listen_voice_command()
    if voice_question:
        if "image" in st.session_state:
            answer = get_vqa_answer(voice_question, st.session_state["image"])
            st.session_state.conversation.append(("User (Voice)", voice_question))
            st.session_state.conversation.append(("VisoChat", answer))
            st.sidebar.markdown(f"**User (Voice):** {voice_question}")
            st.sidebar.markdown(f"**VisoChat:** {answer}")
            audio_ans = text_to_speech(answer)
            st.sidebar.audio(audio_ans, format="audio/mp3")
            if audio_ans:
                st.sidebar.download_button("Download Answer Audio", data=audio_ans, file_name="answer.mp3", mime="audio/mp3")
        else:
            st.sidebar.error("Please process an image first.")

# -------------------------------------------------------------------
if st.sidebar.button("Send Question"):
    if "image" in st.session_state:
        answer = get_vqa_answer(user_question, st.session_state["image"])
        st.session_state.conversation.append(("User", user_question))
        st.session_state.conversation.append(("VisoChat", answer))
    else:
        st.session_state.conversation.append(("VisoChat", "Please process an image first."))

if st.session_state.conversation:
    for speaker, message in st.session_state.conversation:
        st.sidebar.markdown(f"**{speaker}:** {message}")

# --- Emergency Contacts ---
st.sidebar.markdown("### Emergency Contact")
st.sidebar.markdown(
    '<a class="emergency-link" href="tel:112" style="text-decoration: none;">🚨 Emergency</a>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
def main():
    # Main title with custom CSS class
    st.markdown("<h1 class='title'>VisoVox AI - Empowering Accessibility with Vision & Voice</h1>", unsafe_allow_html=True)
    
    # Display greeting in the main area if the name is provided via sidebar.
    if "user_name" in st.session_state and st.session_state["user_name"].strip():
        st.markdown(f"<h2 class='subtitle'>Hello, {st.session_state['user_name']}! Welcome to VisoVox AI.</h2>", unsafe_allow_html=True)
    
    st.write("Upload an image or capture one using your webcam to generate a caption and listen to the description.")
    
    # NEW: Voice Command Control Button
    if st.button("Activate Voice Command Control"):
        handle_voice_control()
    
    # NEW: If voice command set image capture mode, override input method
    if "voice_capture" in st.session_state and st.session_state["voice_capture"] is True:
        st.info("Using webcam for image capture (set via voice command).")
        input_method = "Take a picture (Webcam)"
    else:
        # Choose the image input method.
        input_method = st.radio("Select image input method:", ("Upload image", "Take a picture (Webcam)"))
    
    image = None
    if input_method == "Upload image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        camera_image = st.camera_input("Capture an image")
        if camera_image is not None:
            image = Image.open(camera_image)
    
    # NEW: If voice command set question mode, automatically capture voice question
    if "voice_question" in st.session_state and st.session_state["voice_question"] is True:
        st.session_state["voice_question"] = False  # reset flag
        st.info("Please speak your question now...")
        voice_q = listen_voice_command()
        if "image" in st.session_state:
            answer = get_vqa_answer(voice_q, st.session_state["image"])
            st.session_state.conversation.append(("User", voice_q))
            st.session_state.conversation.append(("VisoChat", answer))
            st.success(f"Answer: {answer}")
            audio_ans = text_to_speech(answer)
            st.audio(audio_ans, format="audio/mp3")
            if audio_ans:
                st.download_button("Download Answer Audio", data=audio_ans, file_name="answer.mp3", mime="audio/mp3")
        else:
            st.error("No image available to answer your question.")
    
    if image is not None:
        # Store image in session state for Q&A usage.
        st.session_state["image"] = image
        # Display image and process it in two columns.
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Selected Image", use_container_width=True)
        with col2:
            st.subheader("Processing Image...")
            with st.spinner("Generating caption and audio..."):
                caption, audio_bytes = process_image(image)
            st.markdown("<p class='subtitle'>Generated Caption:</p>", unsafe_allow_html=True)
            st.write(caption)
            st.markdown("<p class='subtitle'>Text-to-Speech:</p>", unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/mp3")
            if audio_bytes:
                st.download_button("Download Caption Audio", data=audio_bytes, file_name="caption.mp3", mime="audio/mp3")
    else:
        st.info("Please upload or capture an image to get started.")

if __name__ == '__main__':
    main()