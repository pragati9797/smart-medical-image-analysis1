import streamlit as st
from PIL import Image
from model import predict_disease
from chatbot import medical_chatbot

st.set_page_config(page_title="Smart Medical AI", layout="centered")

st.title("🩺 Smart Medical Image Analysis System")
st.write("Upload X-ray or MRI image to detect disease")

# Image Upload
uploaded_file = st.file_uploader(
    "Upload X-ray / MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Medical Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image..."):
         disease, confidence, severity, probabilities, explanation = predict_disease(image)

        st.success("Analysis Complete")

        st.subheader("🧠 Detection Result")
        st.write(f"**Detected Disease:** {disease}")
        st.write(f"**Confidence:** {confidence}%")
        st.write(f"**Severity Level:** {severity}")

        st.subheader("📊 Probability of Each Disease")
        for disease_name, prob in probabilities.items():
            st.write(f"{disease_name}: {prob}%")

        st.subheader("🧾 Explanation")
        st.write(explanation)

        st.session_state["disease"] = disease
        st.session_state["disease"] = disease
        st.session_state["confidence"] = confidence
        st.session_state["severity"] = severity
        st.session_state["probabilities"] = probabilities
        st.session_state["explanation"] = explanation

# Chatbot Section
st.divider()
st.subheader("💬 Medical AI Chatbot")

if "disease" in st.session_state:
    user_question = st.text_input("Ask a question about the image or disease")

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            answer = medical_chatbot(
               st.session_state["disease"],
               user_question,
               confidence=st.session_state.get("confidence"),
               severity=st.session_state.get("severity")
)

            
        st.write("### 🤖 Chatbot Answer")
        st.write(answer)
else:
    st.info("Please upload and analyze an image first.")
