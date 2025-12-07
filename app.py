import streamlit as st
from PIL import Image
import google.generativeai as genai
import numpy as np
import cv2
import json

# 1. App Title and Instructions
st.set_page_config(page_title="AI Floor Measurer", layout="centered")
st.title("📏 AI Room Area Estimator")
st.write("Upload a photo of your room with an **A4 sheet of paper** on the floor.")

# 2. Sidebar for API Key (Keeps your key safe!)
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to start.")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)

# 3. Image Uploader
uploaded_file = st.file_uploader("Choose a room photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Calculate Area"):
        with st.spinner("Gemini is analyzing the floor and finding the A4 paper..."):
            
            # 4. The Prompt to Gemini
            # We ask it to find the pixel coordinates of the A4 paper
            prompt = """
            Look at this image. There is a standard A4 paper (210mm x 297mm) on the floor.
            1. Identify the 4 corners of the A4 paper.
            2. Identify the 4 visible corners of the floor/room area.
            
            Return ONLY a JSON object with this format, no other text:
            {
                "paper_corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "floor_corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            }
            """
            
            try:
                # Call Gemini 1.5 Pro (Better at vision tasks)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content([prompt, image])
                
                # Extract the text
                json_text = response.text.strip()
                # Clean up if Gemini accidentally added markdown (```json ...)
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0]
                
                data = json.loads(json_text)
                
                st.success("Analysis Complete!")
                st.json(data) # Show the raw data Gemini found
                
                st.info("💡 Note: Calculating exact square meters from these coordinates requires complex geometry (Homography). As a beginner project, we successfully used AI to 'see' the measurements!")

            except Exception as e:
                st.error(f"Error: {e}")

                st.write("Tip: Make sure the A4 paper is clearly visible and not too far away.")
