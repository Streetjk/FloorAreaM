import streamlit as st
from PIL import Image, ImageDraw
import google.generativeai as genai
import numpy as np
import cv2
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Floor Measurer", layout="centered")

# --- 1. THE MATH ENGINE (New!) ---
def order_points(pts):
    """Sorts coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference
    return rect

def calculate_real_area(paper_pts, floor_pts):
    """
    Uses the A4 paper as a reference to calculate the floor area.
    paper_pts: List of 4 [x,y] coordinates for the paper.
    floor_pts: List of 4 [x,y] coordinates for the floor.
    """
    # 1. Order the points consistenttly
    paper_rect = order_points(np.array(paper_pts))
    floor_rect = order_points(np.array(floor_pts))

    # 2. Define the real-world size of A4 paper (in Meters)
    # A4 is 0.210 meters x 0.297 meters
    # We map the paper points to a flat rectangle of this size
    dst_pts = np.array([
        [0, 0],
        [0.210, 0],
        [0.210, 0.297],
        [0, 0.297]], dtype="float32")

    # 3. Calculate the "Perspective Transform Matrix" (The Magic Math)
    # This matrix maps the "tilted" pixels to "flat" real-world meters
    matrix = cv2.getPerspectiveTransform(paper_rect, dst_pts)

    # 4. Apply this matrix to the FLOOR corners
    # This converts floor pixels -> real world meters
    floor_poly_real = cv2.perspectiveTransform(floor_rect.reshape(-1, 1, 2), matrix)

    # 5. Calculate area of the new shape using the Shoelace formula (ContourArea)
    area_sq_meters = cv2.contourArea(floor_poly_real)
    
    return area_sq_meters

# --- 2. AUTHENTICATION ---
# (Checks for secrets, or falls back to manual input for testing)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("Enter your API Key to start.")
    st.stop()

genai.configure(api_key=api_key)

# --- 3. MAIN APP ---
st.title("📏 AI Room Area Estimator")
st.write("Upload a photo. The AI will find the A4 paper and calculate the floor size.")

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Photo", use_column_width=True)
    
    if st.button("Calculate Area"):
        with st.spinner("Analyzing geometry..."):
            
            # PROMPT
            prompt = """
            Find the A4 paper and the visible floor boundaries in this image.
            Return ONLY JSON. Do not return markdown.
            Format:
            {
                "paper_corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "floor_corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            }
            """
            
            try:
                model = genai.GenerativeModel('gemini-3-pro-preview') # Use the fast model
                response = model.generate_content([prompt, image])
                
                # Clean JSON
                text = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(text)
                
                # --- VISUALIZATION (Draw the lines) ---
                draw_img = image.copy()
                draw = ImageDraw.Draw(draw_img)
                
                # Draw Paper (Green)
                p_coords = [tuple(p) for p in data["paper_corners"]]
                draw.polygon(p_coords, outline="green", width=5)
                
                # Draw Floor (Red)
                f_coords = [tuple(f) for f in data["floor_corners"]]
                draw.polygon(f_coords, outline="red", width=5)
                
                st.image(draw_img, caption="AI Detection (Green=Paper, Red=Floor)", use_column_width=True)

                # --- CALCULATION (The Result) ---
                area = calculate_real_area(data["paper_corners"], data["floor_corners"])
                
                # Output the big number
                st.success(f"### Estimated Floor Area: {area:.2f} m²")
                
                # Conversion for convenience
                st.write(f"({area * 10.764:.2f} sq ft)")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.write("Tip: Try a photo where the A4 paper is larger/clearer.")

