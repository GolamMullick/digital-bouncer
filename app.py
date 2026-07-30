import streamlit as st
import requests
import os
from datetime import datetime

API_URL = "http://localhost:8000/verify"

# Directory for storing failed verification edge cases locally
EDGE_CASE_DIR = "data/edge_cases/spoof_attempts"
os.makedirs(EDGE_CASE_DIR, exist_ok=True)

st.set_page_config(page_title="The Digital Bouncer", layout="centered")

st.title("🛡️ The Digital Bouncer")
st.write("Upload a live selfie to verify liveness via the FastAPI backend.")

selfie_upload = st.file_uploader("Upload Live Selfie", type=["jpg", "png", "jpeg"])

if st.button("Verify Liveness"):
    if selfie_upload:
        with st.spinner("Communicating with FastAPI microservice..."):
            
            files = {
                "selfie": (selfie_upload.name, selfie_upload.getvalue(), selfie_upload.type)
            }
            
            try:
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    liveness_score = data["liveness_score"]
                    
                    st.divider()
                    
                    if data["status"] == "APPROVED":
                        st.success(f"✅ Access Granted! Liveness Score: {liveness_score}")
                    else:
                        st.error(f"❌ Access Denied. Spoof detected (Score: {liveness_score})")
                        
                        # --- LOCAL STORAGE FEEDBACK LOOP ---
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_path = os.path.join(EDGE_CASE_DIR, f"spoof_{timestamp}.jpg")
                        
                        with open(file_path, "wb") as f:
                            f.write(selfie_upload.getvalue())
                            
                        st.info(f"Edge case saved locally to `{file_path}` for future model retraining.")
                        # -----------------------------------
                            
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Ensure `python api.py` is running on port 8000!")

    else:
        st.warning("Please upload a selfie to continue.")