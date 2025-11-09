import streamlit as st
import joblib
import numpy as np
import time # Import time for the spinner

# --- Configuration ---
MODEL_PATH = "proof_verifier_pipeline.joblib"

# --- Page Config (Do this first!) ---
st.set_page_config(
    page_title="Math Proof Verifier",
    page_icon="🔎",  # Adds a fun icon to the browser tab
    layout="wide",  # Uses the full screen width
    initial_sidebar_state="expanded" # Keeps the sidebar open by default
)

# --- Load Model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(path):
    """Loads the saved model pipeline from disk."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        # We'll show the error in the main app area, not here
        return None
    except Exception as e:
        return str(e) # Return the error message

model = load_model(MODEL_PATH)

# --- Sidebar ---
# All the "about" and "limitations" text goes here
with st.sidebar:
    st.title("🤖 About this App")
    st.markdown("""
    This app is a prototype UI for a **Machine Learning model** trained to classify mathematical proofs.
    """)
    st.info("It is trained on a dataset of ~14,000 correct and flawed proofs.")
    
    st.divider() # A visual separator
    
    st.subheader("⚠️ Model Limitations")
    st.warning("""
    This AI is a **text classifier**, not a formal logic verifier. 
    
    * It **CAN** provide a verdict based on statistical patterns it learned.
    * It **CANNOT** provide a line-by-line critique or identify the *specific* logical flaw.
    
    It learns "this collection of words *feels* flawed," not "this line is a divide-by-zero error."
    """)

# --- Main Page ---
st.title("🔎 Math Proof Verifier")
st.markdown("Paste a mathematical proof below to get a verdict on its validity. The model will analyze the text and provide its classification.")

# Check if the model loaded correctly
if isinstance(model, str):
    st.error(f"An error occurred while loading the model: {model}")
elif model is None:
    st.error(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
else:
    # --- Proof Input ---
    proof_text = st.text_area(
        "Enter Proof Text:", 
        height=300, 
        placeholder="To prove 1=2, let a=b. Then a^2 = ab. So a^2 - b^2 = ab - b^2. Factoring, (a-b)(a+b) = b(a-b). Now, we divide by (a-b) to get a+b = b. Since a=b, we have 2b = b. This implies 2=1."
    )

    # --- Get Verdict Button ---
    # use_container_width=True makes the button span the full width
    if st.button("Get Verdict", type="primary", use_container_width=True):
        if proof_text.strip() != "":
            # --- Add Spinner for better UX ---
            with st.spinner("Analyzing your proof... 🧠"):
                time.sleep(0.5) # A small delay to make the spinner feel more "real"
                try:
                    # --- Make Prediction ---
                    prediction = model.predict([proof_text])[0]
                    
                    # --- Get Confidence Score (if available) ---
                    confidence_score = None
                    if hasattr(model.named_steps['clf'], 'predict_proba'):
                        probabilities = model.predict_proba([proof_text])[0]
                        confidence_score = probabilities[prediction] * 100 

                    # --- Display Verdict using Columns ---
                    st.subheader("Verdict")
                    col1, col2 = st.columns([1, 4]) # Create two columns

                    if prediction == 1:
                        with col1:
                            # Make the emoji bigger using markdown
                            st.markdown("<h1 style='text-align: center; font-size: 4rem;'>✅</h1>", unsafe_allow_html=True)
                        with col2:
                            st.success("**Verdict: Correct**")
                            if confidence_score:
                                st.metric(label="Model Confidence", value=f"{confidence_score:.2f}%")
                            else:
                                st.write("The model has classified this proof as correct.")
                        
                        # Add a fun celebration
                        st.balloons()
                    else:
                        with col1:
                            st.markdown("<h1 style='text-align: center; font-size: 4rem;'>❌</h1>", unsafe_allow_html=True)
                        with col2:
                            st.error("**Verdict: Flawed**")
                            if confidence_score:
                                st.metric(label="Model Confidence", value=f"{confidence_score:.2f}%", delta=f"High Flaw Confidence", delta_color="inverse")
                            else:
                                st.write("The model has classified this proof as flawed.")
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    
        else:
            # This message shows if the text area is empty
            st.warning("Please paste a proof to verify.")
# streamlit run app.py - Run this on terminal to open the webpage.