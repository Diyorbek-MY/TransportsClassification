import streamlit as st
from fastai.vision.all import *
from fastai.learner import load_learner
import pathlib
import plotly.express as px

# Fix compatibility for Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# App title
st.title("Transport Classification Model")

# File uploader
file = st.file_uploader("Upload an image", type=['png', 'webp', 'jpeg', 'pdf', 'gif', 'svg', 'jpg'])

if file:
    # Show uploaded image
    st.image(file, caption="Uploaded Image", use_container_width=True)

    # Convert uploaded file to PIL Image
    img = PILImage.create(file)

    # Load model
    model = load_learner("transport_model.pkl")

    # Predict
    pred, pred_id, probs = model.predict(img)

    # Show prediction
    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {probs[pred_id] * 100:.1f}%")

    # Plot probabilities
    fig = px.bar(
        x=probs * 100,
        y=model.dls.vocab,
        orientation='h',
        labels={'x': 'Probability (%)', 'y': 'Class'},
        title='Prediction Probabilities'
    )
    st.plotly_chart(fig)
