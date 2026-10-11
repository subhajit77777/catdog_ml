import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Cat and Dog Classifier",
    page_icon="🐱"
)

st.title("🐱 Cat and Dog Classifier 🐶")
st.write("Upload a cat or dog image.")

MODEL_PATH = Path(__file__).parent / "model" / "cat_dog_model.keras"

if not MODEL_PATH.exists():
    st.error("Model file not found. Check the model folder.")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    resized_image = image.resize((128, 128))
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    st.subheader("Prediction Result")

    if prediction >= 0.5:
        st.success("🐶 Prediction: DOG")
        confidence = prediction * 100
    else:
        st.success("🐱 Prediction: CAT")
        confidence = (1 - prediction) * 100

    st.write(f"Confidence: {confidence:.2f}%")

else:
    st.info("Please upload a cat or dog image.")