import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import openai
import json
import matplotlib.cm as cm
import os
import gdown

st.set_page_config(page_title="Agri-Smart Advisor", layout="wide")

MODEL_FILE_ID = "https://drive.google.com/file/d/1NzmXgv3nDe0xorHoxhWF06cYLd21UMM4/view?usp=drive_link"
MODEL_PATH = "corn_model.h5"

@st.cache_resource
def load_model_and_indices():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    class_names = {
        0: 'Maize_Blight', 
        1: 'Maize_Common_Rust', 
        2: 'Maize_Gray_Leaf_Spot', 
        3: 'Maize_Healthy',
        4: 'Weed_Broadleaf',
        5: 'Weed_Grass'
    }
    return model, class_names

try:
    with st.spinner("Loading Model..."):
        model, class_names = load_model_and_indices()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def get_real_weather(city_name):
    try:
        url = f"https://wttr.in/{city_name}?format=j1"
        response = requests.get(url)
        data = response.json()
        current = data['current_condition'][0]
        return {
            "temperature": int(current['temp_C']),
            "humidity": int(current['humidity']),
            "condition": current['weatherDesc'][0]['value'],
            "city": city_name
        }
    except:
        return {"temperature": 25, "humidity": 50, "condition": "Unknown", "city": city_name}

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_activation", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def get_openai_advice(api_key, vision_results, weather):
    if not api_key: return "API Key not found."
    client = openai.OpenAI(api_key=api_key)
    problems = ", ".join(vision_results) if vision_results else "None (Healthy)"
    prompt = f"""
    You are an expert agronomist.
    Situation: Crop Issues: {problems} | Weather: {weather['condition']}, {weather['temperature']}C, Humidity {weather['humidity']}%
    Task: Analyze issues. Check weather for spraying safety. Recommend chemical & organic treatment. Concise (150 words).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e: return f"OpenAI Error: {e}"

st.title("Agri-Smart: Corn Doctor")

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

city = st.sidebar.text_input("Farm Location (City)", value="Vellore")
uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "png", "jpeg"])

if uploaded_file and city:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        original_img.thumbnail((800, 800))
        st.image(original_img, caption="Uploaded Image", use_column_width=True)
        
        st.info("Analyzing...")
        width, height = original_img.size
        crops = {
            "Top-Left": original_img.crop((0, 0, width//2, height//2)),
            "Top-Right": original_img.crop((width//2, 0, width, height//2)),
            "Bottom-Left": original_img.crop((0, height//2, width//2, height)),
            "Bottom-Right": original_img.crop((width//2, height//2, width, height))
        }
        
        found_problems = set()
        cols = st.columns(2)
        
        for i, (pos, crop_img) in enumerate(crops.items()):
            img_array = crop_img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_array)
            img_batch = np.expand_dims(img_array, axis=0)
            preprocessed_img = tf.keras.applications.efficientnet_v2.preprocess_input(img_batch.copy())
            
            preds = model.predict(preprocessed_img, verbose=0)
            pred_index = np.argmax(preds)
            confidence = np.max(preds) * 100
            label = class_names[pred_index]
            
            if confidence > 60:
                found_problems.add(label)
                try:
                    heatmap = make_gradcam_heatmap(preprocessed_img, model, "top_activation", pred_index)
                    gradcam_img = overlay_heatmap(img_array, heatmap)
                    with cols[i % 2]: st.image(gradcam_img, caption=f"{pos}: {label} ({confidence:.1f}%)")
                except:
                    with cols[i % 2]: st.image(crop_img, caption=f"{pos}: {label} ({confidence:.1f}%)")

        st.divider()
        weather = get_real_weather(city)
        st.subheader(f"Weather in {weather['city']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Temp", f"{weather['temperature']} C")
        c2.metric("Sky", weather['condition'])
        c3.metric("Humid", f"{weather['humidity']}%")
        
        st.subheader("AI Diagnosis")
        vision_list = list(found_problems)
        if not vision_list: st.success("Crop looks healthy!")
        else:
            advice = get_openai_advice(api_key, vision_list, weather)
            st.write(advice)
            
    except Exception as e: st.error(f"An error occurred: {e}")
