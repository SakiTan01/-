import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato_disease_model.h5")


model = load_model()

st.title("🍅 ระบบตรวจจับโรคใบมะเขือเทศด้วย AI")
st.write("อัปโหลดรูปภาพใบมะเขือเทศ เพื่อตรวจสอบโรคและแนวทางรักษา")

uploaded_file = st.file_uploader(
    "เลือกไฟล์รูปภาพใบมะเขือเทศ...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปใบมะเขือเทศที่อัปโหลด", use_column_width=True)

    # แปลงภาพ
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ทำนายผล
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    # รายชื่อโรคตามลำดับโฟลเดอร์ (เรียงตามตัวอักษรภาษาอังกฤษ A-Z)
    class_names = [
        "โรคจุดแบคทีเรีย (Bacterial Spot)",
        "โรคใบไหม้ช่วงแรก (Early Blight)",
        "โรคใบไหม้ช่วงลุกลาม (Late Blight)",
        "โรคราน้ำค้าง/ราแป้งบนใบ (Leaf Mold)",
        "โรคใบจุดเซปโตเรีย (Septoria Leaf Spot)",
        "โรคไรแดงสองจุด (Spider Mites)",
        "โรคจุดเป้า (Target Spot)",
        "ไวรัสใบหดเหลือง (Yellow Leaf Curl Virus)",
        "ไวรัสใบด่าง (Mosaic Virus)",
        "ใบปกติสมบูรณ์ (Healthy)",
    ]

    st.markdown("---")
    st.subheader("📌 ผลการวิเคราะห์:")
    st.success(f"**ผลการทำนาย:** {class_names[predicted_class]}")
    st.info(f"**ความมั่นใจของ AI:** {confidence:.2f}%")