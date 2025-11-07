import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model_path = "./age_gender_model.h5"
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi tải mô hình Deep Learning: {e}. Vui lòng kiểm tra đường dẫn '{model_path}' và đảm bảo file model đã tồn tại.")
        return None

model = load_model()

if model is None:
    st.stop() # Dừng ứng dụng nếu không tải được model

st.title("Nhận diện tuổi & giới tính")
st.write("Upload ảnh👇")

# === LOAD FACE DETECTION MODEL (DNN) ===
@st.cache_resource
def load_face_detector():
    prototxt = "./opencv_face_detector.pbtxt"
    model_dnn = "./opencv_face_detector_uint8.pb"
    try:
        net = cv2.dnn.readNet(model_dnn, prototxt)
        return net
    except Exception as e:
        st.error(f"Lỗi tải mô hình phát hiện khuôn mặt DNN: {e}. Vui lòng đảm bảo 2 file '{prototxt}' và '{model_dnn}' đã có trong thư mục.")
        return None

net = load_face_detector()

# === UPLOAD ẢNH ===
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lưu tạm và đọc ảnh
    img_path = "temp_img.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Đọc ảnh bằng OpenCV
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        st.error("Lỗi: Không thể đọc được file ảnh.")
        st.stop()

    h, w = img_cv2.shape[:2]
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) # Để hiển thị trong Streamlit
    
    # === PHẦN PHÁT HIỆN KHUÔN MẶT MỚI (Sử dụng OpenCV DNN) ===
    
    # Tạo blob (đầu vào cho DNN)
    blob = cv2.dnn.blobFromImage(img_cv2, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    confidence_threshold = 0.7 
    
    # Lặp qua các phát hiện
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Lọc theo ngưỡng tin cậy
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Đảm bảo tọa độ hợp lệ và không vượt quá kích thước ảnh
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Lưu tọa độ dưới dạng (x, y, w, h) tương thích với logic cũ
            faces.append((startX, startY, endX - startX, endY - startY))
    
    if len(faces) == 0:
        st.error("Không phát hiện được khuôn mặt nào trong ảnh. Vui lòng thử ảnh khác.")
    else:
        # Lấy khuôn mặt có độ tin cậy cao nhất (khuôn mặt đầu tiên)
        x, y, w, h = faces[0]
        
        # Cắt khuôn mặt
        face_img = img_rgb[y:y+h, x:x+w]

        # Resize về đúng kích thước model yêu cầu
        face_img_resized = cv2.resize(face_img, (128, 128))
        face_img_norm = face_img_resized.astype("float32") / 255.0
        input_img = np.expand_dims(face_img_norm, axis=0)

        # Dự đoán
        pred_gender, pred_age = model.predict(input_img, verbose=0)

        # Xử lý kết quả
        # Giả định age_gender_model_1.h5 dự đoán tuổi là giá trị [0, 1] cần scale
        clamped_age = np.clip(pred_age[0][0], 0, 1)
        age_pred = int(clamped_age * 116) if clamped_age > 0 else 1
        gender_pred_label = "Nam" if pred_gender[0][0] < 0.5 else "Nữ"

        # Hiển thị kết quả
        st.image(face_img, caption=f"Khuôn mặt được cắt: {age_pred} tuổi, {gender_pred_label}")
        st.success(f"**Dự đoán:** {gender_pred_label}, khoảng **{age_pred} tuổi**")

        # Vẽ khung lên ảnh gốc
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0,255,0), 2)
        st.image(img_rgb, caption="Ảnh gốc với khung khuôn mặt")