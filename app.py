import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import time
from PIL import Image

# =======================
# Cấu hình trang
# =======================
st.set_page_config(
    page_title="Nhận diện Tuổi & Giới tính",
    page_icon="📸",
    layout="centered"
)

# =======================
# Load CSS
# =======================
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# =======================
# Load Model (Có thông báo)
# =======================
@st.cache_resource
def load_model_keras():
    model_path = "./age_gender_model.h5"
    try:
        model = load_model(model_path, compile=False)
        # Thông báo thành công
        st.success("✅ Đã tải mô hình Tuổi & Giới tính thành công!") 
        return model
    except Exception as e:
        st.error(f"❌ Lỗi tải mô hình Keras: {e}")
        return None

model = load_model_keras()

# =======================
# Load Face Detector (Có thông báo)
# =======================
@st.cache_resource
def load_face_detector():
    prototxt = "./opencv_face_detector.pbtxt"
    model_dnn = "./opencv_face_detector_uint8.pb"
    try:
        net = cv2.dnn.readNet(model_dnn, prototxt)
        # Thông báo thành công
        st.success("✅ Đã tải mô hình Nhận diện khuôn mặt thành công!")
        return net
    except Exception as e:
        st.error(f"❌ Lỗi tải mô hình OpenCV: {e}")
        return None

net = load_face_detector()

if model is None or net is None:
    st.stop()

# =======================
# App Title
# =======================
st.title("📸 Nhận diện Tuổi & Giới tính")
st.markdown("<p style='text-align: center; color: #666;'>Tải ảnh lên hoặc dùng webcam để AI dự đoán</p>", unsafe_allow_html=True)

# =======================
# Input: Upload hoặc Webcam
# =======================
option = st.radio("Chọn nguồn ảnh:", ["Upload ảnh", "Chụp webcam"], horizontal=True)

uploaded_file = None
if option == "Upload ảnh":
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
else:
    camera_photo = st.camera_input("Chụp ảnh")
    if camera_photo:
        uploaded_file = camera_photo

if uploaded_file is not None:
    # Xử lý ảnh
    image = Image.open(uploaded_file).convert("RGB")
    img_cv2 = np.array(image)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    h, w = img_cv2.shape[:2]
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # =======================
    # Phát hiện khuôn mặt
    # =======================
    blob = cv2.dnn.blobFromImage(img_cv2, 1.0, (300, 300), (104,177,123), False, False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    conf_thresh = 0.7
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > conf_thresh:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            startX, startY, endX, endY = box.astype(int)
            startX = max(0,startX); startY=max(0,startY)
            endX = min(w,endX); endY=min(h,endY)
            if (endX - startX) > 10 and (endY - startY) > 10:
                faces.append((startX,startY,endX-startX,endY-startY))

    if len(faces) == 0:
        st.warning("⚠️ Không tìm thấy khuôn mặt nào trong ảnh.")
    else:
        # Lấy khuôn mặt đầu tiên
        x, y, fw, fh = faces[0]
        face_img = img_rgb[y:y+fh, x:x+fw]
        
        try:
            face_img_resized = cv2.resize(face_img, (128,128))
            input_img = np.expand_dims(face_img_resized.astype("float32")/255.0, axis=0)

            # =======================
            # Dự đoán
            # =======================
            with st.spinner("Đang phân tích..."):
                pred_gender, pred_age = model.predict(input_img, verbose=0)
                # time.sleep(0.5) 

            clamped_age = np.clip(pred_age[0][0], 0, 1)
            age_pred = int(clamped_age * 116) if clamped_age > 0 else 1
            gender_pred_label = "Nam" if pred_gender[0][0] < 0.5 else "Nữ"

            # --- Chọn icon ---
            if gender_pred_label == "Nam":
                if age_pred < 18: icon = "🧒"
                elif age_pred <= 50: icon = "👨"
                else: icon = "👴"
            else:
                if age_pred < 18: icon = "🧒"
                elif age_pred <= 50: icon = "👩"
                else: icon = "👵"

            # --- Hiển thị kết quả ---
            col_img, col_info = st.columns(2)
            
            with col_img:
                cv2.rectangle(img_rgb, (x, y), (x + fw, y + fh), (0, 255, 0), 3)
                st.image(img_rgb, caption="Ảnh đã xử lý", use_column_width=True)

            with col_info:
                st.write("### Kết quả dự đoán")
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="emoji-age">
                        <span class="emoji">{icon}</span>
                        <span class="age">{age_pred} tuổi</span>
                    </div>
                    <div class="gender">Giới tính: {gender_pred_label}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.image(face_img, caption="Khuôn mặt", width=100)

        except Exception as e:
            st.error(f"Lỗi xử lý ảnh: {e}")

# =======================
# Footer
# =======================
st.markdown("""
<footer>
    <h2>Thành viên thực hiện</h2>
    <div class="team-container">
        <div class="team-card"><h3>Cao Thành Lâm</h3><p>Thành viên</p></div>
        <div class="team-card"><h3>Bùi Hoàng Linh</h3><p>Nhóm trưởng</p></div>
        <div class="team-card"><h3>Nguyễn Việt An</h3><p>Thành viên</p></div>
    </div>
</footer>
""", unsafe_allow_html=True)