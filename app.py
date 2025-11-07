import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import time
from PIL import Image

# =======================
# Load CSS
# =======================
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# =======================
# Load Model
# =======================
@st.cache_resource
def load_model_keras():
    model_path = "./age_gender_model.h5"
    try:
        model = load_model(model_path, compile=False)
        st.success("✅ Đã tải mô hình thành công.")
        return model
    except Exception as e:
        st.error(f"❌ Lỗi tải mô hình: {e}")
        return None

model = load_model_keras()
if model is None:
    st.stop()

# =======================
# Load Face Detector (OpenCV DNN)
# =======================
@st.cache_resource
def load_face_detector():
    prototxt = "./opencv_face_detector.pbtxt"
    model_dnn = "./opencv_face_detector_uint8.pb"
    try:
        net = cv2.dnn.readNet(model_dnn, prototxt)
        return net
    except Exception as e:
        st.error(f"Lỗi tải mô hình DNN: {e}")
        return None

net = load_face_detector()

# =======================
# App Title
# =======================
st.title("Nhận diện tuổi & giới tính")
st.write("Upload ảnh👇")

# =======================
# Upload ảnh
# =======================
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lưu tạm
    img_path = "temp_img.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Đọc ảnh
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        st.error("Không đọc được ảnh")
        st.stop()

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
            faces.append((startX,startY,endX-startX,endY-startY))

    if len(faces) == 0:
        st.error("Không phát hiện khuôn mặt")
        st.stop()

    # Lấy khuôn mặt đầu tiên
    x, y, fw, fh = faces[0]
    face_img = img_rgb[y:y+fh, x:x+fw]
    face_img_resized = cv2.resize(face_img, (128,128))
    input_img = np.expand_dims(face_img_resized.astype("float32")/255.0, axis=0)

    # =======================
    # Dự đoán tuổi & giới tính
    # =======================
    pred_gender, pred_age = model.predict(input_img, verbose=0)
    clamped_age = np.clip(pred_age[0][0], 0,1)
    age_pred = int(clamped_age*116) if clamped_age>0 else 1
    gender_pred_label = "Nam" if pred_gender[0][0]<0.5 else "Nữ"


if uploaded_file is not None:
    # --- Xử lý face, predict ---
    pred_gender, pred_age = model.predict(input_img, verbose=0)
    clamped_age = np.clip(pred_age[0][0],0,1)
    age_pred = int(clamped_age*116) if clamped_age>0 else 1
    gender_pred_label = "Nam" if pred_gender[0][0]<0.5 else "Nữ"

    # --- Chọn icon ---
    if gender_pred_label=="Nam":
        if age_pred<18: icon="🧒"
        elif age_pred<=50: icon="👨"
        else: icon="👴"
    else:
        if age_pred<18: icon="🧒"
        elif age_pred<=50: icon="👩"
        else: icon="👵"

    # --- Hiển thị ảnh ---
    st.image(face_img, caption="Khuôn mặt được cắt")
    cv2.rectangle(img_rgb,(x,y),(x+fw,y+fh),(0,255,0),2)
    st.image(img_rgb, caption="Ảnh gốc với khung khuôn mặt")

    # --- Animation tuổi + card prediction ---
    card_placeholder = st.empty()
    current_age = 0
    while current_age <= age_pred:
        card_placeholder.markdown(f"""
        <div class="prediction-card">
            <div class="emoji-age">
                <span class="emoji">{icon}</span>
                <span class="age">{current_age} tuổi</span>
            </div>
            <div class="gender">{gender_pred_label}</div>
        </div>
        """, unsafe_allow_html=True)
        current_age += 1
        time.sleep(0.05)



st.markdown("""
<footer>
    <h2>Về chúng tôi</h2>
    <div class="team-container">
        <div class="team-card"><h3>Cao Thành Lâm</h3><p>Thành viên</p></div>
        <div class="team-card"><h3>Bùi Hoàng Linh</h3><p>Nhóm trưởng</p></div>
        <div class="team-card"><h3>Nguyễn Việt An</h3><p>Thành viên</p></div>
    </div>
</footer>
""", unsafe_allow_html=True)
