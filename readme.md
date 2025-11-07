# 🧠 Ứng dụng Nhận Diện Tuổi & Giới Tính bằng CNN

Ứng dụng này sử dụng **Streamlit** kết hợp với **TensorFlow** và **OpenCV DNN** để nhận diện khuôn mặt, dự đoán **tuổi** và **giới tính** từ ảnh người dùng tải lên.

---
## 🚀 Bắt đầu
### 1. 📥 Clone dự án

```bash
git clone https://github.com/kamusarj/DACN.git
cd DACN
```

---

### 2. 🐍 Tạo môi trường ảo với Python 3.11 (nếu chưa tạo)

> ⚠️ Đảm bảo bạn đã cài Python 3.11 trước đó.

```bash
py -3.11 -m venv venv
```

Kích hoạt môi trường ảo:

- **Windows (PowerShell):**

```powershell
.\venv\Scripts\activate.ps1
```

- **Windows (CMD):**

```cmd
.\venv\Scripts\activate.bat
```
hoặc
```cmd
.\venv\Scripts\activate
```

- **macOS/Linux:**

```bash
source venv/bin/activate
```

---

### 3. 📦 Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

---
### 4. 🧪 Chạy ứng dụng với streamlit

```bash
streamlit run app.py
```
## 📝 Lưu ý

- Ứng dụng sử dụng mô hình **`age_gender_model.h5`** để dự đoán tuổi và giới tính.  
- Nếu muốn **huấn luyện lại mô hình**, upload notebook **`notebook.ipynb`** lên Google Colab và sử dụng **Dataset** [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) từ Kaggle hoặc làm theo các bước sau:

1. **Tải dataset** [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) từ Kaggle.  
2. **Tạo môi trường ảo bằng Conda:**
   ```bash
   conda create -n age_gender python=3.12
   conda activate age_gender
3. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install tensorflow opencv-python pillow numpy matplotlib jupyter
4. Mở và chạy notebook **`notebook.ipynb`** để huấn luyện lại mô hình, sử dụng môi trường conda vừa tạo.  
5. Sau khi huấn luyện xong, mô hình mới sẽ được lưu thành file **`age_gender_model.h5`**.

##  Cấu trúc dự án
```
📁 project/
│
├── app.py
├── age_gender_model.h5
├── notebook.ipynb
├── best_model.h5
├── opencv_face_detector.pbtxt
├── opencv_face_detector_uint8.pb
└── temp_img.jpg (sẽ tạo khi upload)
```

