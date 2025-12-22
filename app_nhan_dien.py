import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import os

# 1. CẤU HÌNH & LOAD MODEL
MODEL_PATH = "billboard_checkpoint.keras"
IMG_SIZE = (64, 64)
DEFAULT_THRESHOLD = 0.86

print("Đang tải model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--> Đã tải model thành công!")
except Exception as e:
    print(f"Lỗi: Không tìm thấy file {MODEL_PATH}")
    exit()

# 2. HÀM XỬ LÝ 
def preprocess_image(img_crop_rgb):
    """Chuẩn bị ảnh để đưa vào model"""
    img_resized = cv2.resize(img_crop_rgb, IMG_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_batch

def get_region_proposals(image_rgb):
    """Tìm vùng nghi ngờ bằng OpenCV"""
    # Gradio gửi ảnh RGB, OpenCV cần Gray để xử lý
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 or h < 30: continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.2 or aspect_ratio > 5: continue
        proposals.append((x, y, w, h))
    return proposals

# 3. HÀM CHÍNH CHO GRADIO
def analyze_image(input_img, threshold):
    if input_img is None:
        return None, [], "Vui lòng tải ảnh lên."

    # Copy ảnh để vẽ (Ảnh input của Gradio là RGB)
    output_img = input_img.copy()
    
    # 1. Tìm vùng đề xuất
    proposals = get_region_proposals(input_img)
    
    detected_crops = [] # Danh sách chứa ảnh đã cắt
    log_info = ""       # Chuỗi chứa thông tin tọa độ
    count = 0

    # 2. Duyệt qua từng vùng
    for (x, y, w, h) in proposals:
        # Cắt vùng ảnh (ROI)
        roi = input_img[y:y+h, x:x+w]
        
        # Dự đoán
        processed_roi = preprocess_image(roi) # roi đã là RGB
        prediction = model.predict(processed_roi, verbose=0)[0][0]
        
        # Nếu đạt ngưỡng tin cậy
        if prediction >= threshold:
            count += 1
            
            # A. Vẽ khung lên ảnh lớn
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{prediction*100:.0f}%"
            cv2.putText(output_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # B. Thêm vào danh sách ảnh crop (để hiện lên Gallery)
            # Thêm chú thích cho ảnh crop
            crop_label = f"Tin cậy: {prediction*100:.1f}%"
            detected_crops.append((roi, crop_label))
            
            # C. Ghi log thông tin (Format: class x y w h conf)
            log_info += f"Biển báo #{count}: x={x}, y={y}, w={w}, h={h}, Conf={prediction:.4f}\n"

    status_text = f"KẾT QUẢ: Tìm thấy {count} biển quảng cáo.\n\nCHI TIẾT:\n{log_info}"   
    return output_img, detected_crops, status_text

# 4. GIAO DIỆN GRADIO
with gr.Interface(
    fn=analyze_image,
    inputs=[
        gr.Image(label="Tải ảnh đô thị", type="numpy"),
        gr.Slider(0.5, 1.0, value=DEFAULT_THRESHOLD, label="Độ tin cậy tối thiểu (Threshold)")
    ],
    outputs=[
        gr.Image(label="Ảnh đã nhận diện"),
        gr.Gallery(label="Các biển báo đã cắt (Crops)", columns=4, height=200), # Hiển thị ảnh crop đẹp mắt
        gr.Textbox(label="Thông tin chi tiết (Log)", lines=10)
    ],
    title="🔍 Demo Đồ Án: Nhận diện Biển Quảng Cáo",
    description="Hệ thống Hybrid (OpenCV + CNN) tự động phát hiện, cắt và trích xuất thông tin biển quảng cáo.",
    flagging_mode="never" 
) as demo:
    pass

if __name__ == "__main__":
    demo.launch()