import os
import cv2
import numpy as np
import tensorflow as tf

# 1. CẤU HÌNH
# ĐƯỜNG DẪN ẢNH INPUT 
INPUT_FOLDER = "anh_test"  

MODEL_PATH = "billboard_checkpoint.keras"
IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.94
DISPLAY_HEIGHT = 500 # Chiều cao cố định để hiển thị cho vừa màn hình

OUTPUT_CROP_DIR = "crops"   # Nơi lưu ảnh đã cắt
OUTPUT_LABEL_DIR = "labels" # Nơi lưu file txt tọa độ

# Tạo thư mục nếu chưa có
os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 2. HÀM HỖ TRỢ
# --- TẢI MODEL ---
print("Đang tải model (vui lòng chờ)...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"--> Đã tải model OK.")
except Exception as e:
    print(f"Lỗi: Không tìm thấy file {MODEL_PATH}")
    exit()

def preprocess_image(img_crop_rgb):
    """Chuẩn bị ảnh nhỏ RGB để đưa vào model"""
    img_resized = cv2.resize(img_crop_rgb, IMG_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_batch

def resize_height(img, target_height):
    """Hàm phụ trợ để resize ảnh về chiều cao cố định, giữ tỷ lệ khung hình"""
    h, w = img.shape[:2]
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    resized_img = cv2.resize(img, (target_width, target_height))
    return resized_img

# --- HÀM TÌM VÙNG ---
def get_region_proposals_and_edges(image_bgr):
    """
    Trả về cả danh sách vùng đề xuất và ảnh biên cạnh (edge image)
    """
    proposals = []
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # 2. Làm mờ nhẹ để giảm nhiễu
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3. Phát hiện cạnh Canny (Đây là "Ảnh cắt nét")
    edged = cv2.Canny(blur, 50, 150)
    
    # 4. Tìm contours từ ảnh cạnh
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Lọc sơ bộ các vùng nhiễu
        if w < 30 or h < 30: continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.2 or aspect_ratio > 5: continue
        proposals.append((x, y, w, h))
        
    # Convert ảnh edged từ Grayscale sang BGR để tí nữa ghép được với ảnh màu
    edged_bgr_visualization = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    
    return proposals, edged_bgr_visualization

# -- VÒNG LẶP XỬ LÝ CHÍNH -- 
def run_batch_process():
    valid_extensions = ['.jpg', '.jpeg', '.png']
    try:
        image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_extensions]
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục '{INPUT_FOLDER}'.")
        return

    if len(image_files) == 0:
        print(f"Thư mục {INPUT_FOLDER} rỗng.")
        return

    print(f"--> Tìm thấy {len(image_files)} ảnh. Bắt đầu...")

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None: continue
            
        print(f"Đang xử lý: {img_name}...")

        # Chuẩn bị file text để ghi thông tin (Mỗi ảnh 1 file txt cùng tên)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_name)
        
        # Mở file text để ghi (mode 'w' sẽ ghi mới)
        with open(txt_path, "w") as f_txt:
            
            proposals, edge_img_vis = get_region_proposals_and_edges(original_img)
            final_result_img = original_img.copy()
            detected_count = 0
            
            for idx, (x, y, w, h) in enumerate(proposals):
                roi_bgr = original_img[y:y+h, x:x+w]
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                
                processed_roi = preprocess_image(roi_rgb)
                prediction = model.predict(processed_roi, verbose=0)[0][0]
                
                if prediction >= CONFIDENCE_THRESHOLD:
                    detected_count += 1
                    
                    # --- CROP VÀ LƯU ẢNH ---
                    # Đặt tên file crop: tenanhgoc_thutu.jpg
                    crop_name = f"{os.path.splitext(img_name)[0]}_crop_{idx}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_CROP_DIR, crop_name), roi_bgr)
                    
                    # --- XUẤT THÔNG TIN ---
                    # Ghi vào file txt theo định dạng: class x y w h confidence
                    # (class biển quảng cáo là 0)
                    line_info = f"0 {x} {y} {w} {h} {prediction:.4f}\n"
                    f_txt.write(line_info)
                    
                    # Vẽ lên ảnh 
                    cv2.rectangle(final_result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    label = f"{prediction*100:.0f}%"
                    cv2.putText(final_result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Hiển thị 
        disp_orig = resize_height(original_img, DISPLAY_HEIGHT)
        disp_edge = resize_height(edge_img_vis, DISPLAY_HEIGHT)
        disp_final = resize_height(final_result_img, DISPLAY_HEIGHT)
        combined_view = cv2.hconcat([disp_orig, disp_edge, disp_final])
        
        cv2.imshow("Visualizer", combined_view)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\nĐã xong! Kiểm tra kết quả tại thư mục: {OUTPUT_CROP_DIR} và {OUTPUT_LABEL_DIR}")

if __name__ == "__main__":
    run_batch_process()