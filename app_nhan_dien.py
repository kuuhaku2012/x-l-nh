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

# 3. VÒNG LẶP XỬ LÝ CHÍNH
def run_batch_process():
    valid_extensions = ['.jpg', '.jpeg', '.png']
    try:
        image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_extensions]
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục '{INPUT_FOLDER}'. Kiểm tra lại đường dẫn.")
        return

    if len(image_files) == 0:
        print(f"Thư mục {INPUT_FOLDER} rỗng.")
        return

    print(f"--> Tìm thấy {len(image_files)} ảnh. Bắt đầu...")
    print("-------------------------------------------------")
    print(" [SPACE] : Ảnh tiếp theo")
    print(" [ Q ]   : Thoát")
    print("-------------------------------------------------")

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None: continue
            
        print(f"Đang xử lý ảnh {i+1}/{len(image_files)}: {img_name}...")

        # --- BƯỚC 1: Lấy Proposals và ảnh Edges ---
        proposals, edge_img_vis = get_region_proposals_and_edges(original_img)
        
        # Tạo bản copy để vẽ kết quả cuối cùng
        final_result_img = original_img.copy()
        
        detected_count = 0
        # --- BƯỚC 2: Duyệt và kiểm tra bằng CNN ---
        for (x, y, w, h) in proposals:
            # Cắt vùng ROI (đang là BGR)
            roi_bgr = original_img[y:y+h, x:x+w]
            # Chuyển ROI sang RGB để đưa vào model
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            
            # Predict
            processed_roi = preprocess_image(roi_rgb)
            prediction = model.predict(processed_roi, verbose=0)[0][0]
            
            # Nếu độ tin cậy cao
            if prediction >= CONFIDENCE_THRESHOLD:
                detected_count += 1
                # Vẽ lên ảnh kết quả
                cv2.rectangle(final_result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                label = f"{prediction*100:.0f}%"
                cv2.putText(final_result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Thêm text thông tin
        info = f"File: {img_name} | Found: {detected_count}/{len(proposals)}"
        cv2.putText(final_result_img, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(edge_img_vis, "OpenCV Canny Edges (Proposals)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # --- BƯỚC 3: GHÉP 3 ẢNH ĐỂ HIỂN THỊ ---
        # Resize về cùng chiều cao để ghép cho đẹp
        disp_orig = resize_height(original_img, DISPLAY_HEIGHT)
        disp_edge = resize_height(edge_img_vis, DISPLAY_HEIGHT)
        disp_final = resize_height(final_result_img, DISPLAY_HEIGHT)

        # Ghép ngang (Horizontal Stack)
        combined_view = cv2.hconcat([disp_orig, disp_edge, disp_final])

        # Hiển thị
        cv2.imshow("Visualizer: Original | Edges | CNN Result", combined_view)
        
        # Chờ phím
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nĐã thoát chương trình.")

if __name__ == "__main__":
    run_batch_process()