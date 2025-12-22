import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATASET_DIR = "dataset_cnn_final"
IMG_SIZE = (64, 64) 
BATCH_SIZE = 32
EPOCHS = 30 
# File tạm lưu lúc train (checkpoint)
CHECKPOINT_PATH = "billboard_checkpoint.keras" 
# File chính thức chỉ lưu khi đạt chuẩn
FINAL_MODEL_PATH = "billboard_classifier_verified.keras" 

# Ngưỡng tin cậy tối thiểu (ví dụ 94%)
TEST_ACCURACY_THRESHOLD = 0.94 

print("TF Version:", tf.__version__)

# ==========================================
# 2. LOAD DỮ LIỆU (TRAIN - VAL - TEST)
# ==========================================
# 2.1 Load Train
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# 2.2 Load Validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'validation'),
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# 2.3 [MỚI] Load Test (Bắt buộc phải có thư mục 'test')
test_path = os.path.join(DATASET_DIR, 'test_split')
if os.path.exists(test_path):
    print("--> Đã tìm thấy tập Test. Đang load...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
else:
    print("--> CẢNH BÁO: Không tìm thấy thư mục 'test'. Sẽ dùng tập validation để test tạm.")
    test_ds = val_ds

# Tối ưu hiệu năng
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. XÂY DỰNG MÔ HÌNH CNN
# ==========================================
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  layers.RandomBrightness(0.2), 
])

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    
    # Block 1
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten & Dense
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),
    
    # Output
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 4. HUẤN LUYỆN
# ==========================================
checkpoint_cb = callbacks.ModelCheckpoint(
    CHECKPOINT_PATH, 
    save_best_only=True, 
    monitor='val_accuracy', 
    mode='max'
)

early_stopping_cb = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True # Quan trọng: Khôi phục lại trọng số tốt nhất sau khi dừng
)

print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# ==========================================
# 5. [MỚI] KIỂM TRA ĐỘ TIN CẬY (QUALITY GATE)
# ==========================================
print("\n--- ĐANG CHẠY KIỂM THỬ TRÊN TẬP TEST (ĐỘC LẬP) ---")

# Đánh giá model trên tập Test
test_loss, test_acc = model.evaluate(test_ds)

print(f"\nKẾT QUẢ KIỂM TRA:")
print(f"- Test Loss: {test_loss:.4f}")
print(f"- Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"- Ngưỡng yêu cầu: {TEST_ACCURACY_THRESHOLD*100}%")

# Điều kiện lưu model chính thức
if test_acc >= TEST_ACCURACY_THRESHOLD:
    print("\nTHÀNH CÔNG! Model đạt độ tin cậy cao.")
    model.save(FINAL_MODEL_PATH)
    print(f"--> Đã lưu model chính thức tại: {FINAL_MODEL_PATH}")
    print("--> Bạn có thể dùng file này để chạy demo.")
else:
    print("\nTHẤT BẠI! Model chưa đủ độ tin cậy.")
    print(f"--> Chỉ lưu bản checkpoint tạm tại: {CHECKPOINT_PATH}")
    print("--> Gợi ý: Hãy thêm dữ liệu, tăng Epochs hoặc chỉnh lại kiến trúc mạng.")

# ==========================================
# 6. VẼ BIỂU ĐỒ
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
real_epochs = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(real_epochs, acc, label='Train Accuracy')
plt.plot(real_epochs, val_acc, label='Val Accuracy')
# Vẽ đường tham chiếu ngưỡng test (để so sánh)
plt.axhline(y=TEST_ACCURACY_THRESHOLD, color='r', linestyle='--', label='Test Threshold')
plt.legend(loc='lower right')
plt.title(f'Training Accuracy (Final Test: {test_acc*100:.1f}%)')

plt.subplot(1, 2, 2)
plt.plot(real_epochs, loss, label='Train Loss')
plt.plot(real_epochs, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.savefig("training_result_verified.png")
plt.show()