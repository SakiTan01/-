import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# กำหนดค่าพารามิเตอร์
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # เพิ่มรอบเทรนอีกนิดเพราะข้อมูลมีความละเอียดสูงขึ้น
DATASET_DIR = "dataset"  # โฟลเดอร์ที่รวมเฉพาะโรคและใบมะเขือเทศ

# เตรียมข้อมูลและ Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,  # แบ่ง 20% ไว้ตรวจสอบความถูกต้อง
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

# ดึงจำนวนคลาสของโรคมะเขือเทศอัตโนมัติ
num_classes = len(train_generator.class_indices)
print(f"🌿 จำนวนคลาสโรคมะเขือเทศทั้งหมด: {num_classes} คลาส")

# สร้างโมเดล Transfer Learning (MobileNetV2)
base_model = MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# เริ่มเทรน
print("🚀 กำลังเทรนโมเดลโรคใบมะเขือเทศ...")
history = model.fit(
    train_generator, validation_data=val_generator, epochs=EPOCHS
)

# บันทึกโมเดล
model.save("tomato_disease_model.h5")
print("✅ บันทึกโมเดลโรคมะเขือเทศเป็น tomato_disease_model.h5 เรียบร้อยแล้ว!")