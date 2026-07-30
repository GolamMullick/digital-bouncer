from zenml import step, pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import cv2
import mlflow
import os

@step(enable_cache=False)
def ingest_local_data() -> dict:
    """Loads local images with OpenCV, corrects rotation, and returns train/val NumPy arrays."""
    
    data_dir = "data/liveness_dataset"
    categories = ["real", "spoof"] 
    
    images = []
    labels = []
    
    print(f"\n--- INGESTION START ---")
    print(f"Reading local images from: {os.path.abspath(data_dir)}")
    
    for label_idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"❌ CRITICAL: Directory '{category_path}' does not exist!")
            
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        files_found = 0
        
        for file_name in os.listdir(category_path):
            if file_name.lower().endswith(valid_extensions):
                img_path = os.path.join(category_path, file_name)
                
                # Load image with OpenCV
                img = cv2.imread(img_path)
                if img is not None:
                    # 1. Rotate 90 degrees clockwise to fix orientation
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    
                    # 2. Convert color format and resize
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    
                    images.append(img)
                    labels.append(label_idx)
                    files_found += 1
                    
        print(f"Folder '{category}' -> Found {files_found} valid images.")
                    
    if len(images) == 0:
        raise ValueError(f"❌ CRITICAL: No valid images found in {data_dir}. Are your folders empty?")
        
    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Stratified 80% Train / 20% Validation Split
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Training samples: {len(x_train)} | Validation samples: {len(x_val)}")
    print(f"--- INGESTION END ---\n")
    
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val
    }


@step(experiment_tracker="mlflow_tracker")
def train_liveness_model(datasets: dict) -> tf.keras.Model:
    """Trains 2D CNN with stable learning rate (1e-4) and MLflow validation tracking."""
    mlflow.tensorflow.autolog(log_models=True, disable=False)
    
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).prefetch(AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32).prefetch(AUTOTUNE)
    
    model = tf.keras.Sequential([
        # Pixel Normalization (0-255 to 0-1)
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        
        # Mild Data Augmentation
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        
        # Core Architecture
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Reduced learning rate (1e-4) prevents loss explosion and model collapse
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Beginning Model Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )
    
    os.makedirs("model_registry", exist_ok=True)
    model.save("model_registry/liveness_cnn.keras")
    
    return model


@step
def deployment_trigger(model: tf.keras.Model) -> bool:
    print("✅ Model successfully saved to `model_registry/liveness_cnn.keras`.")
    return True


@pipeline
def local_ekyc_training_pipeline():
    """ZenML Training Pipeline using Local Data."""
    data = ingest_local_data()
    model = train_liveness_model(data)
    deployment_trigger(model)