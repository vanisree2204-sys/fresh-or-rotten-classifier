import tensorflow as tf

# Path to your old model
old_model_path = 'fruitveg_model.h5'  # Change this if your model has a different name
new_model_path = 'converted_model.h5'

try:
    # Load using custom_objects workaround
    model = tf.keras.models.load_model(old_model_path, compile=False)
    print("[✔] Model loaded successfully")

    # Save again with new format
    model.save(new_model_path)
    print(f"[✔] Model re-saved as: {new_model_path}")

except Exception as e:
    print(f"[❌] Failed to convert model: {e}")
