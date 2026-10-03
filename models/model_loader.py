import os
import tensorflow as tf

def load_model(model_class, dataset_name, load_type="m", input_shape=(32, 32, 3), num_classes=10, saved_models_dir="saved_models"):
    """
    Loads a model either by instantiating `model_class` and loading weights (.weights.h5),
    or by loading the entire model architecture + weights (.keras).
    """
    ext = ".weights.h5" if load_type == "w" else ".keras"
    file_path = os.path.join(saved_models_dir, f"resnet50_{dataset_name}{ext}")
    file_path_best = os.path.join(saved_models_dir, f"resnet50_{dataset_name}_best_model.keras")

    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"Model file not found at: {file_path}")

    if load_type == "w":
        model = model_class(input_shape=input_shape, num_classes=num_classes)
        model.load_weights(file_path)
        print(f"Model weights successfully loaded from {file_path}")
    elif load_type == "m":
        model = tf.keras.models.load_model(file_path)
        print(f"Model successfully loaded from {file_path}")
    elif load_type == "b":
        model = tf.keras.models.load_model(file_path_best)
        print(f"Best model successfully loaded from {file_path_best}")
    else:
        raise ValueError("Invalid load_type. Choose 'w', 'm', or 'b'.")

    return model