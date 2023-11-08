import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model("smote_cnnAlzheimer.h5")

# Load and preprocess the custom image
custom_image_path = "dataset/test/NonDemented/26 (74).jpg"  # Replace with the path to your custom image
custom_img = image.load_img(custom_image_path, target_size=(128, 128))
custom_img = image.img_to_array(custom_img)
custom_img = np.expand_dims(custom_img, axis=0)
custom_img = custom_img / 255.0  # Normalize the image

# Predict the class and get the probability
predictions = model.predict(custom_img)
predicted_class = np.argmax(predictions)

class_names =['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
# Print the results
print("Predicted Class:", class_names[predicted_class])

for i, class_name in enumerate(class_names):
    print(f"{class_name} Probability: {predictions[0][i]:.4f}")