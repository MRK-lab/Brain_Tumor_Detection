import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Modeli yükle
model = load_model("brain_tumor_classifier_model.h5")
# model.summary()

# Sınıf adları (Modeline göre değişebilir!)
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # modele göre giriş boyutu
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)
    img_array /= 255.0  # Normalizasyon

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]

    print(f"Resim: {os.path.basename(img_path)}")
    print(f"→ Tahmin Edilen Sınıf: {predicted_label}")
    print(f"→ Güven: %{confidence * 100:.2f}")

# Örnek: Tek bir resim dosyasını test et
predict_image("Te-pi_0241.jpg")
