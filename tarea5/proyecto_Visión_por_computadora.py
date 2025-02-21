import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2

# Ruta del dataset
data_dir = r"C:\Users\huawei\Documents\maestria\UCM Sistemas y control\vision por computadora\tumores_subconjunto"

# Configuración de clases y etiquetas binarias
class_mapping = {
    "glioma": 1,
    "meningioma": 1,
    "no-tumor": 0,
    "pituitary": 0
}

# Recolectar rutas de imágenes con etiquetas
image_paths = []
labels = []
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path) and category in class_mapping:
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            image_paths.append(img_path)
            labels.append(class_mapping[category])

# Mezclar datos
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths[:], labels[:] = zip(*combined)

# Crear generadores de datos con ImageDataGenerator (no flow_from_directory)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generador de entrenamiento
def generate_data(image_paths, labels, batch_size=32):
    while True:
        # Mezclar los datos en cada paso
        indices = np.random.permutation(len(image_paths))
        image_paths = np.array(image_paths)[indices]
        labels = np.array(labels)[indices]

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            images = []
            for img_path in batch_paths:
                image = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
                image_array = img_to_array(image) / 255.0
                images.append(image_array)

            images = np.array(images)
            yield images, batch_labels

train_generator = generate_data(image_paths, labels, batch_size=32)
val_size = int(0.2 * len(image_paths))  # Para la validación
val_generator = generate_data(image_paths[-val_size:], labels[-val_size:], batch_size=32)

# Ajustar arquitectura de red
def cnn_model(input_size=(128, 128, 1)):  # 1 canal (escala de grises)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_size))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Salida binaria
    return model

# Compilar el modelo
model = cnn_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Entrenamiento
history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(image_paths) // 32,
    validation_data=val_generator,
    validation_steps=len(image_paths) // 32,
)

# Graficar pérdida y precisión
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida en validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión en validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    plt.show()

# Mostrar gráficas
plot_training_history(history)

# Función para generar Grad-CAM
def generate_gradcam(model, img_path):
    # Cargar imagen en blanco y negro
    image = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    image_array = img_to_array(image) / 255.0
    image_tensor = tf.expand_dims(image_array, axis=0)

    # Obtener la última capa convolucional
    last_conv_layer = model.get_layer("conv2d_2")

    # Modelo para Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Calcular gradientes
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_tensor)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generar heatmap
    conv_outputs = conv_outputs[0].numpy()
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # Normalizar
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-10)

    # Redimensionar y colorear
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convertir imagen original a RGB para superponer
    image_rgb = cv2.cvtColor(np.uint8(image_array * 255), cv2.COLOR_GRAY2RGB)

    # Superponer heatmap
    superimposed_img = cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)

    # Mostrar resultados
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Tumor Detectado')
    plt.axis('off')
    plt.show()

# Probar Grad-CAM con n imágenes diferentes
def test_gradcam(model, image_paths, n=5):
    count = 0  # Contador de imágenes procesadas
    for i in range(len(image_paths)):
        img_path = image_paths[i]
        
        # Verificar si el nombre de la imagen contiene "glioma"
        if 'glioma' not in img_path.lower():
            continue  # Saltar si no contiene "glioma"
        
        # Procesar solo hasta n imágenes que cumplan la condición
        if count >= n:
            break
        
        print(f"Procesando imagen {count+1}: {img_path}")
        generate_gradcam(model, img_path)
        count += 1

# Probar con 5 imágenes que contengan "glioma"
test_gradcam(model, image_paths, n=5)