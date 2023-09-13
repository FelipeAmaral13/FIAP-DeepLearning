import os
from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from keras.preprocessing import image


app = Flask(__name__)
model = keras.models.load_model(os.path.join("model", "model.h5"))

# Mapeamento de classes
class_mapping = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato__Target_Spot',
    12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy'
}

# Define a função para processar a imagem e fazer a previsão
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(227, 227))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_name = class_mapping[predicted_class]
    return class_name, prediction[0][predicted_class] * 100

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='Nenhum arquivo selecionado')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', message='Nenhum arquivo selecionado')
        
        if file:
            file_path = os.path.join("static/uploads", file.filename)
            file.save(file_path)
            class_name, confidence = predict_image(file_path)
            result = f'Classe: {class_name}, Probabilidade: {confidence:.2f}%'
            return render_template('index.html', message='Arquivo carregado com sucesso', result=result, image_path=file_path)
    
    return render_template('index.html', message='Faça o upload de uma imagem')

if __name__ == '__main__':
    app.run(debug=True)
