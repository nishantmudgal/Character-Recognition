from flask import *
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)  

filename = 'current.png'

def process_image(image_path):
    image_size = 48

    image_arr = cv2.imread('static/current.png', cv2.IMREAD_GRAYSCALE)
    resize_image = cv2.resize(image_arr, (image_size, image_size))
    norm_img = np.zeros((image_size, image_size))
    final_img = cv2.normalize(resize_image,  norm_img, 0, 1, cv2.NORM_MINMAX)

    final_img = final_img.reshape(1, final_img.shape[0], final_img.shape[1], 1)
    print(final_img.shape)
    return final_img 

def predict_image():
    
    loaded_model = load_model("model_cap.h5")
    image_data = process_image(url_for('static', filename=filename))
    p = loaded_model.predict_classes([image_data])

    return chr(int(p)+65)

@app.route('/', methods = ['GET', 'POST'])  
def success():
    global filename  
    if request.method == 'POST':  
        f = request.files['file']
        filename = 'current.png'
        f.save('static/current.png')
    if filename != 'default.png':
        predict = predict_image()
    image_file = url_for('static', filename=filename)
    return render_template("home.html", image_file = image_file, prediction = predict)    
    
if __name__ == '__main__':  
    app.run(debug = True)  