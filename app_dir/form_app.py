from flask import Flask, render_template, request, jsonify
import pickle
from keras.models import load_model
from keras.preprocessing import image
import pdb
from werkzeug.utils import secure_filename
import os
import numpy as np
from large_image_parse import run_all

app = Flask(__name__, static_url_path='/static')


# with open('static/final_model_12_15_18_1.hdf5', 'rb') as f:
model = load_model('static/final_model_12_15_18_1.hdf5')
# model = load_model('models/classifier.h5')
model._make_predict_function()




@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('form/index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('form/submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the image to be classified from an input form and use the
    model to classify.
    """
    dir_path = os.path.dirname(os.path.realpath('./')) + '/app_dir'

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_loc = dir_path + '/static/images_f'+'/'+filename
    file.save(file_loc)

    img = image.load_img(file_loc)
    if img.size[0] < 512:

        img = image.load_img(file_loc, target_size=(256,256))

        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # # Use the loaded model to generate a prediction.
        pred = model.predict(img_tensor)

        # # Prepare and send the response.
        label = np.argmax(pred)
        if pred <= 0.1:
            prediction = {'Classified no Crack: ':pred}
        else:
            prediction = {'Classified as having a Crack: ':pred}


        return render_template('form/predict.html', upload_file=filename, predicted=prediction)

    else:
        run_all(model, img, file_save=dir_path+'/static/images_predicted/'+filename )
        return render_template('form/predict2.html', upload_file=filename)



@app.route('/extra', methods=['POST'])
def extra():

    render_template('form/extra.html', article=file_loc, predicted=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
