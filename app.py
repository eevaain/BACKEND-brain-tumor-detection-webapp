#putting in da libraries YEAH YEAH!?
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image 

import tensorflow as tf
import numpy as np

#instantiating flask
print(tf.__version__)
app = Flask(__name__)
CORS(app)

#defining route 
@app.route("/test", methods=['POST'])
def test():
    fetchedImageData = request.files.get('file','')
    print(fetchedImageData)
    openImage = Image.open(fetchedImageData) #currently card images dont work...
    # openImage.show()

    resized_img = openImage.resize((256, 256)) #this works!!!!
    # resized_img.show() #now am i able to convert it to an array....?

    photoToArray = np.asarray(resized_img)
    # print(photoToArray)

    loaded_model = tf.keras.models.load_model('/app/saved_brain_tumor_cnn_model.h5')
    # test_image = image.load_img(photo, target_size = (256, 256))
    # test_image = image.img_to_array(test_image)
    # ^^ converts PIL image to numpy array
    test_image = np.expand_dims(photoToArray, axis = 0)
    result = loaded_model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Brain Tumor Present'
    else:
        prediction = 'No Brain Tumor'

    print("Predicted:", prediction)
    return jsonify({'Prediction': prediction})


if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=True, port=5000, debug=True) 



