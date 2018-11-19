# curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from PIL import Image
import numpy as np
import flask
import io
import os
import sys
import boto3
import json
from botocore.exceptions import ClientError

default_card_set = "3ed"


with open('config.json') as f:
    credentials = json.load(f)

s3 = boto3.resource('s3', aws_access_key_id=credentials["accessKeyId"], aws_secret_access_key=credentials["secretAccessKey"])

if not os.path.exists('models'):
    os.makedirs('models')

if len(sys.argv) > 1:
    print("Card Set: ", sys.argv[1])
    card_set = sys.argv[1]
else:
    print("Default Card Set: ", default_card_set)
    card_set = default_card_set


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
labels = None

def load_keras_model():
    global model
    global labels

    # Check AWS if a model already exists
    print("checking for existing models")
    latest_val_err = 1000.0
    latest_file = ""
    my_bucket = s3.Bucket("model-" + card_set)
    for obj in my_bucket.objects.all():
        if obj.key == '_labels.json':
            continue
        split_files = obj.key.split('.hdf5')[0].split('-')
        if float(split_files[2]) < latest_val_err:
            latest_val_err = float(split_files[2])
            latest_file = obj.key

    if latest_file == "":
        print("could not load model")
        sys.exit()

    if os.path.isfile(os.path.join('models', latest_file)) != True:
        try:
            print("downloading existing model")
            my_bucket.download_file(latest_file, os.path.join('models', latest_file))
        except ClientError as e:
            print("The object does not exist.")
        print('loading existing model into memory')
        model = load_model(os.path.join('models', latest_file))
    else:
        print('existing model is current')
        model = load_model(os.path.join('models', latest_file))

    # load labels each time
    try:
        my_bucket.download_file('_labels.json', os.path.join('models', '_labels.json'))
    except:
        print("problem downloading labels")
        sys.exit()

    # load labels into memory
    try:
        with open('models/_labels.json') as f:
            labels = json.load(f)
            print(labels)
    except:
        print("could not parse labels")
        sys.exit()

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(400, 400))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image, batch_size=1)
            data["predictions"] = []
            data["prediction"] = labels[str(np.argmax(preds[0]))]

            print(labels)
            print(type(labels))

            for result in preds[0]:
                probability = float(result)
                if probability > 0.0:
                    r = {"label": labels[str(np.argmax(preds[0]))], "probability": probability}
                    data["predictions"].append(r)

            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_keras_model()
    app.run(debug = False, threaded = False)

