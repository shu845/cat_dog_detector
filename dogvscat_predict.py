import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras import backend as K
import cv2 as cv
import os.path

f_model = './model/dogvscat'
model_filename = 'dogvscat_model.json'
weights_filename = 'dogvscat_model_weights.hdf5'

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

image_path = input('Give me a path of a image file.\r\n')
if(os.path.isfile(image_path)):
    model = model_from_json(open(os.path.join(f_model,model_filename)).read())
    model.load_weights(os.path.join(f_model,weights_filename))
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
    image = cv.resize(image, (128, 128))
    image = image.transpose(2,0,1)
    image = image/255
     
    image=image.reshape(1, 128, 128, 3)

    score = model.predict(image, verbose=0)
    c_score = score.argmax(axis=-1)
    kind = ['cat', 'dog']

    print('This image is ' + str(kind[c_score[0]]) + ' with ' + str(score[0][c_score[0]]*100) + '%')
    print('This image is ' + str(kind[1-c_score[0]]) + ' with ' + str(score[0][1-c_score[0]]*100) + '%')
    K.clear_session()
else:
    print('This image file isn\'t a valid file. ')
    K.clear_session()
