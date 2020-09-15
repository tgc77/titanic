import numpy as np
from keras.models import load_model

model_predict = load_model('titanic_NN.h5')
x_example = np.array([[1,0,20,2,2, 10, 1, 2]])

prediction_num = model_predict.predict(x_example)

print(prediction_num)

if prediction_num < 0.5:
  prediction = 'Not survived!'
else:
  prediction = 'Survived!'

print(prediction)
