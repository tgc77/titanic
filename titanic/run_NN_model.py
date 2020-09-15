from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from . import get_xy_train_model

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(8,)))
model.add(Dropout(rate=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x, y = get_xy_train_model()
model_train = model.fit(x, y, epochs=500, batch_size=50, verbose=0, validation_split=0.06)

plt.plot(model_train.history['accuracy'], label='train')
plt.plot(model_train.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

model.save('titanic_NN.h5')
