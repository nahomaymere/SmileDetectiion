import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.metrics import confusion_matrix



#load data
labels = np.genfromtxt("genki4k\\labels.txt", usecols=0)
data = []
path = "genki4k\\files"

for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path,filename))
    img = cv2.resize(img,dsize=(64,64), interpolation = cv2.INTER_CUBIC)
    data.append(img)
data = np.array(data,dtype = 'float')
data /= 255.0

np.save("data.npy",data)
tr_data, test_data, tr_labels, test_labels = train_test_split(data, labels, stratify = labels,test_size = 0.2)
model = Sequential()
model.add(Conv2D(32,3,input_shape = (64,64,3), padding='same'))#output 64*64*32
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))#output 32*32*32
model.add(Conv2D(32,3, padding='same'))#32*32*32
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))#16*16*32
model.add(Conv2D(64,3, padding='same'))#16*16*32
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))#8*8*32
model.add(Conv2D(32,3, padding='same'))#16*16*32
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))#8*8*32
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


from keras.optimizers import SGD
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(tr_data, tr_labels, validation_data=(test_data,test_labels),epochs=50)
model.save("model2.h5")
# summarize history for accuracy
pred = model.predict_classes(test_data)
print("Confusion Matrix")
cnf_matrix = confusion_matrix(test_labels,pred);


print(cnf_matrix)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
