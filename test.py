from keras.models import load_model
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


model = load_model('model2.h5')
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
pred = model.predict_classes(test_data)
print("Confusion Matrix")
cnf_matrix = confusion_matrix(test_labels,pred)
print(cnf_matrix)
pd.set_option('display.max_colwidth', -1)
df_cm = pd.DataFrame(cnf_matrix, range(2),range(2))
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,fmt = 'g')
plt.title('Confusion Matrix')
plt.show()
