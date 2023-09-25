#pip install pretty_confusion_matrix

from IPython.core.display import Pretty
#Data
import pandas as pd 
import numpy as np

#Wavelet Transform 
import pywt

#Train and Test Split
from sklearn.model_selection import train_test_split

#CNN and LSTM Model
import keras 
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv1D, LSTM, Input, MaxPool1D, MaxPooling1D, Dropout, AvgPool1D, Reshape, Concatenate, Dense, Flatten

#Metrics
from sklearn.metrics import confusion_matrix, recall_score, precision_score

#Visualization 
import seaborn as sns
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix
from keras.utils.vis_utils import plot_model

#Import Data
x1 = pd.read_csv('https://github.com/ascdavia/TCC_Engenharia_de_Controle_e_Automacao/blob/main/Database/sEMG_Basic_Hand_movements_upatras_csv_files/Database_1/df1_mov_all.csv?raw=true', compression = None)
x = x1.drop(x1.columns[0], axis=1)

#Reshape
x = x.values.reshape(x.shape[0], x.shape[1], 1)

#Labels
base = np.ones((150,1), dtype=np.int64)
m_cyl = base*0
m_hook = base*1
m_lat = base*2
m_palm = base*3
m_spher = base*4
m_tip = base*5

y = np.vstack([m_cyl,m_hook,m_lat,m_palm,m_spher,m_tip])
#y = pd.DataFrame(y)

#Train, test and validation split
x_train, x_aux, y_train, y_aux = train_test_split(x,y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_aux, y_aux, test_size=0.50, random_state=23)

#CNN Model

batch_size = 100
seq_len = 3000
lr = 0.00001
epochs = 2500

n_classes = 6
n_channels = 1

opt = Adam(learning_rate=lr)

cnn_model = Sequential()
cnn_model.add(Conv1D(filters=4, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(seq_len, n_channels)))
cnn_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
cnn_model.add(Conv1D(filters=8, kernel_size=2, strides=1, padding='same', activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
cnn_model.add(Conv1D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
cnn_model.add(Conv1D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
cnn_model.add(Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
cnn_model.add(Conv1D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Flatten())
cnn_model.add(Dense(n_classes, activation='softmax'))
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

plot_model(cnn_model, to_file='model.png')

#Fit CNN Model
history_cnn = cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=True, validation_data=(x_val, y_val))

plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

avaliacao = cnn_model.evaluate(x_train,y_train)

pred = cnn_model.predict(x_test)
y_pred = pred.argmax(axis=-1)

cm = confusion_matrix(y_test, y_pred)
print(cm)

qualidade = cm.diagonal()/cm.sum(axis=1)
desvio = np.std(qualidade)
print('Qualidade:', qualidade)
print('Desvio:', desvio)

df_cm = pd.DataFrame(cm, range(6),range(6))
pp_matrix(df_cm)