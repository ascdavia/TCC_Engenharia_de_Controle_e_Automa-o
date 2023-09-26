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

#Import Data
x1 = pd.read_csv('https://github.com/ascdavia/TCC_Engenharia_de_Controle_e_Automacao/blob/main/Database/sEMG_Basic_Hand_movements_upatras_csv_files/Database_1/df1_mov_all.csv?raw=true', compression = None)
x1 = x1.drop(x1.columns[0], axis=1)

#Define Wavelet Transform Function

def waveletTransformFourLevels(df):
  
  aux_df = ()
  aux_df = pd.DataFrame(aux_df)

  for i in range (len(df)):
    aux = df.loc[i]
    cA4, cD4, cD3, cD2, cD1= pywt.wavedec(aux,'db2', level = 4) 
    cA4 = pd.DataFrame(cA4).T
    cD4 = pd.DataFrame(cD4).T
    cD3 = pd.DataFrame(cD3).T
    cD2 = pd.DataFrame(cD2).T
    cD1 = pd.DataFrame(cD1).T
    aux_df2 = pd.concat([cA4,cD4,cD3,cD2,cD1], axis=1)
    aux_df = pd.concat([aux_df,aux_df2])

  c = list(range(0, 3010, 1))
  aux_df.set_axis(c, axis='columns', inplace=True)
  aux_df = aux_df.reset_index(drop=True)

  return (aux_df)

#Wavelet Transform Application
x = waveletTransformFourLevels(x1)
x = pd.DataFrame(x)

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


# CNN+LSTM
seq_len = 3010
n_classes = 6
n_channels = 1

lr = 0.00001
batch_size = 100
epochs = 150

opt = Adam(learning_rate=lr)

inputs = Input(shape=(seq_len, n_channels))

laywer1 = Conv1D(filters=64, kernel_size=3, activation='relu',padding='same')(inputs)
pool1 = MaxPool1D(pool_size=2, padding='same')(laywer1)
laywer2 = Conv1D(filters=32, kernel_size=2, activation='relu',padding='same')(pool1)
pool2 = MaxPool1D(pool_size=2, padding='same')(laywer2)

lstm1 = LSTM(100, return_sequences=True)(pool2)
dense1 = Dense(6, activation='relu')(lstm1)
#dropout1 = Dropout(0.2)(dense1)

flat = Flatten()(dense1)
dense2 = Dense(n_classes, activation='softmax')(flat)

model_cnn_lstm = Model(inputs=inputs, outputs=dense2)
model_cnn_lstm.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history_cnn_lstm = model_cnn_lstm.fit(x_train, y_train,
  epochs=epochs, 
  batch_size=batch_size, 
  verbose=True, 
  validation_data=(x_val, y_val))


plt.plot(history_cnn_lstm.history['loss'])
plt.plot(history_cnn_lstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

results = model_cnn_lstm.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("test loss, test acc:", results)


pred3 = model_cnn_lstm.predict(x_test)
y_pred3 = pred3.argmax(axis=-1)

cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)

qualidade3 = cm3.diagonal()/cm3.sum(axis=1)
desvio3 = np.std(qualidade3)
print('Qualidade:', qualidade3)
print('Desvio:', desvio3)

df_cm3 = pd.DataFrame(cm3, range(6),range(6))
pp_matrix(df_cm3)