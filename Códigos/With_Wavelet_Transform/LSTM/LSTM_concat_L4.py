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

#LSTM Model

lstm_size = 6          # 3 times the amount of channels
lstm_layers = 1        # Number of layers
lr = 0.0001
batch_size = 100
seq_len = 3010
epochs = 150

n_classes = 6
n_channels = 1

opt = Adam(learning_rate=lr)

lstm_model = Sequential()
lstm_model.add(LSTM(lstm_size, input_shape=(seq_len, n_channels), return_sequences=True))
lstm_model.add(LSTM(lstm_size, return_sequences=True))
lstm_model.add(Flatten())
lstm_model.add(Dense(n_classes, activation='softmax'))
#lstm_model.add(Dropout(0.2))
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Fit LSTM Model

lstm_model.fit(x_train, y_train,
          epochs=epochs, 
          batch_size=batch_size, 
          verbose=True, 
          validation_data=(x_val, y_val))


avaliacao1 = lstm_model.evaluate(x_train,y_train)


pred1 = lstm_model.predict(x_test)
y_pred1 = pred1.argmax(axis=-1)

cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)

qualidade1 = cm1.diagonal()/cm1.sum(axis=1)
desvio1 = np.std(qualidade1)
print('Qualidade:', qualidade1)
print('Desvio:', desvio1)

df_cm1 = pd.DataFrame(cm1, range(6),range(6))
pp_matrix(df_cm1)