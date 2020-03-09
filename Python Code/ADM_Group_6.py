
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf


# In[10]:


os.chdir('E:/ADM_New/urban-sound-classification')
train=pd.read_csv('train.csv',nrows=113)
train.tail()


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

Class = train.Class.value_counts()
colours = ["#aaaaaa", "#aaaaaa", "#aaaaaa","#aaaaaa","#aaaaaa","#d11111","#aaaaaa","#aaaaaa","#aaaaaa","#d11111"]
f, ax = plt.subplots(figsize=(18,5)) 
ax = sns.countplot(x='Class', data=train, palette=colours)
plt.title('Class Distribution')
plt.show()


# In[11]:


import IPython.display as ipd
audio_path = 'train/Train/2.wav'
ipd.Audio(audio_path)


# In[35]:


import librosa.display

x, sr = librosa.load(audio_path)

# Plot the sample.
plt.figure(figsize=(12, 5))
librosa.display.waveplot(x, sr=sr)

plt.show()




# In[76]:


plt.figure(figsize=(12, 5)) 

plt.plot(x[1000:1100])
plt.grid()
n_crossings = librosa.zero_crossings(x[1000:1100], pad=False) 
print(f'Number of crosses: {sum(n_crossings)}')
plt.show()


# In[13]:


import soundfile as sf

def mean_mfccs(x):
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

def parse_audio(x):
    return x.flatten('F')[:x.shape[0]] 

def get_audios():
    train_path = "train/Train/"
    train_file_names = os.listdir(train_path)
    train_file_names.sort(key=lambda x: int(x.partition('.')[0]))
    
    samples = []
    for file_name in train_file_names:
        x, sr = sf.read(train_path + file_name, always_2d=True)
        x = parse_audio(x)
        samples.append(mean_mfccs(x))
        
    return np.array(samples)

def get_samples():
    df = pd.read_csv('train.csv')
    return get_audios(), df['Class'].values


# In[14]:


NewX, y = get_samples()


# In[15]:


len(y)


# In[16]:


NewX.shape


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(NewX, y)


# In[18]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# # model 1

# In[19]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train_scaled,y_train)


# In[22]:


from sklearn.metrics import confusion_matrix, classification_report
print(f'Model Score: {rf.score(x_test_scaled, y_test)}')

y_predict_rf = rf.predict(x_test_scaled)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict_rf, y_test)}')


print(classification_report(y_predict_rf, y_test))


# # model 2

# In[23]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train_scaled,y_train)


# In[24]:


print(f'Model Score: {knn.score(x_test_scaled, y_test)}')

y_predict_knn = knn.predict(x_test_scaled)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict_knn, y_test)}')


print(classification_report(y_predict_knn, y_test))


# In[36]:


from sklearn.svm import SVC
svmModel = SVC()
svmModel.fit(x_train_scaled,y_train)


# In[37]:


print(f'Model Score: {knn.score(x_test_scaled, y_test)}')

y_predict_svm = svmModel.predict(x_test_scaled)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict_knn, y_test)}')


print(classification_report(y_predict_svm, y_test))


# In[38]:


x_train_scaled.shape


# In[79]:


from sklearn.preprocessing import OneHotEncoder
lb = OneHotEncoder()
y_train_encoded = lb.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = lb.transform(y_test.reshape(-1, 1))


# # Deep Learning model

# In[80]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_labels = 10
filter_size = 2

# build model
model = Sequential()

model.add(Dense(256, input_shape=(20,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[81]:


history = model.fit(x_train_scaled, y_train_encoded, batch_size=32, epochs=10, validation_data=(x_test_scaled, y_test_encoded),
                   verbose = 0)


# In[82]:


#4076/4076 [==============================] - 1s 196us/step - loss: 0.3031 - acc: 0.8989 - val_loss: 0.2627 - val_acc: 0.9249


# In[83]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()

