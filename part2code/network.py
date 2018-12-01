#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Utiliza la memoria que necesita de manera dinamica, puede ser o no en bloque.
config.gpu_options.per_process_gpu_memory_fraction = 1.0 #40%de la ram,
session = tf.Session(config=config)

import pandas as pd
import numpy as np
import re, os, sys

import nltk
import nltk.corpus

import keras

# <a id="cuarto"></a>
# ## 3. CNN sobre texto
# 
# Cuando oimos sobre redes neuronales convolucionales (CNN) normalmente pensamos en visión artificial. Las CNN fueron responsables de los principales avances en la clasificación de imágenes y son el núcleo de la mayoría de los sistemas de *Computer Vision* en la actualidad, desde el etiquetado automático de fotos de Facebook hasta los autos que conducen por sí mismos.
# 
# Más recientemente, también hemos empezado a aplicar CNN a problemas de procesamiento del lenguaje natural (NLP) y hemos obtenido resultados interesantes. Como sabemos, las redes convolucionales tienen importantes ventajas como invarianza a rotaciones y traslaciones así como la conectividad local (características de nivel inferior en una representación de nivel superior), además de lo que las hace fuertemente ventajosas, el **compartir** parámetros.
# 
# 
# **¿Cómo se aplica esto a NLP?**  
# En esta experimentación apicaremos una red CNN al dataset  __[Adzuna](https://www.kaggle.com/c/job-salary-prediction)__ que contiene cientos de miles de registros que en su mayoría corresponden a texto no estructurado versus sólo unos pocos estructurados. Los registros pueden estar en varios formatos diferentes debido a los cientos de diferentes fuentes de registros, los cuales corresponden a anuncios de empleadores en busca de trabajadores.  
# Es decir, cada fila es un anuncio que, en estricto rigor, representa una sentencia típicamente trabajada como vectores de word embeddings como **word2vec** o **GloVe**. Así, para una frase de 10 palabras bajo representaciones de *embeddings* utilizando 100 dimensiones tendríamos una matriz de 10 × 100 como entrada, lo que simularía nuestra "imagen".
# 
# 
# Su tarea es entonces, predecir el salario (valor continuo) de un determinado anuncio en base al texto indicado en éste. Igualmente puede valerse de otros atributos del anuncio como por ejemplo la ubicación, tipo de contrato, etc. 
# 
# 
# A continuación se presenta un código de guía para leer los archivos y pre-procesarlos. Deberá añadir y realizar lo que estime conveniente.

# In[2]:


# ### Embeddings 
# 
# En lugar de entrenar nuestros vectores embeddings utilizaremos el archivo __[Glove](https://www.kaggle.com/terenceliu4444/glove6b100dtxt#glove.6B.100d.txt)__ el cual cuenta con las representaciones vectoriales (de dimensionalidad 100) ya entrenadas sobre una amplia base de datos. Puede encontrar más detalle en https://nlp.stanford.edu/projects/glove/

# In[6]:

x = np.load("all/x.npy")
embedd_matrix = np.load("all/e.npy")
y = np.load("all/y.npy")


# In[9]:


print("x")
print(x)
print("y")
print(np.array(y))
print("embedd_matrix.shape")
print(embedd_matrix.shape)


# ### Modelo

# In[10]:

indexes = np.arange(x.shape[0])
np.random.shuffle(indexes)
x = x[indexes]
y = y[indexes]

x_tr = x[:210000]
y_tr = y[:210000]
x_te = x[210000:]
y_te = y[210000:]


# In[58]:


def model_salary():
    max_input_length = 54
    inlayer = keras.layers.Input(shape=(max_input_length,))
    front = inlayer
    #
    front = keras.layers.Embedding(input_dim=embedd_matrix.shape[0],output_dim=100,
        weights=[embedd_matrix],
        input_length=max_input_length,trainable=False)(front)
    #
    n_filters = [130,200,280]
    pool_sizes = [2,2,2]
    ksizes = [5,4,3]
    for i in range(len(n_filters)):
        for _ in range(2):
            front = keras.layers.Conv1D(n_filters[i],ksizes[i],padding='same',
                activation='relu',kernel_initializer='he_uniform')(front)
        front = keras.layers.MaxPooling1D(pool_size=pool_sizes[i])(front)
        front = keras.layers.BatchNormalization()(front)
        #if i>=1: front = keras.layers.Dropout(0.2)(front)
    #
    front = keras.layers.Flatten()(front)
    front = keras.layers.BatchNormalization()(front)
    for k in range(3):
        front = keras.layers.Dense(1000-k*100,activation='relu',
                                  kernel_initializer='he_uniform')(front)
        front = keras.layers.BatchNormalization()(front)
        #front = keras.layers.Dropout(0.25)(front)
    front = keras.layers.Dense(1,activation='tanh',
                              kernel_initializer='glorot_uniform')(front)
    mean = np.mean(y_tr)
    std = np.std(y_tr)
    
    front = keras.layers.Lambda(lambda x: mean+5*x*std)(front)
    #
    model = keras.models.Model(inputs=inlayer,outputs=front)
    return model


# In[59]:


model = model_salary()
model.summary()


# In[61]:


EPOCHS = 60

model = model_salary()

optimizer = keras.optimizers.Adam(lr=0.01,decay=1.0/EPOCHS)

model.compile(loss='mean_absolute_error',optimizer=optimizer)

history = model.fit(x_tr,y_tr,validation_data=(x_te,y_te),
          epochs=EPOCHS,batch_size=350)

model.save("model.h5")


# In[ ]:


import json

with open('history.json', 'w') as f:
    json.dump(history.history,f)


# ### Evaluación de predicciones
# Para las predicciones evalúe la métrica *Mean Absolute Error* (MAE)
# 
# ```python
# from sklearn.metrics import mean_absolute_error
# print("MAE on train: ",mean_absolute_error(y_train, model.predict(Xtrain)))
# print("MAE on validation: ",mean_absolute_error(y_val, model.predict(Xval)))
# ```
# 
# > **Intente resolver el problema experimentando con las ayudas que se entregan en el código y lo aprendido hasta ahora en el curso. Se espera que llegue a un MAE menor a 7000 en el conjunto de pruebas. No olvide documentar todo lo experimentando en este Informe Jupyter así como el argumento de sus decisiones.**

# In[ ]:




