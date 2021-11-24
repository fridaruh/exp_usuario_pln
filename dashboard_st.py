import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.text('Tablero de visualización de comentarios de detractores y promotores')

df = pd.read_csv('./Womens Clothing E-Commerce Reviews.csv')

df = df[['Review Text','Recommended IND']]

number_head = st.slider('Cuántos datos quieres mostrar',0,10)
df.head(number_head)

df_detractores = df[df['Recommended IND']==0] 
df_promotores = df[df['Recommended IND']==1] 

number_head_prom =st.slider('Cuántos datos quieres mostrar de los promotores',0,10)

df_detractores.sample(number_head_prom)

number_head_detract =st.slider('Cuántos datos quieres mostrar de los detractores',0,10)

df_promotores.sample(number_head_detract)

df_promotores.dropna(inplace=True)
df_detractores.dropna(inplace=True)

df_detractores_list = df_detractores['Review Text'].tolist()
df_promotores_list = df_promotores['Review Text'].tolist()

pattern = r'''(?x)                  # Flag para iniciar el modo verbose
              (?:[A-Z]\.)+          # Hace match con abreviaciones como U.S.A., U.K.
              | \w+(?:-\w+)*        # Hace match con palabras que pueden tener un guión interno (10-20),  e-mail
              | \$?\d+(?:\.\d+)?%?  # Hace match con dinero o porcentajes como $15.5 --> 15, 5 o 100% --> 100
              | \.\.\.              # Hace match con puntos suspensivos
              | [][.,;"'?():-_`]    # Hace match con signos de puntuación 
'''

import nltk
nltk.download('punkt')
from nltk import word_tokenize

texto_promotores = []

for x in range(0, len(df_promotores_list)):
  token_1 = df_promotores_list[x].lower() #convierte a minúsculas
  token_2 = nltk.regexp_tokenize(token_1, pattern) #Quita los patrones definidos arriba y genera tokens
  texto_promotores.append(token_2)

texto_detractores = []

for x in range(0, len(df_detractores_list)):
  token_1 = df_detractores_list[x].lower() #convierte a minúsculas
  token_2 = nltk.regexp_tokenize(token_1, pattern) #Quita los patrones definidos arriba y genera tokens
  texto_detractores.append(token_2)

import string
puntuacion = list(string.punctuation)

nltk.download('stopwords')

newStopWords = nltk.corpus.stopwords.words('english')

df_promotores_limp = [w for w in texto_promotores if w not in newStopWords]
df_promotores_limp_2 = [w for w in df_promotores_limp if w not in puntuacion]
flatten_promotores = [w for l in df_promotores_limp_2 for w in l ]

df_detractores_limp = [w for w in texto_detractores if w not in newStopWords]
df_detractores_limp_2 = [w for w in df_detractores_limp if w not in puntuacion]
flatten_detractores = [w for l in df_detractores_limp_2 for w in l ]

from wordcloud import WordCloud
import matplotlib.pyplot as plt

freqwords_detractores = nltk.FreqDist(flatten_detractores)
freqwords_promotores = nltk.FreqDist(flatten_promotores)

wordcloud = WordCloud(background_color= 'white', collocations=False, max_words=50).fit_words(freqwords_detractores)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud = WordCloud(background_color= 'white', collocations=False, max_words=50).fit_words(freqwords_promotores)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()