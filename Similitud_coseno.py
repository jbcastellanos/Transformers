from typing import TextIO
from unittest import result
from matplotlib import units
from matplotlib.pyplot import title
from matplotlib.style import context
import requests
import ast
import re
from bs4 import BeautifulSoup
from googlesearch import search
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re, unicodedata
from sentence_transformers import SentenceTransformer, util

""" 
Importante para nuevas integraciones

mport nltk
nltk.download('punkt')
nltk.download('stopwords')
 
"""

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# ---> Funciones de Filtrado

def remove_non_ascii(words):
    """Los carácteres no pertenecientes a ASCI los remueve"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convierte cada palabra en minuscula"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remueve los signos de puntución"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remueve las palabras más comunes"""
    new_words = []
    for word in words:
        if word not in stopwords.words('spanish'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words"""
    stemmer = SnowballStemmer('spanish')
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lematizar palabras en inglés"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    """En normalize se aplica la etapa de filtrado"""
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    #words = stem_words(words)
    return words


# ---> Texto Input 

texto_1 = "Para comprobar el tipo de una variable, puedes utilizar la función type(), que toma la variable como entrada. Dentro de esta función, tienes que pasar el nombre de la variable o el propio valor. Y devolverá el tipo de datos de la variable."
contexto_1 = texto_1
texto_1 = word_tokenize(texto_1) 
texto_1 = normalize(texto_1)

print(texto_1)

# Concatenar listas junto con espacios

texto_procesado = ""
texto_procesado= " ".join(map(str, texto_1))

print(texto_procesado)


# ---> Texto Google 

this_context = 'Para comprobar el tipo de una variable, puedes utilizar la función type(), que toma la variable como entrada. Dentro de esta función, tienes que pasar el nombre de la variable o el propio valor. Y devolverá el tipo de datos de la variable.'
 
texto_2 = word_tokenize(this_context)
texto_2 = normalize(texto_2)
# print(texto_2)

# Establecer similitud a través de TF-IDF

vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform([str(texto_1),str(texto_2)])
similarity_matrix = cosine_similarity(X,X)

#Conversión porcentaje 

resultado = similarity_matrix[1,0] * 100

#Mostrar

print('\nSimilitud:'+str(resultado) + ' %')
#print("\n\n"+this_context)

