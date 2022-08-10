from typing import TextIO
from unittest import result
from matplotlib import units
from matplotlib.pyplot import title
from matplotlib.style import context
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re, unicodedata
from sentence_transformers import SentenceTransformer, util

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

texto_1 = "La fracturación hidráulica, fractura hidráulica, hidrofracturación o simplemente fracturado es una técnica para posibilitar o aumentar la extracción de gas y petroleo del subsuelo, siendo una de las técnicas de estimulación de pozos en yacimientos de hidrocarburos."
contexto_1 = texto_1
texto_1 = word_tokenize(texto_1) 
texto_1 = normalize(texto_1)

texto_procesado = ""
texto_procesado= " ".join(map(str, texto_1))
#print(texto_procesado)

# ---> Texto Google 

this_context ='La fracturación hidráulica o fracking es una técnica que permite extraer el llamado gas de esquisto, un tipo de hidrocarburo no convencional que se encuentra literalmente atrapado en capas de roca, a gran profundidad (ver animación arriba y gráfico a la derecha). Luego de perforar hasta alcanzar la roca de esquisto, se inyectan a alta presión grandes cantidades de agua con aditivos químicos y arena para fracturar la roca y liberar el gas, metano. Cuando el gas comienza a fluir de regreso lo hace con parte del fluido inyectado a alta presión. La fracturación hidráulica no es nueva. En el Reino Unido se utiliza para explotar hidrocarburos convencionales desde la década del 50. Pero sólo recientemente el avance de la tecnología y la perforación horizontal permitió la expansión a gran escala del fracking, especialmente en EE.UU., para explotar hidrocarburos no convencionales.'
 
texto_2 = word_tokenize(this_context)
texto_2 = normalize(texto_2)
#print(texto_2)

texto_procesado_2 = ""
texto_procesado_2 = " ".join(map(str, texto_2))

#print(texto_procesado_2)

# Establecer similitud a través de TF-IDF

vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform([str(texto_1),str(texto_2)])
similarity_matrix = cosine_similarity(X,X)

#Conversión porcentaje 

resultado = similarity_matrix[1,0] * 100

#Mostrar

print('\nSimilitud:'+str(resultado) + ' %')
#print("\n\n"+this_context)


model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode(texto_procesado_2)

passage_embedding = model.encode(['La fracturación hidráulica, fractura hidráulica, ​ hidrofracturación​ o simplemente fracturado es una técnica para posibilitar o aumentar la extracción de gas y petroleo del subsuelo, siendo una de las técnicas de estimulación de pozos en yacimientos de hidrocarburos.',
                                  'Es uno de los mejores jugadores de fútbol',
                                  'La inteligencia artificial (IA) hace posible que las máquinas aprendan de la experiencia, se ajusten a nuevas aportaciones y realicen tareas como seres humanos.',
                                  'Es un deportista de alto nivel',
                                  texto_procesado])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))