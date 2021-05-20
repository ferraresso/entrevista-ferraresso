import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from nlp_transformer import NLPCleaner
import numpy as np
import pandas as pd

@st.cache
def load_model(allow_output_mutation=True):
    return joblib.load('pipe.pkl')

@st.cache(allow_output_mutation=True)
def load_data():
	return pd.read_csv('data/training_data.csv')
st.title('Demo app review peliculas')


model= load_model()
df = load_data()
modo = st.radio(
		"Seleccione como desea probar",
		('Usar ejemplo del dataset', 'Escribir una review'))


if modo == 'Usar ejemplo del dataset':
	if st.button('Generar'):
		s = df.sample()
		st.write(s.review.values[0][:500])
		
		result = model.predict(s.review)[0]
		st.markdown('**Es un comentario positvo!**' if result else '**Es un comentario negativo**')
elif modo == 'Escribir una review':
	s = st.text_input('Ingrese un comentario (ej: "muy buena pelicula, divertida, entretenida, llena de comedia)"')
	if len(s) > 2:
		result = model.predict([s])[0]
		st.markdown('**Es un comentario positvo!**' if result else '**Es un comentario negativo**')