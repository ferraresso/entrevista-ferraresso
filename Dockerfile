# base image
FROM python:3.7

# streamlit-specific commands
#RUN mkdir -p /root/.streamlit
#RUN bash -c 'echo -e "\
#[general]\n\
#email = \"\"\n\
#" > /root/.streamlit/credentials.toml'
#RUN bash -c 'echo -e "\
#[server]\n\
#enableCORS = false\n\
#" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# copy over and install packages
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# copying everything over
COPY demo.py ./demo.py
COPY data/training_data.csv data/training_data.csv
COPY pipe.pkl ./pipe.pkl
COPY lemmatization-es.csv ./lemmatization-es.csv
COPY nlp_transformer.py ./nlp_transformer.py

RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]

# run app
CMD streamlit run demo.py