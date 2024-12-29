FROM python:3.11-slim
RUN pip install --no-cache-dir pandas nltk
ENV PYTHONUNBUFFERED=TRUE

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Get code for preprocessing
COPY src/utils /opt/ml/utils
COPY src/preprocess.py /opt/ml/

ENTRYPOINT ["python3"]
