FROM python:3.11-slim

RUN pip install --no-cache-dir pandas scikit-learn joblib nltk

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Copy inference code and utils
COPY src/utils /opt/ml/code/utils
COPY src/inference.py /opt/ml/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code
