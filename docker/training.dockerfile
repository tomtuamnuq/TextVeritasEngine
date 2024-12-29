FROM python:3.11-slim

RUN pip install --no-cache-dir pandas scikit-learn joblib

# Copy training code
COPY src/utils /opt/ml/code/utils
COPY src/train.py /opt/ml/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code
ENTRYPOINT ["python", "train.py"]
