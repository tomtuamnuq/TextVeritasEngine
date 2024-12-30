FROM python:3.11-slim

RUN pip install --no-cache-dir scikit-learn==1.6.0 pandas==2.2.3

# Copy training code
COPY src/utils /opt/ml/utils
COPY src/train.py /opt/ml/

WORKDIR /opt/ml
ENTRYPOINT ["python", "train.py"]
