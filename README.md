# Fake News Detection Project

## Overview

This project implements a machine learning system for detecting fake news articles using natural language processing and classification techniques. The model achieves 94% accuracy using Logistic Regression with TF-IDF feature extraction, and is deployed as a Flask API endpoint for real-time inference.

## Problem Statement

The spread of fake news has become a significant challenge in today's digital age. This project aims to automatically identify potentially fake news articles using machine learning techniques, helping users make more informed decisions about the content they consume.

## Dataset

The training data combines three distinct datasets:

- Combined dataset size: 50,000+ news articles
- Mix of verified real and fake news articles
- Diverse sources to ensure model robustness
- Features include article titles and full text content

### Dataset Setup

This project uses three datasets from Kaggle focused on fake news detection. Follow these steps to set up the data:

1. Download datasets from Kaggle:

   - [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data)
   - [English Fake News Dataset](https://www.kaggle.com/datasets/evilspirit05/english-fake-news-dataset/data)
   - [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

2. Create a `data` directory in the project root:

   ```bash
   mkdir data
   ```

3. Extract the downloaded files to the `data` directory:
   ```
   data/
   ├── Fake.csv
   ├── True.csv
   ├── final_en.csv
   └── WELFake_Dataset.csv
   ```

Note: You need a [kaggle.com](https://www.kaggle.com) account to download the datasets.

## Exploratory Data Analysis

Key findings from the EDA phase:

- Identified significant keywords and phrases commonly associated with fake news
- Found "Donald Trump" as a prominent feature in the dataset
- Analyzed text length distributions and linguistic patterns

## Model Development

### Feature Engineering

- Text preprocessing including tokenization and stopword removal
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Length of title and text

### Model Selection

Compared three machine learning models:

| Model               | Accuracy | Std Dev |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.893    | 0.185   |
| Naive Bayes         | 0.801    | 0.186   |
| Random Forest       | 0.900    | 0.216   |

### Model Performance

The Logistic Regression model provides a good balance between model complexity and performance:

- Accuracy: 94% with Simple hyperparameter tuning
- Number of features in TF-IDF vectorization is essential
- 5-Fold Cross Validation

### Running Scripts Locally

```bash
# Create a virtual environment
micromamba create -n veritas python=3.11

# Activate environment
micromamba activate veritas

# Install required packages
micromamba install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud textblob jupyter

# Add the jupyter kernel
micromamba run -n veritas python -m ipykernel install --user --name veritas --display-name "Python 3.11"

# Explore the EDA and model_selection notebooks in the notebooks directory
cd notebooks
jupyter lab

# Run preprocessing script
python src/preprocess.py --input-data-dir data/ --output-data-dir data/

# Run model training script (hyperparameter search)
python src/train.py --data-dir data/ --model-dir model/ --n-iter 3
```

### Running Preprocessing Script inside Docker container

```bash
docker build -t preprocessing -f docker/preprocessing.dockerfile .

docker run -v $(pwd)/data:/opt/ml/processing/input \
          -v $(pwd)/data:/opt/ml/processing/output \
          preprocessing \
          --input-data-dir /opt/ml/processing/input \
          --output-data-dir /opt/ml/processing/output
```

### Perform Hyperparameter optimization inside Docker container

```bash
docker build -t training -f docker/training.dockerfile .

docker run -v $(pwd)/data:/opt/ml/input \
          -v $(pwd)/model:/opt/ml/model \
          training \
          --data-dir /opt/ml/input \
          --model-dir /opt/ml/model \
          --n-iter 5
```

## Deployment

The model is deployed as a Flask API endpoint:

- RESTful API interface
- JSON input/output format
- Docker containerization for easy deployment
- Health monitoring endpoints

### Test inference inside Docker container

```bash
docker build -t inference -f docker/inference.dockerfile .

docker run -p 8080:8080 -v $(pwd)/model:/opt/ml/model inference

python test/test_predict.py
```

Check the Flask app logs from the docker `inference` container. The `test_predict.py` script should return

`{'fake': 1, 'probability': {'fake': 0.8072652133649492, 'real': 0.19273478663505084}}`

Feel free to change the input data (title and text) inside the `test_predict.py` script.
