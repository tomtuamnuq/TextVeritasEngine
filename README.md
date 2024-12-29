# TextVeritasEngine

## Dataset Setup

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

Note: You need a Kaggle account to download the datasets. If you don't have one, sign up at [kaggle.com](https://www.kaggle.com).

## Running Scripts Locally

```bash
# Create a virtual environment
micromamba create -n veritas python=3.11

# Activate environment
micromamba activate veritas

# Install required packages
micromamba install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud textblob jupyter

# Add the jupyter kernel
micromamba run -n veritas python -m ipykernel install --user --name veritas --display-name "Python 3.11"

# Run preprocessing script
python src/preprocess.py --input-data-dir data/ --output-data-dir data/

# Run model training script (hyperparameter search)
python src/train.py --data-dir data/ --model-dir model/


```

## Running Preprocessing Script inside Docker container

```bash
docker build -t preprocessing -f docker/preprocessing.Dockerfile .

docker run -v $(pwd)/data:/opt/ml/processing/input \
          -v $(pwd)/data:/opt/ml/processing/output \
          preprocessing /opt/ml/preprocess.py \
          --input-data-dir /opt/ml/processing/input \
          --output-data-dir /opt/ml/processing/output
```

## Perform Hyperparameter optimization inside Docker container

```bash
docker build -t training -f docker/training.dockerfile .

docker run -v $(pwd)/data:/opt/ml/input/data/training \
          -v $(pwd)/model:/opt/ml/model \
          -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
          -e SM_MODEL_DIR=/opt/ml/model \
          training
```

## Test inference inside Docker container

```bash
docker build -t inference -f docker/inference.dockerfile .

docker run -v $(pwd)/model:/opt/ml/model \
          -v $(pwd)/test:/opt/ml/code/test \
          -e SM_MODEL_DIR=/opt/ml/model \
          inference python /opt/ml/code/test/test_predict.py
```

### **Using the AWS CLI for Docker Image Management**

This subsection explains how to set up the AWS CLI on Arch Linux, push the Docker image to Amazon ECR Public, and retrieve the image locally.

#### **Step 1: Install the AWS CLI**

On Arch Linux, you can install the AWS CLI using the `aws-cli` package:

```bash
yay -S aws-cli
```

#### **Step 2: Configure the AWS CLI**

Set up the AWS CLI with your credentials and default region:

```bash
aws configure
```

You will be prompted to enter:

- **AWS Access Key ID**
- **AWS Secret Access Key**

Ensure that your AWS credentials are valid and the required permissions are assigned to your user account for ECR Public operations.

#### **Step 3: Upload the Docker Image**

Use the provided `push_to_ecr.sh` script to push the Docker images to Amazon ECR Public:

```bash
cd scripts
./push_to_ecr.sh
```
