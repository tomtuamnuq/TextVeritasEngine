import argparse
import os
import pandas as pd

from utils.namings import PROCESSED_DATA_FILENAME
from utils.text_processing import preprocess_dataset, download_nltk_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-dir", type=str, required=True)
    parser.add_argument("--output-data-dir", type=str, required=True)
    return parser.parse_args()


def combine_news_datasets(input_dir):
    fake_df = pd.read_csv(os.path.join(input_dir, "Fake.csv"))
    true_df = pd.read_csv(os.path.join(input_dir, "True.csv"))
    final_en_df = pd.read_csv(os.path.join(input_dir, "final_en.csv"))
    welfake_df = pd.read_csv(os.path.join(input_dir, "WELFake_Dataset.csv"))

    fake_df["fake"] = 1
    true_df["fake"] = 0
    final_en_df["fake"] = 1 - final_en_df["lebel"]
    welfake_df["fake"] = welfake_df["label"]

    dfs = []
    for df in [fake_df, true_df, final_en_df, welfake_df]:
        df = df[["title", "text", "fake"]].copy()
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["title"] = combined_df["title"].fillna("").str.strip()
    combined_df["text"] = combined_df["text"].fillna("").str.strip()
    combined_df = combined_df[
        (combined_df["title"] != "")
        & (combined_df["text"] != "")
        & (~combined_df["title"].isna())
        & (~combined_df["text"].isna())
    ]
    combined_df = combined_df.drop_duplicates(subset=["title"])
    combined_df["fake"] = combined_df["fake"].astype(int)
    return combined_df


def remove_length_outliers(df, threshold_percentile=95):
    length_threshold = df["text_length"].quantile(threshold_percentile / 100)
    return df[df["text_length"] <= length_threshold].copy()


if __name__ == "__main__":
    args = parse_args()
    download_nltk_data()

    # Process the data
    df = combine_news_datasets(args.input_data_dir)
    df = preprocess_dataset(df)
    df = remove_length_outliers(df)

    # Save results
    output_path = os.path.join(args.output_data_dir, PROCESSED_DATA_FILENAME)
    df[["title_clean", "text_clean", "title_length", "text_length", "fake"]].to_csv(
        output_path, index=False
    )
