import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main.config import STOPWORDS
import re
from typing import List, Dict
import argparse

def load_data(dataset_loc: str, num_samples: int = None) -> pd.DataFrame:
    """ Load data from a CSV file into a Pandas DataFrame.

    Args:
        dataset_loc (str): Location of the dataset
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        pd.DataFrame: Dataset represented by a Pandas DataFrame
    """

    df = pd.read_csv(dataset_loc)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if num_samples:
        df = df.head(num_samples)
    return df



def stratify_split(dataset: pd.DataFrame, test_size: float = 0.20) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """ Perform a stratified train-test split on a Pandas Dataframe



    Args:
        dataset (pd.DataFrame): Input DataFrame with 'text' and 'tag' columns
        test_size (float, optional): Proportion of dataset to allocate to test set. Defaults to 0.20.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series, pd.Series]: _description_
    """
    X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['tag'], stratify=dataset['tag'], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """ Clean raw text string

    Args:
        text (str): Raw text to clean
        stopwords (List, optional): List of stopwords. Defaults to STOPWORDS.

    Returns:
        str: Cleaned text
    """

        # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text


def preprocess(df: pd.DataFrame) -> Dict:
    """ Preprocess the dataset

    Args:
        df (pd.DataFrame): Raw Dataframe to preprocess

    Returns:
        Dict: Preprocessed data
    """
    df['text'] = df['title'] + " " + df['description']
    df['text'] = df.text.apply(clean_text)
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")
    df = df[["text", "tag"]]
    return df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data preprocessing steps.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to load.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset to use for testing.")
    

    args = parser.parse_args()

    print("\nLoading dataset...")
    df = load_data(args.dataset, args.num_samples)
    print(f"Dataset loaded with {len(df)} samples.")

    print("\nPreprocessing dataset...")
    df = preprocess(df)
    print("Preprocessing complete. Sample:")
    print(df.head())

    print("\nPerforming stratified train-test split...")
    X_train, X_test, y_train, y_test = stratify_split(df, args.test_size)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("\nSample processed training data:")
    print(X_train.head())