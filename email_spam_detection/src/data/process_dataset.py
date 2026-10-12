# Lifecycle stage 4 — Data Preparation (full dataset)
 
import pandas as pd
from src.data.preprocess import clean_batch
 
def process_dataset():
 
    print("Script Started")
 
    df = pd.read_csv("data/raw/spam.csv",encoding='latin-1')
 
    print("Dataset Loaded")
 
    # Lowercasing + regex are fast as vectorized pandas ops,
    # so we do them here and leave only tokenizing/lemmatizing to spaCy.
 
    texts = df["v2"].astype(str).str.lower()
    texts = texts.str.replace(r"http\S+", "", regex=True)
    texts = texts.str.replace(r"[^a-zA-Z ]", "", regex=True)
 
    df["clean_email"] = clean_batch(texts)
 
    print("Cleaning Completed")
 
    df.to_csv("data/processed/cleaned_emails.csv", index=False)
 
    print("File Saved")
