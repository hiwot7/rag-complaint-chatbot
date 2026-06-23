import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set clean styling for visualizations
sns.set_theme(style="whitegrid")

def run_exploratory_analysis(df):
    """
    Performs initial EDA calculations on the raw complaint data.
    """
    print("\n📊 --- STEP 1: INITIAL EXPLORATORY DATA ANALYSIS ---")
    
    # 1. Total records profile
    total_records = len(df)
    print(f"Total raw complaints loaded: {total_records:,}")
    
    # 2. Check for missing narratives
    # Assuming 'Consumer complaint narrative' is the typical CFPB column name
    narrative_col = 'Consumer complaint narrative'
    if narrative_col not in df.columns:
        # Fallback to lowercases or alternate names if adjusted
        narrative_col = [col for col in df.columns if 'narrative' in col.lower()][0]
        
    has_narrative = df[narrative_col].notna().sum()
    missing_narrative = df[narrative_col].isna().sum()
    print(f"Complaints WITH consumer narratives: {has_narrative:,} ({has_narrative/total_records*100:.2f}%)")
    print(f"Complaints WITHOUT consumer narratives: {missing_narrative:,} ({missing_narrative/total_records*100:.2f}%)")
    
    # 3. Product distributions
    print("\nDistribution of complaints across top products:")
    product_counts = df['Product'].value_counts()
    for prod, count in product_counts.head(10).items():
        print(f" - {prod}: {count:,} ({count/total_records*100:.2f}%)")
        
    return narrative_col

def clean_text(text):
    """
    Applies standard token cleaning, lowecasing, boilerplate removal, and stripping.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Remove standard CFPB mask templates (e.g., XXXX, XX/XX/XXXX)
    text = re.sub(r'x{2,}', '', text)
    
    # 3. Strip common introductory boilerplate language
    boilerplate_patterns = [
        r"i am writing to file a complaint regarding",
        r"to whom it may concern",
        r"i am writing to complain about"
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)
        
    # 4. Remove special characters and punctuation, preserving single spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 5. Collapse multiple white spaces down to one single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_pipeline(input_path, output_path):
    """
    Runs the complete extraction, filtering, text-cleaning, and export workflow.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: Raw file not found at {input_path}")
        print("Please ensure your CFPB dataset file is inside 'data/raw/' directory.")
        return
        
    print("🚀 Ingesting raw CFPB dataset...")
    # Read CSV (using low_memory=False to manage mixed types safely)
    df = pd.read_csv(input_path, low_memory=False)
    
    # Run the exploratory summaries
    narrative_col = run_exploratory_analysis(df)
    
    print("\n🔍 --- STEP 2: DATA FILTERING & TARGET SELECTION ---")
    # Define Target Categories required by CrediTrust
    target_products = [
        "Credit card or prepaid card", 
        "Credit card",
        "Student loan",
        "Consumer Loan",
        "Vehicle loan or lease",
        "Payday loan, title loan, or personal loan",
        "Checking or savings account",
        "Savings account",
        "Money transfer, virtual currency, or money service"
    ]
    
    # Standardize map lookup to match CrediTrust's 4 primary pillars
    product_mapping = {
        'Credit card or prepaid card': 'Credit Card',
        'Credit card': 'Credit Card',
        'Checking or savings account': 'Savings Account',
        'Savings account': 'Savings Account',
        'Money transfer, virtual currency, or money service': 'Money Transfer',
        'Payday loan, title loan, or personal loan': 'Personal Loan',
        'Consumer Loan': 'Personal Loan'
    }
    
    # Filter by products
    df_filtered = df[df['Product'].isin(target_products)].copy()
    print(f"Filtered for financial domain products: {len(df_filtered):,} rows remaining.")
    
    # Map target columns to normalized names
    df_filtered['product_category'] = df_filtered['Product'].map(product_mapping)
    # Drop rows that fell outside the clean 4 core types if any mismatched map values exist
    df_filtered = df_filtered.dropna(subset=['product_category'])
    
    # Remove empty narrative fields
    df_filtered = df_filtered.dropna(subset=[narrative_col])
    # Clear out empty strings hidden as white spaces
    df_filtered = df_filtered[df_filtered[narrative_col].astype(str).str.strip() != ""]
    print(f"Removed records with empty narratives: {len(df_filtered):,} rows remaining.")
    
    print("\n🧼 --- STEP 3: EXECUTING TEXT CLEANING ENGINE (Lowercasing/Boilerplate Removal) ---")
    # Enable a tracking progress bar
    tqdm.pandas(desc="Cleaning Narratives")
    df_filtered['cleaned_narrative'] = df_filtered[narrative_col].progress_apply(clean_text)
    
    # Filter rows that became completely blank after scrubbing boilerplate text
    df_filtered = df_filtered[df_filtered['cleaned_narrative'].str.strip() != ""]
    
    # Compute sentence metrics for documentation and EDA summary
    df_filtered['narrative_word_count'] = df_filtered['cleaned_narrative'].apply(lambda x: len(x.split()))
    
    print("\n📉 Summary Statistics for Cleaned Narrative Lengths (Word Count):")
    print(df_filtered['narrative_word_count'].describe())
    
    # Save the output to processed folder
    print(f"\n💾 Saving polished output dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    print("✅ Preprocessing and EDA stage complete!")

if __name__ == "__main__":
    # Point these filenames exactly to what you have saved in your folders!
    RAW_DATA_PATH = "data/raw/complaints.csv" 
    PROCESSED_DATA_PATH = "data/processed/filtered_complaints.csv"
    
    preprocess_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)