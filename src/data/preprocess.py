import pandas as pd
import numpy as np
import os

# Paths
RAW_PATH = '/mnt/d/watchtower_data/cicids2017/raw/'
PROCESSED_PATH = '/mnt/d/watchtower_data/cicids2017/processed/'
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Files to load
FILES = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
]

# Attack categories to exclude (out of scope)
EXCLUDE_LABELS = ['Heartbleed']

def load_and_combine(files, raw_path):
    dfs = []
    for file in files:
        print(f'Loading {file}...')
        df = pd.read_csv(os.path.join(raw_path, file))
        # Fix leading/trailing spaces in column names
        df.columns = df.columns.str.strip()
        dfs.append(df)
    print('Combining all files...')
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    print(f'Initial shape: {df.shape}')

    # 1. Strip label values
    df['Label'] = df['Label'].str.strip()

    # 2. Remove excluded labels
    df = df[~df['Label'].isin(EXCLUDE_LABELS)]
    print(f'After removing excluded labels: {df.shape}')

    # 3. Remove duplicates
    df = df.drop_duplicates()
    print(f'After removing duplicates: {df.shape}')

    # 4. Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 5. Impute missing values with column median
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f'After handling infinites and nulls: {df.shape}')

    # 6. Print final label distribution
    print('\nFinal label distribution:')
    print(df['Label'].value_counts())

    return df

def split_and_save(df, processed_path):
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df['Label']
    )

    print(f'\nTrain shape: {train_df.shape}')
    print(f'Test shape: {test_df.shape}')
    print('\nTrain labels:')
    print(train_df['Label'].value_counts())
    print('\nTest labels:')
    print(test_df['Label'].value_counts())

    train_df.to_csv(os.path.join(processed_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(processed_path, 'test.csv'), index=False)
    print('\nSaved train.csv and test.csv to', processed_path)

if __name__ == '__main__':
    df = load_and_combine(FILES, RAW_PATH)
    df = preprocess(df)
    split_and_save(df, PROCESSED_PATH)
