import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for WSL
import matplotlib.pyplot as plt
import os

# Paths
TRAIN_PATH = '/mnt/d/watchtower_data/cicids2017/processed/train.csv'
OUTPUT_PATH = '/mnt/d/watchtower_data/cicids2017/analysis/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data():
    print('Loading training data...')
    df = pd.read_csv(TRAIN_PATH)
    print(f'Loaded {df.shape[0]} rows, {df.shape[1]} columns')
    return df

def get_feature_columns(df):
    """Returns feature columns only — excludes the Label column."""
    return [col for col in df.columns if col != 'Label']

def analyze_feature_importance(df):
    """
    Uses Random Forest to rank features by importance across all attack
    categories. This tells us which features are most discriminative overall.
    """
    print('\nAnalyzing feature importance...')
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['Label']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train a shallow Random Forest for importance scoring
    print('Training Random Forest for feature importance...')
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y_encoded)

    # Get feature importances
    importances = pd.Series(
        rf.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print('\nTop 20 most important features (overall):')
    print(importances.head(20))

    # Save to CSV
    importances.to_csv(os.path.join(OUTPUT_PATH, 'feature_importance.csv'),
                       header=['importance'])
    print(f'\nSaved to {OUTPUT_PATH}feature_importance.csv')

    return importances

def analyze_per_attack(df):
    """
    For each attack category, computes the mean value of each feature
    for attack flows vs benign flows. Large differences indicate
    discriminative features for that attack.
    """
    print('\nAnalyzing features per attack category...')
    feature_cols = get_feature_columns(df)
    attack_labels = [l for l in df['Label'].unique() if l != 'BENIGN']
    benign = df[df['Label'] == 'BENIGN'][feature_cols]

    results = {}
    for label in attack_labels:
        attack = df[df['Label'] == label][feature_cols]

        # Compute ratio of attack mean to benign mean for each feature
        # Large ratios = feature behaves very differently in attacks
        benign_mean = benign.mean()
        attack_mean = attack.mean()

        # Avoid division by zero
        ratio = attack_mean / (benign_mean + 1e-10)

        top_features = ratio.abs().sort_values(ascending=False).head(10)
        results[label] = top_features

        print(f'\nTop 10 discriminative features for {label}:')
        print(top_features)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_PATH, 'per_attack_features.csv'))
    print(f'\nSaved to {OUTPUT_PATH}per_attack_features.csv')

    return results

def plot_feature_importance(importances):
    """Saves a bar chart of top 20 features."""
    print('\nGenerating feature importance plot...')
    plt.figure(figsize=(12, 8))
    importances.head(20).plot(kind='bar')
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance.png'))
    plt.close()
    print(f'Saved plot to {OUTPUT_PATH}feature_importance.png')

if __name__ == '__main__':
    df = load_data()
    importances = analyze_feature_importance(df)
    analyze_per_attack(df)
    plot_feature_importance(importances)
    print('\nFeature analysis complete.')
