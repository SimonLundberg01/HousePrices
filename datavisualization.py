import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
from model import conversion_list, features, conditions
from model import calculate_skewness,none_transform
from model import transform_based_on_skewness, remove_outliers

# Load data
def read_data(file_path):
    return pd.read_csv(file_path)

# Plot distributions for continuous and ordinal features
def plot_numeric_distributions(df, features, pdf_pages):
    numeric_features = features["Continuous"] + features["Ordinal"]
    for feature in numeric_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature], kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        pdf_pages.savefig()
        plt.close()

# Plot count plots for categorical features
def plot_categorical_distributions(df, features, pdf_pages):
    categorical_features = features["Categorical"]
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, hue=feature, order=df[feature].value_counts().index, palette='viridis', legend=False)
        plt.title(f'Count Plot of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf_pages.savefig()
        plt.close()

# Main function to execute all plotting
def main():
    file_path = 'train.csv'
    df = read_data(file_path)
    df.drop('Id', axis=1, inplace=True)
    with PdfPages('distributions_pre.pdf') as pdf:
        plot_numeric_distributions(df, features, pdf)
        plot_categorical_distributions(df, features, pdf)
   
    df = none_transform(df)
    df = remove_outliers(df)
    df = transform_based_on_skewness(df)
    with PdfPages('distributions_post.pdf') as pdf:
        plot_numeric_distributions(df, features, pdf)
        plot_categorical_distributions(df, features, pdf)
    
    print("All feature plots have been saved to 'feature_distributions.pdf'.")

if __name__ == '__main__':
    main()
