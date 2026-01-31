import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class TitanicPipeline:
    def __init__(self, url):
        self.url = url
        self.df = None
        self.encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
        self.imputer = IterativeImputer(max_iter=10, random_state=42)

    def load_data(self):
        """Loads the dataset from the URL."""
        self.df = pd.read_csv(self.url)
        print("Data Loaded Successfully.")
        return self.df

    def explore_data(self):
        """Prints basic statistics and missing values."""
        print("\n--- Data Info ---")
        print(self.df.info())
        print("\n--- Missing Values ---")
        print(self.df.isna().sum())

    def preprocess(self):
        """Handles Encoding and Imputation."""
        # 1. Encode Sex
        # We use fit_transform and update the dataframe state
        sex_reshaped = self.df['Sex'].values.reshape(-1, 1)
        self.df['Sex'] = self.encoder.fit_transform(sex_reshaped)
        
        # 2. Impute Age
        # IterativeImputer expects a 2D array
        self.df['Age'] = self.imputer.fit_transform(self.df[['Age']])
        
        print("\nPreprocessing Complete: Sex encoded and Age imputed.")

    def visualize_correlations(self):
        """Generates a heatmap of numerical features."""
        corr_data = self.df.select_dtypes(include=['int64', 'float64']).corr()
        plt.figure(figsize=(8, 7))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm')
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def check_distribution(self):
        """Checks skewness and descriptive stats for specific columns."""
        cols = ['Age', 'Fare']
        print("\n--- Skewness ---")
        print(self.df[cols].skew())
        print("\n--- Descriptive Stats ---")
        print(self.df[cols].describe())

# --- Execution ---
if __name__ == "__main__":
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    # Instantiate the object
    pipeline = TitanicPipeline(titanic_url)
    
    # Run the workflow
    pipeline.load_data()
    pipeline.explore_data()
    pipeline.preprocess()
    pipeline.visualize_correlations()
    pipeline.check_distribution()