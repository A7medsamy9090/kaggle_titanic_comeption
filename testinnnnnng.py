import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Traditional ML Requirements
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # The "Gold Standard" for Kaggle
from sklearn.metrics import accuracy_score

class TitanicPipeline:
    def __init__(self, file_path):
        # Requirement: Finding available datasets and loading them
        self.data = pd.read_csv(file_path)
        self.final_model = None
        print("--- Data Loaded Successfully ---")

    def eda(self):
        # Requirement: Exploring and visualizing data to gain understanding
        print("\n[EDA] Data Info:")
        print(self.data.info())
        print("\n[EDA] Missing Values:")
        print(self.data.isna().sum())

    def dropping_unnecessary_items(self):
        # Requirement: Data wrangling/cleaning
        cols_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
        self.data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        print(f"--- Dropped: {cols_to_drop} ---")

    def impute_and_encode_data(self):
        # Requirement: Data parsing and wrangling
        # Simple imputation for entry-level traditional ML
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
        self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())

        # Manual Encoding (Better for production than get_dummies sometimes)
        self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
        self.data = pd.get_dummies(self.data, columns=['Embarked'], drop_first=True)
        print("--- Imputation and Encoding Complete ---")

    def test_train_split(self):
        # Requirement: Building data pipelines
        X = self.data.drop(['Survived'], axis=1)
        y = self.data['Survived']
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"--- Data Split: Train size {len(self.x_train)}, Test size {len(self.x_test)} ---")

    def research_and_rank_models(self):
        # Requirement: Analyzing and ranking algorithms by success probability
        models = {
            "k-NN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(gamma='auto'),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        print("\n--- Researching Appropriate ML Algorithms ---")
        ranking = []
        for name, model in models.items():
            model.fit(self.x_train, self.y_train)
            score = model.score(self.x_test, self.y_test)
            ranking.append({"Model": name, "Accuracy": score})
        
        ranking_df = pd.DataFrame(ranking).sort_values(by="Accuracy", ascending=False)
        print(ranking_df)
        return ranking_df

    def tune_and_train_final_model(self):
        # Requirement: Training models and tuning hyperparameters
        print("\n--- Tuning Decision Tree (Top Candidate) ---")
        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10]
        }
        
        grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        grid.fit(self.x_train, self.y_train)
        
        self.final_model = grid.best_estimator_
        print(f"Best Params Found: {grid.best_params_}")
        
        # Requirement: Analyzing errors of the model
        y_pred = self.final_model.predict(self.x_test)
        print("\n[Final Evaluation] Classification Report:")
        print(classification_report(self.y_test, y_pred))

    def deploy_preparation(self):
        # Requirement: build and prepare AI models to be deployed in production
        joblib.dump(self.final_model, 'titanic_model_v1.pkl')
        print("\n--- Model Exported as 'titanic_model_v1.pkl' for Production ---")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Path to your file
    CSV_PATH = r"C:\Users\Ahmed\Desktop\titanic_data\train.csv"

    # Step-by-step Execution (The Pipeline)
    pipeline = TitanicPipeline(CSV_PATH)
    pipeline.eda()
    pipeline.dropping_unnecessary_items()
    pipeline.impute_and_encode_data()
    pipeline.test_train_split()
    
    # Requirement: Rank models
    pipeline.research_and_rank_models()
    
    # Requirement: Tune and finalize
    pipeline.tune_and_train_final_model()
    
    # Requirement: Prep for production
    pipeline.deploy_preparation()