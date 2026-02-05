import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Requirements: Traditional ML Algorithms
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Requirement: Advanced Proficiency
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class MLProductionPipeline:
    def __init__(self, data_path, target_column):
        self.df = pd.read_csv(data_path)
        self.target = target_column
        self.model_suite = {}
        self.best_model = None
        
    def clean_and_prepare(self):
        """Requirement: Data parsing, scraping, and wrangling"""
        # 1. Drop high-cardinality strings (Names, IDs)
        # You can customize this list per problem
        cols_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
        self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True)
    
        
        
        # 2. Handle Missing Values (Imputation)
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # 3. Encoding (Convert Text to Numbers)
        self.df = pd.get_dummies(self.df, drop_first=True)
        print("--- Data Wrangling Complete ---")

    def perform_feature_engineering(self):
        # 1. Title Extraction
        self.df['Title'] = self.df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        self.df['Title'] = self.df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        self.df['Title'] = self.df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        self.df['Title'] = self.df['Title'].replace('Mme', 'Mrs')

        # 2. Family Dynamics
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)

        # 3. Fare Engineering
        self.df['Fare'] = self.df['Fare'].replace(0, self.df['Fare'].median())
        self.df['Fare_Log'] = np.log1p(self.df['Fare'])
        self.df['Fare_Per_Person'] = self.df['Fare'] / self.df['FamilySize']

        # 4. Drop columns that are now "processed"
        self.df = self.df.drop(['Name', 'SibSp', 'Parch', 'Fare'], axis=1)
        return self.df    

    def split_df(self):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def run_tournament(self):
        """Requirement: Research, implement, and rank ML algorithms"""
        self.model_suite = {
            "k-NN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        
        results = []
        print("Features being used for training:", self.X_train.columns.tolist())
        print("First row of training data:\n", self.X_train.iloc[0])
        for name, model in self.model_suite.items():
            # Cross-validation proves the model isn't just lucky
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            results.append({"Model": name, "CV_Mean_Acc": cv_scores.mean()})
            
        self.ranking = pd.DataFrame(results).sort_values(by="CV_Mean_Acc", ascending=False)
        print("\n--- Model Ranking (Success Probability) ---")
        print(self.ranking)

    def tune_winner(self):
        """Requirement: Training models and tuning their hyperparameters"""
        
        best_model_name = self.ranking.iloc[0]['Model']

        if best_model_name == "Random Forest":
            params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
            base_model = RandomForestClassifier()
        elif best_model_name == "XGBoost":
            params = {'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
            base_model = XGBClassifier()
        else:
            # Fallback for simpler models
            self.best_model = self.model_suite[best_model_name].fit(self.X_train, self.y_train)
            return

        grid = GridSearchCV(base_model, params, cv=3, scoring='accuracy')
        grid.fit(self.X_train, self.y_train)
        self.best_model = grid.best_estimator_
        print(f"Best Params: {grid.best_params_}")
        print(f"\n--- Tuning the Winner: {best_model_name} ---")
        # After you have your best_model
       
    def evaluate_and_export(self):
        """Requirement: Analyzing errors and preparing for production"""
        preds = self.best_model.predict(self.X_test)
        print("\n--- Final Model Performance ---")
        print(classification_report(self.y_test, preds))
        importances = self.best_model.feature_importances_
        feat_names = self.X_train.columns
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("What actually mattered for Survival?")
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feat_names[i] for i in indices], rotation=45)
        plt.show()
                 
        # Save for Linux Production Server
        joblib.dump(self.best_model, 'final_ai_model.pkl')
        print("--- Model Exported: final_ai_model.pkl ---")

# --- HOW TO USE THIS FOR ANY PROJECT ---
if __name__ == "__main__":
    # Titanic Path (You can change this to any CSV file!)
    PATH = r"C:\Users\Ahmed\Desktop\titanic_data\train.csv"
    TARGET = "Survived"
    
    pipeline = MLProductionPipeline(PATH, TARGET)
    pipeline.perform_feature_engineering()

    pipeline.clean_and_prepare()
    pipeline.split_df()
    pipeline.run_tournament()
    pipeline.tune_winner()
    pipeline.evaluate_and_export()