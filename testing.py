import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class TitanicPipeline:
    def __init__(self, df):
        self.df = pd.read_csv(r"C:\Users\Ahmed\Desktop\titanic_data\train.csv")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def dropping_unnecessary_items(self):
        # Ticket and PassengerId usually don't have predictive power
        self.df.drop(['Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True, errors='ignore')
        print("Dropped unnecessary columns.")

    def eda(self):
        print("Data Summary:")
        print(self.df.info())
        print(self.df.describe())

    def feature_engineering(self):
        # 1. Extract Title from Name
        self.df['Title'] = self.df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        self.df['Title'] = self.df['Title'].replace(['Lowry', 'Sir', 'Countess', 'Capt', 'Col', 'Don', 
                                                     'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
        # 2. Create FamilySize
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df.drop(['Name'], axis=1, inplace=True)
        print("Feature engineering complete.")

    def impute_and_encode_data(self):
        # Fill Age with median, Embarked with mode
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median())
        
        # One-hot encoding for categorical variables
        self.df = pd.get_dummies(self.df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
        print("Data imputed and encoded.")

    def correlation_map(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Feature Correlation Map")
        plt.show()

    def plot_age_dist(self):
        sns.histplot(self.df['Age'], kde=True)
        plt.title("Age Distribution")
        plt.show()

    def train_and_evaluate(self):
        # Split features and target
        X = self.df.drop('Survived', axis=1)
        y = self.df['Survived']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training
        self.model.fit(X_train, y_train)
        
        # Evaluation
        predictions = self.model.predict(X_test)
        print("\n--- Model Performance ---")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        # Visualizing results
        sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# --- Execution ---
df=pd.read_csv(r"C:\Users\Ahmed\Desktop\titanic_data\train.csv")
pipeline = TitanicPipeline(df)
pipeline.dropping_unnecessary_items()
pipeline.feature_engineering()
pipeline.impute_and_encode_data()
pipeline.train_and_evaluate()