import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class TitanicProductionEngine:
    """
    PORTFOLIO GRADE ML ENGINE
    Architecture: Modular Pipeline with Iterative Imputation and Hyperparameter Tuning.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.feature_names = None

    def _load_and_clean(self):
        df = pd.read_csv(self.data_path)
        
        # FEATURE ENGINEERING: Logic-driven extraction
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # INTERACTION FEATURE: Validated Family Logic
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        return df

    def run_full_pipeline(self):
        # 1. Prepare Data
        df = self._load_and_clean()
        X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize', 'Embarked']]
        y = df['Survived']
        
        # 2. Stratified Split (Portfolio standard for imbalanced/small data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 3. Define Preprocessing Logic
        # Numeric: Iterative Imputer handles 'Age' based on 'Pclass' & 'Fare'
        num_transformer = Pipeline([
            ('imputer', IterativeImputer(random_state=42)),
            ('scaler', RobustScaler())
        ])

        # Categorical: Most Frequent handles 'Embarked'
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_transformer, ['Age', 'Fare', 'FamilySize']),
            ('cat', cat_transformer, ['Sex', 'Title', 'Pclass', 'Embarked'])
        ])

        # 4. Define the Model Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # 5. HYPERPARAMETER TUNING (The Optimization Layer)
        param_grid = {
            'classifier__n_estimators': [100, 300],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_leaf': [1, 2, 4]
        }

        # 5-Fold Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("🚀 Optimizing Model Parameters...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        
        # 6. EVALUATION
        self._evaluate_model(X_test, y_test)
        
    def _evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print("\n✅ PRODUCTION MODEL EVALUATION")
        print(classification_report(y_test, predictions))
        
        # Confusion Matrix Logic
        # 
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        
    def export_model(self, filename="titanic_v1.joblib"):
        """Save the entire pipeline for deployment."""
        joblib.dump(self.model, filename)
        print(f"📦 Model exported successfully to {filename}")

# EXECUTION
if __name__ == "__main__":
    engine = TitanicProductionEngine("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    engine.run_full_pipeline()
    engine.export_model()