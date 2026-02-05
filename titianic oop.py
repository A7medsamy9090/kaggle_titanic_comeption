import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Requirement: Advanced Proficiency
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

class titanic:
    def __init__(self):
        self.data=pd.read_csv(r"C:\Users\Ahmed\Desktop\titanic_data\train.csv")
        self.encoder=OneHotEncoder(sparse_output=False)
        self.imputer=IterativeImputer(max_iter=10,random_state=42)
        

    def eda(self):
      print(self.data.info())
      print(self.data.describe())
      print(self.data.head())
      print(self.data.isna().sum())

    def dropping_unnesscirly_items (self):

     self.data=self.data.drop(['PassengerId','Ticket','Cabin','Embarked','Name'],axis=1)  
     return self.data
    
    def impute_and_encode_data(self):
     self.data['Age']=self.imputer.fit_transform(self.data[['Age']])
     
     self.data['Sex']=self.encoder.fit_transform(self.data[['Sex']])

     print(self.data.isna().sum()) 

     return self.data

    def correaltion_map(self):
    
     plt.figure(figsize=(8,7))

     sns.heatmap(self.data.select_dtypes(['int64','float64']).corr(),cmap='coolwarm',annot=True)   
     plt.title("the correltion between numericalfeatures")
     plt.show() 
    def plot_age_dist(self):
     sns.histplot(data=self.data, x='Age', hue='Survived', kde=True, element="step")
     plt.title('Age Distribution by Survival Status')
     plt.show()

    def test_train_split(self):
     
     x=self.data.drop(['Survived'],axis=1)
     y=self.data['Survived']
     
     self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,test_size=0.2,random_state=42)
     return self.x_test,self.x_train,self.y_test,self.y_train 
    
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
        
        result=[]
        for name,modle in self.model_suite.items():
         cv=cross_val_score(modle,self.x_train,self.y_train,cv=3)
         self.result_modles=result.append({'model':name,'cv_scores_result':cv.mean()}) 
         self.results_ranking=pd.DataFrame(result).sort_values(by='cv_scores_result',ascending=False)
        print("\n--- Model Ranking (Success Probability) ---")
        print(self.results_ranking)

    def tunning_winner(self):
      best_model = self.results_ranking.iloc[0]
      if best_model == 'Random Forest':

      
        
 
if __name__ == "__main__":

   pipeline=titanic()
   pipeline.dropping_unnesscirly_items()
   pipeline.eda()        
   pipeline.impute_and_encode_data()
   pipeline.correaltion_map()
   pipeline.plot_age_dist() 
   pipeline.test_train_split()
   pipeline.run_tournament()
   
