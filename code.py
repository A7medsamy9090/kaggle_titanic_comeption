import pandas  as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data=pd.read_csv(url)
train_data=data.drop(['PassengerId','Ticket','Cabin','Embarked'],axis=1)
y=train_data['Survived']
x=train_data.drop('Survived', axis=1)
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_data['Sex']=train_data['Sex'].map({'male':1,'female':0})
#performing EDA
print(train_data['Title'].value_counts())
print(train_data.shape)
print(train_data.describe())
print(train_data.isna().sum())
missing_pct = (train_data['Age'].isnull().sum() / len(train_data['Age'])) * 100

print('the misisng percantge of age coulmn is ',missing_pct)
# Data preprocessing
plt.figure(figsize=(10, 8))

sns.heatmap(train_data.select_dtypes(['int64', 'float64']).corr(),annot=True,cmap='coolwarm')
plt.title('correltaion map')
plt.show()
plt.figure(figsize=(10, 8))
sns.boxplot(x=train_data['Fare'])
plt.title('correltaion map')
plt.show()
# encoding titles
train_data['Title']=train_data['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
train_data['Title'] = train_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data = pd.get_dummies(train_data, columns=['Title'], drop_first=True)

# 3. View the new columns
print(train_data.columns)
# 4. imputing the missing age 
it_imputer = IterativeImputer(max_iter=10, random_state=42)
# It "guesses" Age based on Pclass, Sex, and Fare
train_data['Age'] = it_imputer.fit_transform(train_data[['Age', 'Pclass', 'Fare']])
train_data['Age'] = it_imputer.fit_transform(train_data[['Age']])
