import pandas  as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
train_data=pd.read_csv(url)
print(train_data.describe())
print(train_data.info())
print(train_data.head())
print(train_data.duplicated())
print(train_data.isna().sum())
encod=OneHotEncoder(sparse_output=False,drop='if_binary')
train_data['Sex']=encod.fit_transform(train_data['Sex'].values.reshape(-1, 1))
print(train_data['Sex'])
corr_data=train_data.select_dtypes(include=['int64','float64']).corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_data,annot=True,cmap='coolwarm')
plt.show()
