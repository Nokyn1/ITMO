import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV


# Тренировочные данные
train_data = "C:/Users/User/Downloads/Bank_Personal_Loan_Modelling_train.csv"
data = pd.read_csv(train_data)

data = data.drop(columns=['ID', 'ZIP Code'])
data['Experience'] = data['Experience'] + 3
data['CCAvg'] = data['CCAvg'] * 12

feature = 'Mortgage'
df = data
# Вычисление Z-score для выбранного признака
df['z_score'] = zscore(df[feature])
# Фильтрация строк: оставить только те, где |z_score| <= 3
filtered_df = df[np.abs(df['z_score']) <= 3].drop(columns=['z_score'])

X = filtered_df.drop('Personal Loan', axis=1)
y = filtered_df['Personal Loan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

print("Исходное количество классов в обучающей выборке:", Counter(y_train))
smote = SMOTE(random_state=21)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Новое количество классов после применения SMOTE:", Counter(y_train_res))

clf = DecisionTreeClassifier(random_state=21, criterion='gini', class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

class_0_percentage = (y_train_res == 0).mean()

print(f'Dолля элементов класса 0 в тренировочной выборке: {class_0_percentage:.4f}')

dt = DecisionTreeClassifier(random_state=21)

# Определение параметров для поиска
param_grid = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Применение GridSearchCV для поиска лучших гиперпараметров
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Лучшие параметры
print("Best parameters:", grid_search.best_params_)
