import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import zscore
import numpy as np
from sklearn.model_selection import GridSearchCV

train_data = "C:/Users/User/Downloads/Bank_Personal_Loan_Modelling_train.csv"
test_data = "C:/Users/User/Downloads/Bank_Personal_Loan_Modelling_reserved.csv"
train = pd.read_csv(train_data)
X_test = pd.read_csv(test_data)

train = train.drop(columns=['ID', 'ZIP Code'])
train['Experience'] = train['Experience'] + 3
train['CCAvg'] = train['CCAvg'] * 12

X_test = X_test.drop(columns=['ID', 'ZIP Code'])
X_test['Experience'] = X_test['Experience'] + 2
X_test['CCAvg'] = X_test['CCAvg'] * 12

feature = 'Mortgage'
df = train
# Вычисление Z-score для выбранного признака
df['z_score'] = zscore(df[feature])
# Фильтрация строк: оставить только те, где |z_score| <= 3
filtered_df = df[np.abs(df['z_score']) <= 3].drop(columns=['z_score'])


X_train = filtered_df.drop('Personal Loan', axis=1)
y_train = filtered_df['Personal Loan']

clf = DecisionTreeClassifier(random_state=21, criterion='entropy', min_samples_leaf=1, min_samples_split=5, max_depth=20)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_v = ','.join(map(str, y_pred))
print(y_v)

dt = DecisionTreeClassifier(random_state=21)

# Определение параметров для поиска
param_grid = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Применение GridSearchCV для поиска лучших гиперпараметров
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# Лучшие параметры
print("Best parameters:", grid_search.best_params_)