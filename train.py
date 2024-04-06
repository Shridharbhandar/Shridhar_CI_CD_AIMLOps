import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

#model = LogisticRegression().fit(X, y)
#model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)  # Score - 0.428

gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=1, max_depth=100)
model = gbc.fit(X,y)

# model = RandomForestClassifier().fit(X, y)

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [10, 100, 250], 
#     'max_depth': [None, 25]
# }

# #100,250 & cv, md - 25 || Score - 0.442

# # Instantiate GridSearchCV, fit, and find best parameters
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=25)
# grid_search.fit(X, y)
# best_params = grid_search.best_params_ 

# # Train the model with best parameters
# model = RandomForestClassifier(**best_params)
# model.fit(X, y)



with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
