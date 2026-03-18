import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

print("starting")

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
print('X shape:', X.shape)
print('y shape:', y.shape)

#file_path = ""
#data = pd.read_csv(file_path)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                      #Maybe change
    scoring="neg_root_mean_squared_error",  
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"Best model: {grid_search.best_params_}")

predictions = best_model.predict(X_test)

rmse = mean_squared_error(y_test, predictions)
print("Test RMSE:", rmse)
print('y_test:', y_test)
print('predictions:', predictions)

feature_importance = best_model.feature_importances_
plt.barh(iris.feature_names, feature_importance)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features with XGBoost')


for i in range(3):
  plt.figure(figsize = (10, 7))
  xgb.plot_tree(best_model, tree_idx=i, ax=plt.gca())

plt.savefig("tree.png")