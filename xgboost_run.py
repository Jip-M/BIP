import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import shap
import numpy as np


file_path = "/mnt/c/Users/Lovisa/Downloads/nfi_germany_treedata/bwi_tree.csv"
data = pd.read_csv(file_path, sep = ";", decimal=",")
data["plot_hydrol_class"] = data["plot_hydrol_class"].astype(str)
data = data.sample(n=10000, random_state=42)
#print(data.columns.values)
y = data["dbh"]
X = data.drop(columns=["dbh"])
print(len(X))
X = X.dropna(axis=1, how="all")
print(len(X))
mask = y.notna()
X = X[mask]
y = y[mask]

#Handle categorical data:
#X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


cat_cols = data.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")
    X_test[col] = X_test[col].cat.set_categories(X_train[col].cat.categories)

model = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    #"subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,                      #Maybe change
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
print('y_test:', y_test[0:10])
print('predictions:', predictions[0:10])


plt.figure()
xgb.plot_importance(best_model, max_num_features=25)
plt.savefig("ft_importance.png")


#Shap
plt.figure()
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

shap.plots.bar(shap_values)

plt.savefig("shap_bar_plot.png", dpi=300, bbox_inches="tight")
plt.close()

