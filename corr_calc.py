import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression


file_path = "/mnt/c/Users/Lovisa/Downloads/nfi_germany_treedata/bwi_tree.csv"
data = pd.read_csv(file_path, sep = ";", decimal=",")
df = data.sample(n=600, random_state=42)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
print(num_cols)
print(cat_cols)

corr_num = df[num_cols].corr(method='pearson')
print(corr_num)
#corr_num.to_csv("/mnt/c/Users/Lovisa/Downloads/corr_num.csv")






def cramers_v(x, y):
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    r, k = contingency.shape
    return np.sqrt((chi2/n) / (min(k-1, r-1)))

corr_matrix = pd.DataFrame(
    np.zeros((len(cat_cols), len(cat_cols))),
    index=cat_cols,
    columns=cat_cols
)

# fill matrix
for i, col in enumerate(cat_cols):
    for j, col1 in enumerate(cat_cols):
        corr_matrix.loc[col, col1] = cramers_v(df[col], df[col1])

corr_matrix.to_csv("/mnt/c/Users/Lovisa/Downloads/corr_cat.csv")

cols = df.columns
corr_matrix = pd.DataFrame(index=cols, columns=cols)

for i, col1 in enumerate(cols):
    for col2 in cols[i:]:
        if col1 == col2:
            corr_matrix.loc[col1, col2] = 1
        
        elif col1 in num_cols and col2 in num_cols:
            corr_matrix.loc[col1, col2] = df[col1].corr(df[col2])
        
        elif col1 in cat_cols and col2 in cat_cols:
            corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
        
        else:
            # mixed case → mutual information
            
            
            x = df[[col1]] if col1 in num_cols else pd.get_dummies(df[[col1]])
            y = df[col2] if col2 in num_cols else pd.factorize(df[col2])[0]
            
            corr_matrix.loc[col1, col2] = mutual_info_regression(x, y)[0]

corr_matrix.to_csv("/mnt/c/Users/Lovisa/Downloads/corr_matrix.csv")