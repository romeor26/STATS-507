# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.inspection import permutation_importance

# Load the data and inspect its structure
data = pd.read_csv('/Users/yining/Downloads/insurance.csv')
print("First three rows of the dataset:")
print(data.iloc[:3, :])
print("\nDataset info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe().T)

# Count the frequency of each unique value for all columns
print("\nValue counts for each column:")
for i in data.columns:
    print(f"\nValue counts for column: {i}")
    x = data[i].value_counts()
    print(x)

# Build a correlation matrix for numerical columns
num = data.select_dtypes(['int64', 'float64'])  # Select only numerical columns
correlation_matrix = num.corr()


# Heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Categorical data preview
cat = data[['sex', 'children', 'smoker', 'region']]
print("\nCategorical data preview:")
print(cat.head())

# Pie charts for categorical variables
n_cols = len(cat.columns)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(cat.columns):
    x = cat[col].value_counts()
    axes[i].pie(x, labels=x.index, autopct='%1.1f%%', wedgeprops={"linewidth": 2, "edgecolor": "white"})
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Smokers vs. non-smokers data analysis
smoker = data[data['smoker'] == "yes"]
non_smoker = data[data['smoker'] == "no"]
print(f"\nThere are {smoker.shape[0]} smokers and {non_smoker.shape[0]} non-smokers.")

# Age distributions for smokers and non-smokers
plt.figure(figsize=(10, 6))
sns.histplot(smoker['age'], kde=True, label='Smokers', color='blue', alpha=0.7)
sns.histplot(non_smoker['age'], kde=True, label='Non-Smokers', color='green', alpha=0.7)
plt.title("Age Distribution for Smokers vs Non-Smokers")
plt.legend()
plt.show()

# Expense distributions for smokers and non-smokers
plt.figure(figsize=(10, 6))
sns.histplot(smoker['expenses'], kde=True, label='Smokers', color='blue', alpha=0.7)
sns.histplot(non_smoker['expenses'], kde=True, label='Non-Smokers', color='green', alpha=0.7)
plt.title("Expense Distribution for Smokers vs Non-Smokers")
plt.legend()
plt.show()

# Scatter plots for smokers vs non-smokers
sns.scatterplot(data=data, x='age', y='expenses', hue='smoker', alpha=0.7)
plt.title("Age vs Expenses by Smoking Status")
plt.show()

sns.scatterplot(data=data, x='children', y='expenses', hue='smoker', alpha=0.7)
plt.title("Children vs Expenses by Smoking Status")
plt.show()

# Crosstab and bar chart for children and smoker
child_smoker_crosstab = pd.crosstab(data['children'], data['smoker'])
print("\nCrosstab of children vs smoker:")
print(child_smoker_crosstab)
child_smoker_crosstab.plot(kind='bar', figsize=(10, 6), colormap="viridis")
plt.title("Children vs Smoking Status")
plt.show()

# Bar chart for children vs expenses by smoking status
sns.barplot(data=data, x='children', y='expenses', hue='smoker', errorbar=None)
plt.title("Children vs Expenses by Smoking Status")
plt.show()

# Histograms for numerical columns between smokers and non-smokers
fig, axes = plt.subplots(len(num.columns), 2, figsize=(12, 20))
axes = axes.flatten()
for i, col in enumerate(num.columns):
    sns.histplot(smoker[col], kde=True, ax=axes[2 * i], color='blue', alpha=0.7)
    axes[2 * i].set_title(f"Smokers: {col}")

    sns.histplot(non_smoker[col], kde=True, ax=axes[2 * i + 1], color='green', alpha=0.7)
    axes[2 * i + 1].set_title(f"Non-Smokers: {col}")
plt.tight_layout()
plt.show()

# Encoding categorical variables
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data['region'] = data['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

# Feature selection and train-test split
X = data.drop(['expenses'], axis=1)
y = data['expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression and Random Forest Regressor
linear_model = LinearRegression()
random_forest = RandomForestRegressor(random_state=42)

# Train and evaluate Linear Regression
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print(f"Linear Regression - Mean Squared Error: {mean_squared_error(y_test, y_pred_linear):.2f}, "
      f"R^2: {r2_score(y_test, y_pred_linear):.2f}")

# Train and evaluate Random Forest
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
print(f"Random Forest - Mean Squared Error: {mean_squared_error(y_test, y_pred_rf):.2f}, "
      f"R^2: {r2_score(y_test, y_pred_rf):.2f}")

# Compute the updated correlation matrix
correlation_matrix_updated = data.corr()

# Display updated correlation matrix
print("\nCorrelation matrix (including 'smoker'):")
print(correlation_matrix_updated)

# Heatmap for the updated correlation matrix
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_updated, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap (Including Smoker)")
plt.show()

# Regression analysis with statsmodels
X_with_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_const).fit()
print("\nOLS Regression Summary:")
print(ols_model.summary())

# P-Values and significant predictors
p_values_df = pd.DataFrame({'Predictor': X_with_const.columns, 'P-Value': ols_model.pvalues})
significant_predictors = p_values_df[p_values_df['P-Value'] < 0.05]
print("\nP-values for Predictors:")
print(p_values_df)
print("\nSignificant Predictors (P < 0.05):")
print(significant_predictors)

# Visualize p-values
plt.figure(figsize=(10, 6))
sns.barplot(data=p_values_df, x='Predictor', y='P-Value')
plt.axhline(y=0.05, color='r', linestyle='--', label="Significance Level (P=0.05)")
plt.title("P-Values for Predictors")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Perform permutation importance testing on the Random Forest model
perm_importance = permutation_importance(random_forest, X_test, y_test, n_repeats=30, random_state=42)

# Create a DataFrame to store permutation importance results
perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
}).sort_values(by='Importance Mean', ascending=False)

# Display the permutation importance
print("\nPermutation Importance Results (Random Forest):")
print(perm_importance_df)

# Visualize the permutation importance
plt.figure(figsize=(10, 6))
sns.barplot(data=perm_importance_df, x='Importance Mean', y='Feature', errorbar=None)
plt.xlabel("Mean Importance (Decrease in R²)")
plt.ylabel("Feature")
plt.title("Permutation Importance of Predictors in Random Forest")
plt.show()

# Identify significant predictors from OLS
ols_significant_predictors = p_values_df[p_values_df['P-Value'] < 0.05]['Predictor']

# Identify top predictors from Random Forest (e.g., top 3 by importance mean)
top_rf_predictors = perm_importance_df['Feature'].head(3)

# Display overlaps
common_predictors = set(ols_significant_predictors).intersection(set(top_rf_predictors))
print("\nCommon Predictors Between OLS and Random Forest:")
print(common_predictors)

# Summary of results
print(f"\nSignificant Predictors from OLS (p < 0.05): {list(ols_significant_predictors)}")
print(f"Top Predictors from Random Forest: {list(top_rf_predictors)}")
