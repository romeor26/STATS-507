# Import tools needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data and inspect its basic structure
data = pd.read_csv('/Users/yining/Downloads/insurance.csv')

# Display the first 3 rows of the dataset
print("First three rows of the dataset:")
print(data.iloc[:3, :])

# Display basic information about the dataset
print("\nDataset info:")
print(data.info())

# Display summary statistics for numerical columns
print("\nSummary statistics:")
print(data.describe().T)

# Execute basic data analysis

# Count the frequency of each unique value for all columns
print("\nValue counts for each column:")
for i in data.columns:
    print(f"\nValue counts for column: {i}")
    x = data[i].value_counts()
    print(x)

# Build a correlation matrix for numerical columns
num = data.select_dtypes(['int64', 'float64'])  # Select only numerical columns
correlation_matrix = num.corr()
print("\nCorrelation matrix:")
print(correlation_matrix)

# Build a heatmap for the correlation matrix
print("\nHeatmap of correlation matrix:")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Select categorical columns for further analysis
cat = data[['sex', 'children', 'smoker', 'region']]
print("\nCategorical data preview:")
print(cat.head())

# Build pie charts for categorical variables
n_cols = len(cat.columns)  # Number of categorical columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # Adjust rows/columns as needed
axes = axes.flatten()  # Flatten axes for easier iteration

print("\nPie charts for categorical variables:")
for i, col in enumerate(cat.columns):
    x = cat[col].value_counts()
    axes[i].pie(x, labels=x.index, autopct='%1.1f%%', wedgeprops={"linewidth": 2, "edgecolor": "white"})
    axes[i].set_title(f"Distribution of {col}")

# Adjust layout to avoid overlapping
plt.tight_layout()
plt.show()

# Analyze the number of people who smoke and don't smoke
smoker = data[data['smoker'] == "yes"]  # Filter rows for smokers
non_smoker = data[data['smoker'] == "no"]  # Filter rows for non-smokers

print("\nSmokers and Non-Smokers data:")
print(f"There are {smoker.shape[0]} smokers and {non_smoker.shape[0]} non-smokers.")

# Analyze age distributions
print("\nAge distributions for smokers and non-smokers:")
plt.figure(figsize=(10, 6))
sns.histplot(smoker['age'], kde=True, label='Smokers', color='blue', alpha=0.7)
sns.histplot(non_smoker['age'], kde=True, label='Non-Smokers', color='green', alpha=0.7)
plt.title("Age Distribution for Smokers vs Non-Smokers")
plt.legend()
plt.show()

# Analyze expense distributions
print("\nExpense distributions for smokers and non-smokers:")
plt.figure(figsize=(10, 6))
sns.histplot(smoker['expenses'], kde=True, label='Smokers', color='blue', alpha=0.7)
sns.histplot(non_smoker['expenses'], kde=True, label='Non-Smokers', color='green', alpha=0.7)
plt.title("Expense Distribution for Smokers vs Non-Smokers")
plt.legend()
plt.show()

# Scatter plot of numerical columns with target 'smoker'
print("\nScatter plots comparing smokers and non-smokers:")
sns.scatterplot(data=data, x='age', y='expenses', hue='smoker', alpha=0.7)
plt.title("Age vs Expenses by Smoking Status")
plt.show()

sns.scatterplot(data=data, x='children', y='expenses', hue='smoker', alpha=0.7)
plt.title("Children vs Expenses by Smoking Status")
plt.show()

# Crosstab analysis for children and smoker, followed by a bar plot
child_smoker_crosstab = pd.crosstab(data['children'], data['smoker'])
print("\nCrosstab of children vs smoker:")
print(child_smoker_crosstab)
child_smoker_crosstab.plot(kind='bar', figsize=(10, 6), colormap="viridis")
plt.title("Children vs Smoking Status")
plt.show()

# Bar chart for children and expenses based on smoking status
print("\nBar chart: Children vs Expenses by Smoking Status:")
sns.barplot(data=data, x='children', y='expenses', hue='smoker', errorbar=None)
plt.title("Children vs Expenses by Smoking Status")
plt.show()

# Histogram comparisons for numerical columns between smokers and non-smokers
print("\nHistograms for numerical columns:")
fig, axes = plt.subplots(len(num.columns), 2, figsize=(12, 20))
axes = axes.flatten()

for i, col in enumerate(num.columns):
    sns.histplot(smoker[col], kde=True, ax=axes[2 * i], color='blue', alpha=0.7)
    axes[2 * i].set_title(f"Smokers: {col}")

    sns.histplot(non_smoker[col], kde=True, ax=axes[2 * i + 1], color='green', alpha=0.7)
    axes[2 * i + 1].set_title(f"Non-Smokers: {col}")

plt.tight_layout()
plt.show()

# Enoding the values in the sex column to 0 and 1
# Encoding the values in the sex column to 0 and 1
gender = {'male': 0, 'female': 1}
data['sex'] = data['sex'].apply(lambda sex: gender[sex])
data.head()


# Encoding the data in the smoker column (no: 0, yes: 1)
smoker = {'no': 0, 'yes': 1}
data['smoker'] = data['smoker'].map(smoker)

# Encoding the data in the region column (southwest: 0, southeast: 1, northwest: 2, northeast: 3)
regions = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
data['region'] = data['region'].map(regions)


# Feature selection and target variable
X = data.drop(['expenses'], axis=1)  # Features (all except 'expenses')
y = data['expenses']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize machine learning models
linear_model = LinearRegression()
random_forest = RandomForestRegressor(random_state=42)

# Train the Linear Regression model
print("\nTraining Linear Regression model...")
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluate the Linear Regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Linear Regression - Mean Squared Error: {mse_linear:.2f}, R^2 Score: {r2_linear:.2f}")

# Train the Random Forest Regressor
print("\nTraining Random Forest Regressor...")
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# Evaluate the Random Forest Regressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - Mean Squared Error: {mse_rf:.2f}, R^2 Score: {r2_rf:.2f}")

# Compare predictions with actual values using scatter plots
plt.figure(figsize=(12, 6))

# Scatter plot for Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")

# Scatter plot for Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")

plt.tight_layout()
plt.show()

# Feature Importance for Random Forest
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': random_forest.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("\nFeature Importances from Random Forest:")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature')
plt.title("Feature Importances from Random Forest")
plt.show()
