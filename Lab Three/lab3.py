# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load Titanic dataset from seaborn
data = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# Step 2: Data Cleaning
# 1. Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 2. Handle missing values
# Impute 'age' and 'fare' with the median, and 'embarked' with the mode
imputer_median = SimpleImputer(strategy='median')
data['age'] = imputer_median.fit_transform(data[['age']])
data['fare'] = imputer_median.fit_transform(data[['fare']])

from sklearn.impute import SimpleImputer

# Define the imputer to fill 'embarked' with the most frequent value (mode)
imputer_mode = SimpleImputer(strategy='most_frequent')

# Apply the imputer and assign the transformed column back to 'embarked'
data['embarked'] = imputer_mode.fit_transform(data[['embarked']]).ravel()


# Drop columns with excessive missing data (like 'deck' and 'embark_town')
data = data.drop(columns=['deck', 'embark_town'])

# Verify missing values have been handled
print("\nData after handling missing values:")
print(data.isnull().sum())

# Step 3: Handling Outliers
# Visualize outliers using boxplots for 'age' and 'fare'
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data['age']).set_title('Age Outliers')
plt.subplot(1, 2, 2)
sns.boxplot(data['fare']).set_title('Fare Outliers')
plt.show()

# Cap outliers at the 99th percentile for 'age' and 'fare'
age_cap = data['age'].quantile(0.99)
fare_cap = data['fare'].quantile(0.99)
data['age'] = np.where(data['age'] > age_cap, age_cap, data['age'])
data['fare'] = np.where(data['fare'] > fare_cap, fare_cap, data['fare'])

# Step 4: Data Normalization
scaler = MinMaxScaler()
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])

# Step 5: Feature Engineering
# 1. Create family_size feature
data['family_size'] = data['sibsp'] + data['parch']

# 2. Extract title from name (if name column exists)
if 'name' in data.columns:
    data['title'] = data['name'].str.extract('([A-Za-z]+)\\.', expand=False)
    print("\nTitles extracted from name:")
    print(data['title'].value_counts())

# Step 6: Feature Selection
# Drop columns that are not informative for prediction
# Specify the columns you want to drop
columns_to_drop = ['name', 'ticket', 'sibsp', 'parch']

# Drop columns only if they exist in the DataFrame
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])


# One-hot encode categorical features
data = pd.get_dummies(data, columns=['sex', 'embarked', 'class', 'who', 'title'], drop_first=True)

# Calculate correlation matrix and select features based on high correlation with 'survived'
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 7: Model Building
# Split the data into train and test sets
X = data.drop(columns=['survived'])
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
