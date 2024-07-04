## Predicting Calories Burned Using Machine Learning

![image](https://github.com/FaizanDhankwala/CaloriesBurnedPrediction/assets/55712375/e964ee58-6351-4757-8f22-24766c827642)

## The Collab Link is included on the left as well as below
https://colab.research.google.com/drive/156C0pGDyzvfSOGZccsGkt-0dUOegi30S#scrollTo=wqUGLsK0ZURq

### Introduction
This machine learning program predicts calories burned during exercise using the XGBoost algorithm. It analyzes physiological and exercise-related features to estimate calorie burning.

### 1. Data Preparation and Exploration
#### Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```
## Loading and Combining Data
```
# Load calories and exercise datasets
calories = pd.read_csv("/content/calories.csv")
exercise_data = pd.read_csv("/content/exercise.csv")

# Combine datasets on User_ID
calories_data = pd.concat([exercise_data, calories["Calories"]], axis=1)
```
## Data Analysis
# Below is just basic analysis to give us insight of what data we are working with
```
# Visualize Gender distribution
sns.countplot(calories_data['Gender'])
plt.title('Distribution of Gender')

# Plot Age distribution
sns.histplot(calories_data['Age'], kde=True)
plt.title('Distribution of Age')

# Plot Calories burned distribution
sns.histplot(calories_data['Calories'], kde=True)
plt.title('Distribution of Calories Burned')

# Plot Height and Weight distributions
sns.histplot(calories_data['Height'], kde=True)
plt.title('Distribution of Height')

sns.histplot(calories_data['Weight'], kde=True)
plt.title('Distribution of Weight')
```

## Correlation
Now we have to determine which features in the dataset are related.

## Correlation Analysis
```
# Calculate correlations
numeric_data = calories_data.select_dtypes(include='number')
correlation = numeric_data.corr()

# Visualize correlations using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.2f', annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
```

## Data Preprocessing
```
# Convert Gender from categorical to numerical
calories_data.replace({"Gender": {"male": 0, "female": 1}}, inplace=True)
```

## Model Building and Evaluation
Now we actually have to buld the model

Define Features and Target
```
# Select features and target
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']
```

Train-Test Split
```
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```
Model Training and Evaluation
```
# Instantiate and train XGBoost Regressor
model = XGBRegressor()
model.fit(X_train, Y_train)

# Predict on test data
test_data_prediction = model.predict(X_test)

# Evaluate model performance
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print(f"Mean Absolute Error: {mae}")
```

## Conclusion
This program effectively analyzes exercise data to predict calories burned using XGBoost, achieving a low mean absolute error in predictions. Honestly, coding this program was tough- as I kept getting stuk at the Data Preprocessing Stage- and did not take into account that the "Gender" Column was not being added to the predictions to start off with because they were strings. However, I realized that it was not being included later by plotting everything as you can see in the collab link on the side.

Until next time,
Faizan.
