# Problem 1
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json

# Load the GoodReads data
with open('data/young_adult_10000.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Extract features (number of '!' and star ratings)
exclamation_count = [review['review_text'].count('!') for review in data]
ratings = [review['rating'] for review in data]

# Prepare the features and labels
X = pd.DataFrame(exclamation_count, columns=['exclamation_count'])
y = pd.Series(ratings)

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get θ0 and θ1
theta_0 = model.intercept_
theta_1 = model.coef_[0]

# Predict and calculate MSE
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

print(f"θ0: {theta_0}, θ1: {theta_1}, MSE: {mse}")

# Problem 2
# Add review length as a feature
review_length = [len(review['review_text']) for review in data]

# Prepare the features (exclamation count and review length)
X = pd.DataFrame({'exclamation_count': exclamation_count, 'review_length': review_length})

# Train the model
model_with_length = LinearRegression()
model_with_length.fit(X, y)

# Get coefficients
theta_0_length = model_with_length.intercept_
theta_1_length, theta_2_length = model_with_length.coef_

# Predict and calculate MSE
predictions_with_length = model_with_length.predict(X)
mse_with_length = mean_squared_error(y, predictions_with_length)

print(f"θ0: {theta_0_length}, θ1: {theta_1_length}, θ2: {theta_2_length}, MSE: {mse_with_length}")

# Problem 3
from sklearn.preprocessing import PolynomialFeatures

# Fit polynomial models from degree 1 to 5
for degree in range(1, 6):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X[['exclamation_count']])
    
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    
    predictions_poly = model_poly.predict(X_poly)
    mse_poly = mean_squared_error(y, predictions_poly)
    
    print(f"Degree {degree}, MSE: {mse_poly}")

# Problem 4
from sklearn.model_selection import train_test_split

# Split the data into 50% train and 50% test
X_train, X_test, y_train, y_test = train_test_split(X[['exclamation_count']], y, test_size=0.5, random_state=42)

# Fit polynomial models on training set and evaluate on test set
for degree in range(1, 6):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model_poly_split = LinearRegression()
    model_poly_split.fit(X_train_poly, y_train)
    
    predictions_test_poly = model_poly_split.predict(X_test_poly)
    mse_test_poly = mean_squared_error(y_test, predictions_test_poly)
    
    print(f"Degree {degree}, Test MSE: {mse_test_poly}")

# Problem 5
import numpy as np
from sklearn.metrics import mean_absolute_error

# Best possible predictor is the median of the test set
median_pred = np.median(y_test)

# Calculate MAE
mae_median = mean_absolute_error(y_test, [median_pred]*len(y_test))

print(f"Best constant predictor (median): {median_pred}, MAE: {mae_median}")

# Problem 6
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the beer review data
data_beer = []
with open('data/beer_50000.json', 'r') as f:
    data = f.readline()
    while len(data) > 2:
        data_beer.append(eval(data))
        data = f.readline()

# Now create the DataFrame
data_beer = pd.DataFrame(data_beer)

# Remove rows without gender specified
data_beer = data_beer.dropna(subset=['user/gender'])


# Extract features and labels
X_beer = pd.DataFrame([review.count('!') for review in data_beer['review/text']], columns=['exclamation_count'])
y_beer = data_beer['user/gender'].apply(lambda x: 1 if x == 'Female' else 0)  # 1 for female, 0 for male

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_beer, y_beer)

# Predict
y_pred = log_reg.predict(X_beer)

# Confusion matrix and error metrics
tn, fp, fn, tp = confusion_matrix(y_beer, y_pred).ravel()
balanced_error_rate = (fp / (fp + tn) + fn / (fn + tp)) / 2

print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, Balanced Error Rate: {balanced_error_rate}")

# Problem 7
# Train logistic regression model with balanced class weights
log_reg_balanced = LogisticRegression(class_weight='balanced')
log_reg_balanced.fit(X_beer, y_beer)

# Predict and calculate confusion matrix
y_pred_balanced = log_reg_balanced.predict(X_beer)
tn, fp, fn, tp = confusion_matrix(y_beer, y_pred_balanced).ravel()

# Calculate balanced error rate
balanced_error_rate_balanced = (fp / (fp + tn) + fn / (fn + tp)) / 2

print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, Balanced Error Rate: {balanced_error_rate_balanced}")

# Problem 8
from sklearn.metrics import precision_score

# Sort predictions by predicted probability
y_prob_balanced = log_reg_balanced.predict_proba(X_beer)[:, 1]
sorted_indices = np.argsort(y_prob_balanced)[::-1]

# Calculate precision@K for K = 1, 10, 100, 1000, 10000
for K in [1, 10, 100, 1000, 10000]:
    top_k_indices = sorted_indices[:K]
    precision_at_k = precision_score(y_beer.iloc[top_k_indices], y_pred_balanced[top_k_indices])
    print(f"Precision@{K}: {precision_at_k}")
