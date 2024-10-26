# homework2.py

import json
import gzip
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

# 1. Load and parse the bankruptcy dataset
f = open("data/5year.arff", 'r')

# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l:
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)

# Preparing features and target variables
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# Initialize answers dictionary
answers = {}

# 1. Train a logistic regressor with C = 1.0
model = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
ber = 1 - balanced_accuracy_score(y, y_pred)
answers['Q1'] = [accuracy, ber]
print(accuracy, ber)


model_balanced = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=0)
model_balanced.fit(X, y)
y_pred_balanced = model_balanced.predict(X)
accuracy_balanced = accuracy_score(y, y_pred_balanced)
ber_balanced = 1 - balanced_accuracy_score(y, y_pred_balanced)
answers["Q2"] = [accuracy, ber]
print(accuracy, ber)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

model_split = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=0)
model_split.fit(X_train, y_train)

train_ber = 1 - balanced_accuracy_score(y_train, model_split.predict(X_train))
val_ber = 1 - balanced_accuracy_score(y_val, model_split.predict(X_val))
test_ber = 1 - balanced_accuracy_score(y_test, model_split.predict(X_test))
answers["Q3"] = [train_ber, val_ber, test_ber]

ber_results = {}
for c in [10**i for i in range(-4, 5)]:
    model_reg = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=0)
    model_reg.fit(X_train, y_train)
    val_ber = 1 - balanced_accuracy_score(y_val, model_reg.predict(X_val))
    ber_results[c] = val_ber

answers["Q4"] = ber_results

best_c = min(ber_results, key=ber_results.get)
model_best = LogisticRegression(C=best_c, max_iter=1000, class_weight='balanced', random_state=0)
model_best.fit(X_train, y_train)
test_ber_best = 1 - balanced_accuracy_score(y_test, model_best.predict(X_test))
answers["Q5"] = [best_c, test_ber_best]

with gzip.open('data/young_adult_10000.json.gz', 'rt', encoding='utf-8') as f:
    goodreads_data = [json.loads(line) for line in f]

from sklearn.model_selection import train_test_split

# Split data into training (90%) and test (10%) sets
train_data, test_data = train_test_split(goodreads_data, test_size=0.1, random_state=42)

from collections import defaultdict

# Dictionaries to store users for each item
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)

# Populate the dictionary with users who reviewed each item
for d in train_data:
    user = d['user_id']
    item = d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

# Jaccard similarity calculation for the first item
first_item_id = '2767052'
first_item_users = usersPerItem[first_item_id]

# Calculate Jaccard similarities for each item compared to the first item
jaccard_similarities = []
for item_id, users in usersPerItem.items():
    if item_id != first_item_id:  # Exclude the first item itself
        intersection = first_item_users.intersection(users)
        union = first_item_users.union(users)
        similarity = len(intersection) / len(union)
        jaccard_similarities.append((similarity, item_id))

# Sort and get the top 10 items with the highest Jaccard similarity
top_10_jaccard = sorted(jaccard_similarities, reverse=True, key=lambda x: x[0])[:10]
answers["Q6"] = top_10_jaccard

import numpy as np
from sklearn.metrics import mean_squared_error

# User-item matrix: stores ratings given by each user to each item
user_item_matrix = defaultdict(dict)
for d in train_data:
    user = d['user_id']
    item = d['book_id']
    rating = d['rating']
    user_item_matrix[user][item] = rating

# Prediction function using Jaccard similarity
def predict_rating(user, item, user_item_matrix, usersPerItem):
    # Mean rating of the user
    user_ratings = user_item_matrix.get(user, {})
    user_mean_rating = np.mean(list(user_ratings.values())) if user_ratings else 0

    # Calculate the predicted rating
    num, den = 0, 0
    for other_item in user_ratings:
        if other_item != item:
            # Jaccard similarity between the current item and other items the user rated
            sim = len(usersPerItem[item].intersection(usersPerItem[other_item])) / len(usersPerItem[item].union(usersPerItem[other_item]))
            num += (user_ratings[other_item] - user_mean_rating) * sim
            den += sim

    return user_mean_rating + (num / den if den != 0 else 0)

# Calculate predictions and MSE on test set
y_true, y_pred = [], []
for d in test_data:
    user = d['user_id']
    item = d['book_id']
    actual_rating = d['rating']
    
    # Predict and collect results
    predicted_rating = predict_rating(user, item, user_item_matrix, usersPerItem)
    y_true.append(actual_rating)
    y_pred.append(predicted_rating)

mse_jaccard = mean_squared_error(y_true, y_pred)
answers["Q7"] = mse_jaccard

# Prediction function using user-user similarity
def predict_rating_user_similarity(user, item, user_item_matrix, itemsPerUser):
    # Mean rating for the target item
    item_ratings = {u: user_item_matrix[u][item] for u in usersPerItem[item] if item in user_item_matrix[u]}
    item_mean_rating = np.mean(list(item_ratings.values())) if item_ratings else 0

    # Calculate the predicted rating
    num, den = 0, 0
    for other_user in item_ratings:
        if other_user != user:
            # Jaccard similarity between current user and other users who rated the item
            sim = len(itemsPerUser[user].intersection(itemsPerUser[other_user])) / len(itemsPerUser[user].union(itemsPerUser[other_user]))
            num += (user_item_matrix[other_user][item] - item_mean_rating) * sim
            den += sim

    return item_mean_rating + (num / den if den != 0 else 0)

# Calculate predictions and MSE on test set using user-user similarity
y_true, y_pred = [], []
for d in test_data:
    user = d['user_id']
    item = d['book_id']
    actual_rating = d['rating']
    
    # Predict and collect results
    predicted_rating = predict_rating_user_similarity(user, item, user_item_matrix, itemsPerUser)
    y_true.append(actual_rating)
    y_pred.append(predicted_rating)

mse_user_similarity = mean_squared_error(y_true, y_pred)
answers["Q8"] = mse_user_similarity

with open("answers_hw2.txt", "w") as f:
    f.write(str(answers))