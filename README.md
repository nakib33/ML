import { useState, useMemo } from "react";

const TOPICS = [
  {
    id: 1, emoji: "📈", title: "Linear Regression", level: "Beginner",
    tagline: "Teaching a machine to draw a best-fit line",
    analogy: "🏠 House Price Analogy: Imagine you're trying to predict a house's price. You notice bigger houses cost more. Linear Regression is just drawing the best straight line through a scatter of house size vs price points — so you can predict any new house's price from its size.",
    explanation: "Linear Regression finds the relationship between one or more input features (X) and a continuous output (Y) by fitting a straight line: Y = mX + b. The algorithm adjusts m (slope) and b (intercept) to minimize the total prediction error.",
    keyPoints: [
      "Works for predicting continuous numbers (price, temperature, score)",
      "Assumes a linear relationship between input and output",
      "Mean Squared Error (MSE) measures how wrong our line is",
      "Gradient Descent is how we find the best m and b",
      "Overfitting can happen when the model memorizes training data",
    ],
    interviewQA: [
      { q: "What is the difference between Simple and Multiple Linear Regression?", a: "Simple uses 1 input feature (e.g., size → price). Multiple uses 2+ features (e.g., size + bedrooms + location → price)." },
      { q: "What does R² (R-squared) mean?", a: "R² tells you the % of variance your model explains. R²=1 = perfect model. R²=0 = your model is no better than just guessing the average." },
      { q: "What assumptions does Linear Regression make?", a: "Linearity, independence of errors, homoscedasticity (equal variance), normality of residuals, and no multicollinearity." },
      { q: "How do you handle outliers?", a: "Use robust regression, remove outliers after investigation, or use regularization (Ridge/Lasso) to reduce their influence." },
    ],
    code: `import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simple example: predict salary from years of experience
X = np.array([[1], [2], [3], [5], [7], [10]])  # years
y = np.array([35000, 40000, 48000, 60000, 75000, 95000])  # salary

model = LinearRegression()
model.fit(X, y)

print(f"Slope (m): {model.coef_[0]:.0f}")        # salary increase per year
print(f"Intercept (b): {model.intercept_:.0f}")   # base salary

# Predict salary for 6 years experience
pred = model.predict([[6]])
print(f"Predicted salary at 6 years: ${pred[0]:.0f}")

# Score the model
print(f"R² score: {model.score(X, y):.3f}")  # closer to 1 = better`,
  },
  {
    id: 2, emoji: "🎯", title: "Logistic Regression", level: "Beginner",
    tagline: "Yes or No? Teaching machines to classify",
    analogy: "🚦 Spam Filter Analogy: Is this email spam or not? Logistic Regression doesn't draw a line to predict a number — it draws a line to separate two groups (spam vs not-spam) and outputs a PROBABILITY (0 to 1) of belonging to each group.",
    explanation: "Logistic Regression is used for classification (not regression despite the name!). It applies the sigmoid function to transform the linear output into a probability between 0 and 1. If the probability > 0.5, predict class 1; otherwise class 0.",
    keyPoints: [
      "Output is a probability between 0 and 1 (not a raw number)",
      "Uses sigmoid function: σ(x) = 1 / (1 + e^(-x))",
      "Decision boundary is where probability = 0.5",
      "Loss function is Binary Cross-Entropy (not MSE)",
      "Can be extended to multi-class with Softmax",
    ],
    interviewQA: [
      { q: "Why can't we use Linear Regression for classification?", a: "Linear Regression outputs any number (-∞ to +∞). For classification we need probabilities (0 to 1). Also, it's sensitive to outliers and doesn't model the S-shaped probability curve well." },
      { q: "What is the sigmoid function and why use it?", a: "Sigmoid = 1/(1+e^-x). It squashes any real number to (0,1), making it interpretable as a probability. It has smooth gradients useful for optimization." },
      { q: "What is Binary Cross-Entropy loss?", a: "BCE = -[y·log(p) + (1-y)·log(1-p)]. It penalizes confident wrong predictions heavily. Much better than MSE for classification tasks." },
      { q: "How do you handle class imbalance in Logistic Regression?", a: "Use class_weight='balanced', oversample minority class (SMOTE), undersample majority class, or adjust the decision threshold." },
    ],
    code: `from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load cancer dataset (predict malignant vs benign)
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:3]  # probabilities for first 3

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\\nProbabilities for first 3 samples:")
print("  [P(malignant), P(benign)]")
for p in probs:
    print(f"  {p}")

print("\\n", classification_report(y_test, y_pred,
      target_names=['malignant', 'benign']))`,
  },
  {
    id: 3, emoji: "🌳", title: "Decision Tree", level: "Beginner",
    tagline: "A flowchart that makes decisions",
    analogy: "🎮 20 Questions Game: 'Is it an animal? Does it have 4 legs? Can it fly?' A Decision Tree is literally this — a series of yes/no questions that splits data into groups, arriving at a prediction at the leaves.",
    explanation: "A Decision Tree asks questions about features to split data. At each node, it picks the feature and threshold that best separates the classes (measured by Gini impurity or Information Gain). It keeps splitting until it reaches pure leaf nodes or a stopping criterion.",
    keyPoints: [
      "Splits data at each node based on the best feature/threshold",
      "Gini Impurity: how mixed a node's classes are (0 = pure, 0.5 = half-half)",
      "Information Gain: how much a split reduces uncertainty",
      "max_depth controls overfitting (deeper = more complex)",
      "Very easy to visualize and explain to non-technical people",
    ],
    interviewQA: [
      { q: "What is Gini Impurity?", a: "Gini = 1 - Σ(pᵢ²). It measures how often a randomly chosen element would be incorrectly classified. A node with all same class has Gini=0 (pure). A 50/50 split has Gini=0.5." },
      { q: "What is Information Gain?", a: "IG = Entropy(parent) - weighted average Entropy(children). We pick the split with the highest IG because it gives us the most new information about the target." },
      { q: "Why do Decision Trees overfit?", a: "A fully grown tree memorizes training data (every leaf = one sample). It learns noise, not patterns. Fix: limit max_depth, min_samples_split, or prune the tree." },
      { q: "What's the difference between CART and ID3?", a: "ID3 uses Information Gain and handles only categorical features. CART uses Gini Impurity and handles both categorical and continuous features, and is used by sklearn." },
    ],
    code: `from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

# Classify iris flowers (3 species)
iris = load_iris()
X, y = iris.data, iris.target

# Shallow tree to avoid overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# See the actual decision rules as text!
rules = export_text(model, feature_names=iris.feature_names)
print(rules)

# Feature importance - which features matter most?
for name, imp in zip(iris.feature_names, model.feature_importances_):
    print(f"{name:30s}: {imp:.3f} ({'█' * int(imp * 30)})")

print(f"\\nTraining accuracy: {model.score(X, y):.2%}")`,
  },
  {
    id: 4, emoji: "🌲", title: "Random Forest", level: "Intermediate",
    tagline: "Wisdom of the crowd — but for trees",
    analogy: "👥 Ask Many Experts: Instead of asking one person for a decision, ask 100 different experts (each trained on slightly different data) and take a majority vote. Random Forest does exactly this with Decision Trees — many trees, each seeing random subsets of data and features, then voting together.",
    explanation: "Random Forest builds many Decision Trees using Bootstrap Aggregation (Bagging): each tree trains on a random sample of data (with replacement) and a random subset of features. Final prediction = majority vote (classification) or average (regression). This reduces variance and overfitting dramatically.",
    keyPoints: [
      "Bagging: each tree trains on a random bootstrap sample",
      "Feature randomness: each split considers only √(n_features) features",
      "Final answer = majority vote (classification) / average (regression)",
      "Out-of-Bag (OOB) error: free validation on samples not used in training",
      "Feature importances: which features contribute most across all trees",
    ],
    interviewQA: [
      { q: "Why is Random Forest better than a single Decision Tree?", a: "A single tree overfits. Random Forest introduces two types of randomness (data sampling + feature sampling) making trees diverse. Diverse trees that each make different errors cancel out those errors when averaged/voted." },
      { q: "What is Out-of-Bag (OOB) error?", a: "Each bootstrap sample leaves out ~37% of data. These unused samples can validate each tree for free! OOB error is the average validation error across all trees using their respective out-of-bag samples." },
      { q: "What's the difference between Bagging and Boosting?", a: "Bagging (Random Forest) trains trees in parallel, independently, and averages results to reduce variance. Boosting (XGBoost) trains trees sequentially, where each tree fixes the previous tree's mistakes, reducing bias." },
      { q: "How many trees should you use?", a: "More trees = more stable, but diminishing returns. Usually 100-500 trees is enough. Use OOB error to monitor when adding more trees stops helping." },
    ],
    code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42)

# 100 trees, using OOB scoring
model = RandomForestClassifier(
    n_estimators=100, 
    oob_score=True,      # free validation!
    random_state=42
)
model.fit(X_train, y_train)

print(f"Test Accuracy:  {model.score(X_test, y_test):.2%}")
print(f"OOB Score:      {model.oob_score_:.2%}  (free validation!)")

# Top 5 most important features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:5]
print("\\nTop 5 most important features:")
for rank, i in enumerate(indices, 1):
    bar = '█' * int(importances[i] * 50)
    print(f"  {rank}. {wine.feature_names[i]:25s} {bar} {importances[i]:.3f}")`,
  },
  {
    id: 5, emoji: "🚀", title: "XGBoost", level: "Intermediate",
    tagline: "The Kaggle competition killer — sequential boosting",
    analogy: "📚 Student Improving From Mistakes: A student takes a test, fails some questions. They study specifically those failed questions, take the test again, fail different ones, study those... Each attempt corrects prior mistakes. XGBoost does this with trees — each new tree focuses on the errors of all previous trees.",
    explanation: "XGBoost (Extreme Gradient Boosting) builds trees sequentially. Each tree is trained to predict the RESIDUAL ERRORS of all previous trees combined. It uses second-order gradients (Newton's method) for faster convergence, plus L1/L2 regularization to prevent overfitting.",
    keyPoints: [
      "Gradient Boosting: each tree learns from the errors of previous trees",
      "Uses 2nd-order Taylor expansion for faster, more accurate optimization",
      "Built-in L1 (Lasso) and L2 (Ridge) regularization",
      "Handles missing values automatically",
      "learning_rate × n_estimators controls model complexity",
    ],
    interviewQA: [
      { q: "What makes XGBoost faster than regular Gradient Boosting?", a: "XGBoost uses: (1) approximate tree building algorithms, (2) column subsampling like Random Forest, (3) parallel processing within trees, (4) cache-aware access patterns, and (5) out-of-core computing for large datasets." },
      { q: "What is the learning_rate (eta) hyperparameter?", a: "It shrinks each tree's contribution (0 < lr ≤ 1). Lower lr = each tree matters less, need more trees = more regularization. Common practice: set lr=0.01 or 0.1, then tune n_estimators with early stopping." },
      { q: "What is early stopping in XGBoost?", a: "Training stops when validation performance doesn't improve for N consecutive rounds. Prevents overfitting without manual tuning of n_estimators. Use early_stopping_rounds=50 typically." },
      { q: "XGBoost vs LightGBM vs CatBoost — when to use which?", a: "XGBoost: reliable baseline, great support. LightGBM: faster training on large data (leaf-wise growth). CatBoost: best for categorical features, less tuning needed. Start with XGBoost, switch if speed/categoricals matter." },
    ],
    code: `import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,       # shrink each tree's contribution
    max_depth=4,             # tree depth
    subsample=0.8,           # use 80% of data per tree
    colsample_bytree=0.8,    # use 80% of features per tree
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)])

print(f"Best iteration: {model.best_iteration}")
print(f"Test Accuracy:  {accuracy_score(y_test, model.predict(X_test)):.2%}")

# SHAP-style feature importance
print("\\nTop 5 features by importance:")
import numpy as np
scores = model.feature_importances_
top = np.argsort(scores)[::-1][:5]
for i in top:
    print(f"  feature_{i}: {scores[i]:.4f}")`,
  },
  {
    id: 6, emoji: "⚡", title: "LightGBM", level: "Intermediate",
    tagline: "XGBoost's faster cousin — leaf-wise growth",
    analogy: "🌿 Smart Tree Trimmer: Instead of growing a tree level by level (like XGBoost), LightGBM grows the single most promising leaf first — like a gardener who always waters the most thirsty plant first. This makes training much faster while keeping accuracy high.",
    explanation: "LightGBM introduces two key innovations: (1) Gradient-based One-Side Sampling (GOSS) — keep all high-gradient samples but randomly sample low-gradient ones; (2) Exclusive Feature Bundling (EFB) — bundle sparse features together. Result: much faster training with similar accuracy.",
    keyPoints: [
      "Leaf-wise tree growth vs level-wise (XGBoost): faster, but prone to overfitting on small data",
      "GOSS: focuses on hard samples (large gradient = large error)",
      "EFB: bundles mutually exclusive features to reduce feature count",
      "Handles large datasets with millions of rows efficiently",
      "num_leaves is the key hyperparameter (replaces max_depth logic)",
    ],
    interviewQA: [
      { q: "What is leaf-wise vs level-wise tree growth?", a: "Level-wise: grow all leaves of a level before going deeper (balanced tree). Leaf-wise: always split the leaf with highest loss reduction (asymmetric tree). Leaf-wise converges faster but can overfit on small datasets." },
      { q: "When should you choose LightGBM over XGBoost?", a: "Choose LightGBM when: (1) dataset > 100K rows, (2) training speed is critical, (3) memory is limited. Use XGBoost for smaller datasets or when you need more tuning control." },
      { q: "How does LightGBM handle categorical features?", a: "LightGBM has native support for categoricals — just mark them with categorical_feature parameter. It uses an optimal split finding algorithm for categories, often better than one-hot encoding." },
      { q: "What does min_child_samples do?", a: "Sets the minimum number of samples required in a leaf. Higher value = more regularization = less overfitting. Very important for LightGBM since leaf-wise growth can create tiny leaves." },
    ],
    code: `import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,          # max leaves per tree (key param!)
    'learning_rate': 0.05,
    'feature_fraction': 0.8,   # like colsample_bytree
    'bagging_fraction': 0.8,   # like subsample
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(
    params, train_data, 
    num_boost_round=200,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)

preds = (model.predict(X_test) > 0.5).astype(int)
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")`,
  },
  {
    id: 7, emoji: "🐱", title: "CatBoost", level: "Intermediate",
    tagline: "Gradient boosting that loves categorical data",
    analogy: "🏷️ Smart Label Handler: Most algorithms hate categorical data (Male/Female, City names) and require manual encoding. CatBoost is like a chef who natively understands ingredients without you having to chop them first — it handles categoricals automatically and intelligently.",
    explanation: "CatBoost (Categorical Boosting) uses Ordered Boosting to prevent target leakage during training, and Target Statistics to encode categorical features without overfitting. It often requires minimal preprocessing and performs excellently on datasets with many categorical features out of the box.",
    keyPoints: [
      "Native categorical feature support — no manual encoding needed",
      "Ordered Boosting: uses only previous samples to compute residuals (prevents target leakage)",
      "Target Statistics: encodes categories using target mean (with Bayesian smoothing)",
      "Often works well with default hyperparameters",
      "GPU support for fast training",
    ],
    interviewQA: [
      { q: "What is target leakage and how does CatBoost prevent it?", a: "Target leakage: using information about the target from the same sample to encode its features (circular reasoning). CatBoost uses Ordered Boosting — when computing stats for sample i, only uses samples 1 to i-1." },
      { q: "How does CatBoost encode categorical features internally?", a: "Target Statistics: for category C, encode as P(target | category) using prior samples + a Bayesian prior. This avoids the curse of high-cardinality one-hot encoding and captures target-feature relationships." },
      { q: "When is CatBoost the best choice?", a: "When you have many high-cardinality categorical features, limited time for preprocessing, or need a model that works well out-of-the-box. Also great when you want symmetric trees for faster inference." },
      { q: "What are symmetric trees in CatBoost?", a: "CatBoost uses oblivious trees — all nodes at the same depth use the same split condition. This prevents overfitting, enables faster prediction (just follow the same condition at each level), and makes the model size predictable." },
    ],
    code: `from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset with categorical features (no encoding needed!)
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age':     np.random.randint(18, 70, n),
    'city':    np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n),
    'job':     np.random.choice(['engineer', 'doctor', 'teacher', 'artist'], n),
    'income':  np.random.normal(60000, 20000, n),
    'bought':  np.random.randint(0, 2, n)   # target
})

X = df.drop('bought', axis=1)
y = df['bought']
cat_features = ['city', 'job']  # just tell CatBoost which are categorical!

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    cat_features=cat_features,   # no encoding needed!
    verbose=50
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))
print(f"\\nAccuracy: {model.score(X_test, y_test):.2%}")`,
  },
  {
    id: 8, emoji: "🔔", title: "Naive Bayes", level: "Beginner",
    tagline: "Probability-based classification using Bayes' theorem",
    analogy: "🕵️ Detective with Prior Knowledge: A detective sees clues (features) and updates their suspicion about a crime. 'The criminal is probably tall AND drives a red car AND was seen near the bank.' Naive Bayes multiplies individual probabilities together, 'naively' assuming each clue is independent.",
    explanation: "Naive Bayes applies Bayes' theorem: P(class | features) ∝ P(features | class) × P(class). It's 'naive' because it assumes all features are independent of each other given the class. Despite this unrealistic assumption, it works surprisingly well for text classification and spam detection.",
    keyPoints: [
      "Based on Bayes' Theorem: P(A|B) = P(B|A) × P(A) / P(B)",
      "Naive assumption: features are independent given the class",
      "GaussianNB for continuous features, MultinomialNB for counts (text), BernoulliNB for binary",
      "Very fast training and prediction — O(nd) time",
      "Works great for text classification, spam detection, sentiment analysis",
    ],
    interviewQA: [
      { q: "Why is Naive Bayes called 'naive'?", a: "Because it naively assumes all features are conditionally independent given the class label. In reality, features are often correlated (e.g., 'free' and 'win' in spam emails). Despite this, it still works well in practice." },
      { q: "What is the Zero Frequency Problem and how do you fix it?", a: "If a feature value never appeared in training with a class, P(feature|class)=0, making the entire product 0 (one zero kills everything). Fix: Laplace Smoothing — add 1 to all counts so no frequency is ever zero." },
      { q: "When should you use Naive Bayes?", a: "Use when: (1) very fast training/prediction needed, (2) limited training data (works well with small datasets), (3) text classification tasks, (4) as a baseline before trying complex models." },
      { q: "What is the difference between GaussianNB, MultinomialNB, and BernoulliNB?", a: "GaussianNB: features follow normal distribution (continuous data). MultinomialNB: features are counts (word frequencies, TF-IDF). BernoulliNB: features are binary (word present/absent)." },
    ],
    code: `from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Classic use case: Spam Detection
emails = [
    "Win a free iPhone now click here",
    "Congratulations you won $1000 prize",
    "Free money limited time offer",
    "Meeting at 3pm tomorrow about project",
    "Please review the attached document",
    "Team lunch this Friday at noon",
    "Your invoice for last month services",
    "Click here to claim your free gift",
]
labels = [1, 1, 1, 0, 0, 0, 0, 1]  # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.25, random_state=42)

# Pipeline: text → counts → Naive Bayes
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB(alpha=1.0))  # alpha=Laplace smoothing
])
pipeline.fit(X_train, y_train)

# Test on new emails
new_emails = ["Urgent: claim your free prize now!", "See you at the meeting"]
predictions = pipeline.predict(new_emails)
for email, pred in zip(new_emails, predictions):
    print(f"{'SPAM' if pred == 1 else 'HAM ':4s} | {email}")`,
  },
  {
    id: 9, emoji: "✂️", title: "SVM", level: "Intermediate",
    tagline: "Find the widest street between two neighborhoods",
    analogy: "🏙️ City Divider: Imagine two neighborhoods (red and blue houses) on a map. SVM finds the widest possible road you can draw between them — such that red houses are on one side and blue on the other. The 'support vectors' are the houses closest to this road.",
    explanation: "Support Vector Machine finds the optimal hyperplane that maximizes the margin between classes. Points closest to the hyperplane are called support vectors — they define the boundary. The kernel trick allows SVM to find non-linear boundaries by implicitly projecting data into higher dimensions.",
    keyPoints: [
      "Maximizes the margin between classes (max-margin classifier)",
      "Support vectors: training points closest to the decision boundary",
      "Kernel trick: RBF/polynomial kernels for non-linear boundaries",
      "C parameter: controls bias-variance tradeoff (high C = low bias, high variance)",
      "γ (gamma): how far each training point's influence reaches",
    ],
    interviewQA: [
      { q: "What is the kernel trick?", a: "Instead of actually mapping data to higher dimensions (expensive!), kernel functions compute dot products in that high-dimensional space cheaply. K(x,z) = φ(x)·φ(z). Common kernels: Linear, Polynomial, RBF (Gaussian)." },
      { q: "What is the C parameter in SVM?", a: "C controls how much you penalize misclassifications. High C: tries to classify all training points correctly, smaller margin, can overfit. Low C: allows some misclassifications for a wider margin, more regularized." },
      { q: "When is SVM a good choice?", a: "Good when: high-dimensional data (text, genomics), small-to-medium datasets (slow on large data), clear margin of separation exists, or you need a non-probabilistic classifier." },
      { q: "What's the difference between hard margin and soft margin SVM?", a: "Hard margin: no misclassifications allowed (only works if data is linearly separable). Soft margin: allows some misclassifications (controlled by C). In practice, always use soft margin SVM." },
    ],
    code: `from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# Non-linearly separable data (moons shape)
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# IMPORTANT: Always scale features for SVM!
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),   # SVM is sensitive to scale!
    ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True))
])

# Tune hyperparameters
param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.1, 1]}
grid = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params:    {grid.best_params_}")
print(f"CV Accuracy:    {grid.best_score_:.2%}")
print(f"Test Accuracy:  {grid.score(X_test, y_test):.2%}")`,
  },
  {
    id: 10, emoji: "👥", title: "KNN", level: "Beginner",
    tagline: "You are who you're closest to",
    analogy: "🏘️ Neighborhood Voting: New person moves to town — what's their political leaning? Look at their 5 nearest neighbors. If 4 are conservative and 1 is liberal → predict conservative. KNN is literally just this: find K nearest training points, take a majority vote.",
    explanation: "K-Nearest Neighbors is the simplest ML algorithm. To classify a new point: (1) calculate distance to ALL training points, (2) find K closest points, (3) take majority class vote. No training step — all computation happens at prediction time (lazy learner).",
    keyPoints: [
      "Non-parametric: no training phase, memorizes all training data",
      "K=1: overfits. K=N: underfits. Good K is found by cross-validation",
      "Distance metric matters: Euclidean, Manhattan, Minkowski",
      "Feature scaling is CRUCIAL (large-scale features dominate distance)",
      "Slow prediction: O(n×d) per query. Not suitable for large datasets",
    ],
    interviewQA: [
      { q: "How do you choose the optimal K?", a: "Use cross-validation: try odd K values (avoids ties) from 1 to √N. Plot validation error vs K. K=1 overfits, large K underfits. The 'elbow' in the plot is often a good choice." },
      { q: "Why must you normalize features for KNN?", a: "KNN uses distance. If feature A ranges 0-1 but feature B ranges 0-1000, feature B dominates distance calculations. Normalize all features to [0,1] or standardize to mean=0, std=1." },
      { q: "What is the curse of dimensionality for KNN?", a: "In high dimensions, all points become equally far apart. The concept of 'nearest neighbor' loses meaning because distances converge. KNN degrades badly with many features — use PCA first." },
      { q: "KNN for regression vs classification?", a: "Classification: majority vote of K neighbors. Regression: average (or weighted average) of K neighbors' target values. KNeighborsRegressor in sklearn." },
    ],
    code: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

# Find the best K using cross-validation
best_k, best_score = 1, 0
scores_list = []

for k in range(1, 21):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),   # MUST scale for KNN!
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    mean_score = cv_scores.mean()
    scores_list.append((k, mean_score))
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print("K  |  CV Accuracy")
print("-" * 25)
for k, score in scores_list[:10]:
    bar = '█' * int(score * 20)
    print(f"K={k:2d} | {score:.2%} {bar}")

print(f"\\nBest K: {best_k} with accuracy {best_score:.2%}")`,
  },
  {
    id: 11, emoji: "🔵", title: "Clustering", level: "Intermediate",
    tagline: "Finding natural groups — without labels",
    analogy: "🛒 Supermarket Shopper Groups: A store has thousands of customers. Without knowing their categories, can you discover that some are 'budget shoppers', others are 'health food fans', others are 'impulse buyers'? Clustering finds these natural groups from patterns in the data, with no predefined labels.",
    explanation: "Clustering is unsupervised learning — finding groups in data without labels. K-Means assigns points to K centroids and iterates. DBSCAN finds dense regions and marks outliers as noise. Hierarchical Clustering builds a tree (dendrogram) of nested clusters. Each works differently and suits different data shapes.",
    keyPoints: [
      "Unsupervised: no labels needed — discover structure in data",
      "K-Means: assign to nearest centroid, update centroid, repeat",
      "K-Means weakness: assumes spherical clusters, sensitive to outliers",
      "DBSCAN: density-based, finds arbitrary shapes, handles noise/outliers",
      "Elbow Method / Silhouette Score to choose K",
    ],
    interviewQA: [
      { q: "How does K-Means work step by step?", a: "1) Initialize K centroids randomly. 2) Assign each point to nearest centroid. 3) Recompute centroids as cluster means. 4) Repeat 2-3 until centroids stop moving. Convergence guaranteed but may be local minimum — run multiple times." },
      { q: "How do you choose K in K-Means?", a: "Elbow Method: plot inertia (sum of squared distances to centroids) vs K. Find the 'elbow' where adding more K stops helping much. Silhouette Score: measures how tight clusters are vs how separated — higher is better." },
      { q: "What is DBSCAN and when is it better than K-Means?", a: "DBSCAN groups points with enough nearby neighbors (density). It finds clusters of arbitrary shape, doesn't require K, and automatically labels outliers as noise (-1). Better when clusters aren't spherical or you have outliers." },
      { q: "What is the Silhouette Score?", a: "For each point: s = (b - a) / max(a, b), where a = average distance to same cluster, b = average distance to nearest other cluster. Range [-1,1]. Score near 1 = well clustered. Near 0 = on boundary. Negative = wrong cluster." },
    ],
    code: `from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

np.random.seed(42)
# 3 natural clusters
X = np.vstack([
    np.random.normal([0, 0],   0.5, (50, 2)),   # cluster 1
    np.random.normal([5, 5],   0.5, (50, 2)),   # cluster 2
    np.random.normal([10, 0],  0.5, (50, 2)),   # cluster 3
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means: try K=2,3,4 ---
print("K-Means Silhouette Scores:")
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"  K={k}: {score:.3f} {'← best' if k == 3 else ''}")

# --- DBSCAN: no K needed ---
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(X_scaled)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"\\nDBSCAN found {n_clusters} clusters, {n_noise} noise points")`,
  },
  {
    id: 12, emoji: "📉", title: "PCA", level: "Intermediate",
    tagline: "Simplify complexity — keep what matters",
    analogy: "📸 Photography Shadow: A 3D object has a 2D shadow. The shadow captures most of the object's shape in fewer dimensions. PCA finds the 'best angle' to project your high-dimensional data into fewer dimensions while preserving as much structure (variance) as possible.",
    explanation: "Principal Component Analysis finds new axes (principal components) that capture the maximum variance. PC1 captures the most variance, PC2 captures the second most (orthogonal to PC1), etc. By keeping only top K components, you compress data while retaining most information.",
    keyPoints: [
      "Finds directions (PCs) of maximum variance in data",
      "Components are orthogonal (uncorrelated) to each other",
      "Explained variance ratio: how much of total variance each PC captures",
      "Scale features first! PCA is sensitive to feature magnitudes",
      "Used for: dimensionality reduction, noise reduction, visualization, speed-up",
    ],
    interviewQA: [
      { q: "How does PCA work mathematically?", a: "1) Center data (subtract mean). 2) Compute covariance matrix. 3) Find eigenvectors and eigenvalues. 4) Sort eigenvectors by eigenvalue (descending). 5) Project data onto top K eigenvectors. The eigenvectors are the principal components." },
      { q: "How many components should you keep?", a: "Keep enough components to explain 95% (or 99%) of total variance. Plot the cumulative explained variance ratio and pick the 'knee' point, or set n_components=0.95 in sklearn to auto-select." },
      { q: "PCA vs t-SNE vs UMAP — what's the difference?", a: "PCA: linear, fast, good for general dimensionality reduction. t-SNE: non-linear, great for visualization (2D/3D), but slow and non-deterministic. UMAP: non-linear, faster than t-SNE, preserves both local and global structure better." },
      { q: "Does PCA help with overfitting?", a: "Indirectly. By reducing dimensions, PCA reduces the number of features the model sees, which can reduce overfitting. But it's primarily used for computational efficiency and visualization, not regularization." },
    ],
    code: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

# Digits: 64 features (8×8 pixels) → reduce to fewer
digits = load_digits()
X = digits.data  # shape: (1797, 64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# How many components explain 95% of variance?
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Original features:        64")
print(f"Features for 95% variance: {n_95}")
print(f"Compression ratio:         {64/n_95:.1f}x")

# Apply PCA with optimal components
pca = PCA(n_components=n_95)
X_reduced = pca.fit_transform(X_scaled)
print(f"Shape before: {X_scaled.shape}")
print(f"Shape after:  {X_reduced.shape}")

# Show per-component variance
print("\\nTop 5 PCs and their explained variance:")
for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
    bar = '█' * int(var * 200)
    print(f"  PC{i}: {var:.2%} {bar}")`,
  },
  {
    id: 13, emoji: "🧠", title: "Neural Networks", level: "Intermediate",
    tagline: "Layers of math inspired by the human brain",
    analogy: "🧪 Recipe Mixing: A neuron is like a mixer. It takes inputs (ingredients), multiplies by weights (how much of each ingredient), adds a bias (seasoning adjustment), then passes through an activation function (cooking transformation). Stack layers of these mixers, and you can approximate any function.",
    explanation: "A neural network is layers of neurons: Input → [Hidden layers] → Output. Each neuron computes: output = activation(W·x + b). Backpropagation computes gradients of the loss with respect to each weight using chain rule. Gradient Descent then updates weights to minimize loss.",
    keyPoints: [
      "Forward pass: compute predictions layer by layer",
      "Loss function: measures how wrong the prediction is",
      "Backpropagation: chain rule to compute gradients efficiently",
      "Gradient Descent: update weights in direction of steepest loss decrease",
      "Activation functions: ReLU (hidden layers), Softmax/Sigmoid (output)",
    ],
    interviewQA: [
      { q: "Why do we need activation functions?", a: "Without activation functions, stacking linear layers = still one big linear function. Activations add non-linearity, allowing networks to learn complex patterns. ReLU = max(0,x) is most popular: simple, prevents vanishing gradients." },
      { q: "What is the vanishing gradient problem?", a: "During backpropagation, gradients are multiplied together through layers. Sigmoid/tanh squash gradients near 0, so early layers receive ~0 gradient and learn nothing. Fix: use ReLU (gradient is 1 for x>0), batch normalization, or residual connections." },
      { q: "What is Batch Normalization?", a: "After each layer, normalize activations to mean=0, std=1, then scale with learned γ and β. Benefits: faster training, higher learning rates, acts as regularization, and mitigates vanishing/exploding gradients." },
      { q: "What's the difference between batch, stochastic, and mini-batch gradient descent?", a: "Batch GD: use all data per update (accurate but slow). SGD: one sample per update (fast but noisy). Mini-batch (standard): small batches (32-256) — balances accuracy and speed, used in practice." },
    ],
    code: `import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
scaler = StandardScaler()
X = torch.FloatTensor(scaler.fit_transform(X))
y = torch.LongTensor(y)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),     # input layer
            nn.ReLU(),             # activation
            nn.BatchNorm1d(64),    # batch normalization
            nn.Dropout(0.3),       # regularization
            nn.Linear(64, 32),     # hidden layer
            nn.ReLU(),
            nn.Linear(32, 2)       # output layer (2 classes)
        )
    def forward(self, x):
        return self.layers(x)

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()       # reset gradients
    outputs = model(X)          # forward pass
    loss = criterion(outputs, y)  # compute loss
    loss.backward()             # backpropagation!
    optimizer.step()            # update weights
    if epoch % 10 == 0:
        acc = (outputs.argmax(1) == y).float().mean()
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.2%}")`,
  },
  {
    id: 14, emoji: "👁️", title: "CNN", level: "Advanced",
    tagline: "Teaching machines to see",
    analogy: "🔍 Magnifying Glass on Images: When you read a page, your eyes don't process the whole page at once — they focus on small regions. CNN does exactly this with 'filters' (small windows) that slide across an image detecting edges, textures, then shapes, then objects — from simple to complex, layer by layer.",
    explanation: "Convolutional Neural Networks process images with three key operations: Convolution (feature detection with learnable filters), Pooling (downsampling to reduce size), and Fully Connected (final classification). Filters learn to detect edges → textures → parts → objects hierarchically.",
    keyPoints: [
      "Convolution: sliding filter computes dot product at each position",
      "Filters learn features automatically (no hand-crafting)",
      "Pooling: Max or Average pooling reduces spatial dimensions",
      "Feature maps: output of applying a filter to the whole image",
      "Transfer Learning: use pretrained CNNs (ResNet, VGG) as feature extractors",
    ],
    interviewQA: [
      { q: "What is a convolution operation?", a: "Slide a small filter (e.g., 3×3) across the input image. At each position, compute element-wise multiplication and sum. The result is a 'feature map' showing where/how strongly that pattern appears in the image." },
      { q: "Why use CNNs instead of fully connected networks for images?", a: "(1) Weight sharing: same filter applied everywhere — fewer parameters. (2) Translation invariance: can detect a cat regardless of where it is in the image. (3) Local connectivity: nearby pixels are related; no need to connect to all pixels." },
      { q: "What is Max Pooling?", a: "Take a 2×2 window, keep only the maximum value. Reduces spatial size by 2×, making the model robust to small translations. Also reduces computation and helps prevent overfitting." },
      { q: "What is Transfer Learning?", a: "Use a CNN pretrained on ImageNet (e.g., ResNet, VGG) as a starting point. Freeze early layers (they detect generic edges/textures), retrain final layers on your specific task. Works great with small datasets." },
    ],
    code: `import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Build a CNN from scratch
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 28x28 → 14x14

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # 64 × 7 × 7 = 3136
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN(num_classes=10)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# For real projects, use Transfer Learning:
# pretrained = models.resnet18(pretrained=True)
# pretrained.fc = nn.Linear(512, num_classes)  # replace last layer`,
  },
  {
    id: 15, emoji: "🔄", title: "RNN / LSTM / GRU", level: "Advanced",
    tagline: "Memory for sequential data",
    analogy: "📖 Reading a Book: When you read 'The bank was steep' vs 'The bank charged fees', the meaning of 'bank' depends on context. RNNs work like reading — they have memory of previous words when processing the current one. LSTM adds selective memory: it can remember things from very early in the sequence.",
    explanation: "Recurrent Neural Networks process sequences by maintaining a hidden state that captures context from previous steps. The vanishing gradient problem limits RNN memory. LSTM solves this with gates (forget gate, input gate, output gate) that control what to remember, forget, and output. GRU is a simplified version of LSTM.",
    keyPoints: [
      "RNN processes sequences step by step, passing hidden state forward",
      "Vanishing gradient: RNNs forget long-range dependencies",
      "LSTM: Cell state + 3 gates (forget, input, output) for long memory",
      "GRU: 2 gates (reset, update), simpler and often as good as LSTM",
      "Bidirectional: process sequence forward AND backward",
    ],
    interviewQA: [
      { q: "What is the vanishing gradient problem in RNNs?", a: "Gradients are multiplied together across time steps during backpropagation. If weights < 1, gradients shrink exponentially → early time steps get ~0 gradient and can't be learned. LSTM/GRU solve this with additive gradient paths through the cell state." },
      { q: "Explain LSTM gates.", a: "Forget gate: 'what to erase from memory.' Input gate: 'what new info to add to memory.' Output gate: 'what to output from memory.' Cell state (conveyor belt): the long-term memory that flows through with minimal modification." },
      { q: "LSTM vs GRU — when to use which?", a: "GRU is faster, uses less memory, often performs equally well on shorter sequences. LSTM is better for very long sequences requiring complex memory. Start with GRU; switch to LSTM if not enough performance." },
      { q: "What is Bidirectional RNN?", a: "Process the sequence both forward (t=1→T) and backward (t=T→1), concatenate hidden states. Useful when full context of a sequence is available (e.g., machine translation, NER). Not suitable for online/streaming prediction." },
    ],
    code: `import torch
import torch.nn as nn

# Compare RNN vs LSTM vs GRU
class SequenceModel(nn.Module):
    def __init__(self, model_type='lstm', input_size=10, hidden_size=64, num_classes=2):
        super().__init__()
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.rnn(x)      # out shape: (batch, seq_len, hidden)
        out = out[:, -1, :]       # take last time step
        return self.fc(out)

# Create fake sequence data: (batch=32, seq_len=50, features=10)
x = torch.randn(32, 50, 10)

for model_type in ['rnn', 'lstm', 'gru']:
    model = SequenceModel(model_type=model_type)
    params = sum(p.numel() for p in model.parameters())
    output = model(x)
    print(f"{model_type.upper():4s} | Params: {params:,} | Output shape: {output.shape}")`,
  },
  {
    id: 16, emoji: "🤖", title: "Transformers", level: "Advanced",
    tagline: "Attention is all you need — the architecture behind GPT & BERT",
    analogy: "👀 Spotlight on the Stage: A theatre director watching a play focuses attention on different actors at different moments. Transformers work similarly — when processing each word, they 'pay attention' to ALL other words in the sequence and decide how much each one matters for understanding the current word.",
    explanation: "Transformers use Self-Attention: for each token, compute how relevant every other token is, then create a weighted combination. Multi-Head Attention runs multiple attention heads in parallel. Positional Encoding adds position information. The Encoder-Decoder architecture (used in translation) was replaced by decoder-only (GPT) and encoder-only (BERT) variants.",
    keyPoints: [
      "Self-Attention: each token attends to all other tokens",
      "Q, K, V matrices: Query, Key, Value (attention = softmax(QK^T/√d)·V)",
      "Multi-Head Attention: multiple attention heads capture different relationships",
      "Positional Encoding: adds position info (since attention is permutation-invariant)",
      "BERT: encoder-only for understanding. GPT: decoder-only for generation",
    ],
    interviewQA: [
      { q: "Explain the Query, Key, Value mechanism.", a: "Think of a library: Query = your question, Keys = book titles, Values = book content. Attention = how well your question matches each title → use those scores to blend the book contents. Q·K^T gives relevance scores, V provides content." },
      { q: "Why are Transformers better than RNNs for NLP?", a: "(1) Parallel processing (all tokens at once, not sequential). (2) Direct attention between any two positions (no vanishing gradient across long range). (3) Scale: can train on massive data with GPUs efficiently." },
      { q: "What is the difference between BERT and GPT?", a: "BERT: Encoder only, bidirectional attention (sees full context). Trained with Masked Language Modeling. Good for understanding tasks (classification, NER). GPT: Decoder only, causal attention (left-to-right only). Trained with next-token prediction. Good for generation." },
      { q: "What is positional encoding?", a: "Attention is permutation-invariant (doesn't know token order). Positional encoding adds a position-dependent vector to each token embedding (using sin/cos waves of different frequencies). This gives the model information about where each token is in the sequence." },
    ],
    code: `import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """Multi-Head Self-Attention (simplified)"""
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        
        # Project input to Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, D = x.shape
        
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        scale = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn   = torch.softmax(scores, dim=-1)   # attention weights!
        
        # Weighted sum of values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), attn

# Test it
attn_layer = SelfAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
output, weights = attn_layer(x)
print(f"Input shape:      {x.shape}")
print(f"Output shape:     {output.shape}")
print(f"Attention shape:  {weights.shape}  (batch, heads, seq, seq)")`,
  },
  {
    id: 17, emoji: "🎨", title: "Generative Models", level: "Advanced",
    tagline: "Teaching machines to create",
    analogy: "🎭 Artist vs Critic: A GAN has two networks playing a game. The Generator is a forger trying to create fake paintings that look real. The Discriminator is an art expert trying to tell real from fake. They compete — the forger gets better because the critic gets better. Eventually, the fakes are indistinguishable from real.",
    explanation: "Generative models create new data. GANs: Generator creates fake samples, Discriminator classifies real/fake — trained adversarially. VAEs: Encode data to a latent distribution, sample from it, decode back — smooth latent space for interpolation. Diffusion Models: gradually add noise, then learn to reverse it (DALL-E, Stable Diffusion).",
    keyPoints: [
      "GAN: Generator vs Discriminator in minimax game",
      "Mode collapse: Generator produces limited variety — major GAN challenge",
      "VAE: Encoder → μ and σ (latent distribution) → Decoder. KL divergence regularizes latent space",
      "Diffusion: add Gaussian noise T steps, learn to denoise step by step",
      "Stable Diffusion, DALL-E 2/3, Midjourney all use Diffusion models",
    ],
    interviewQA: [
      { q: "What is mode collapse in GANs?", a: "The Generator finds a few 'safe' outputs that fool the Discriminator and only produces those (e.g., only generating one type of face). Solutions: Wasserstein GAN (gradient penalty), mini-batch discrimination, unrolled GANs." },
      { q: "How does a VAE differ from a regular autoencoder?", a: "Regular AE: encodes to a fixed vector (may not be smooth). VAE: encodes to a distribution (μ and σ). Sample z ~ N(μ, σ²). The KL divergence loss forces the latent space to be smooth and continuous — enabling meaningful interpolation and generation." },
      { q: "Why are Diffusion Models better than GANs for image generation?", a: "GANs: harder to train (mode collapse, training instability), no likelihood estimate. Diffusion: stable training, diversity in outputs, better image quality, and can be guided with text (text-to-image). Downside: slower inference (many denoising steps)." },
      { q: "What is the reparameterization trick in VAEs?", a: "We need to backprop through sampling, but sampling is non-differentiable. Trick: z = μ + σ × ε, where ε ~ N(0,1). Now z is differentiable w.r.t. μ and σ, so gradients can flow through the sampling operation." },
    ],
    code: `import torch
import torch.nn as nn
import torch.optim as optim

# Minimal VAE for 2D data
class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden=64):
        super().__init__()
        # Encoder
        self.enc_fc = nn.Linear(input_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_lv = nn.Linear(hidden, latent_dim)  # log variance
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
    
    def encode(self, x):
        h = torch.relu(self.enc_fc(x))
        return self.enc_mu(h), self.enc_lv(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)   # sample noise
        return mu + eps * std         # reparameterization trick!
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.dec(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

# Quick test
vae = VAE()
x = torch.randn(32, 2)
recon, mu, lv = vae(x)
loss = vae_loss(recon, x, mu, lv)
print(f"Input: {x.shape} | Recon: {recon.shape} | Loss: {loss.item():.2f}")`,
  },
  {
    id: 18, emoji: "💬", title: "NLP", level: "Intermediate",
    tagline: "Teaching computers to understand language",
    analogy: "🗺️ Words as Locations in Space: Word2Vec puts every word in a 'meaning space' where similar words are close together. 'King - Man + Woman ≈ Queen'. This magical property comes from training a neural network to predict surrounding words — the positions learned encode meaning.",
    explanation: "NLP transforms text into numerical representations. Pipeline: Tokenization → Embeddings → Model → Output. Key techniques: TF-IDF for traditional ML, Word2Vec/GloVe for word vectors, BERT/GPT for contextual embeddings. Modern LLMs (Large Language Models) unify all NLP tasks in one model.",
    keyPoints: [
      "Tokenization: split text into tokens (words, subwords, characters)",
      "TF-IDF: term frequency × inverse document frequency (classic feature)",
      "Word2Vec: learn word embeddings from context prediction",
      "BERT: contextual embeddings — same word has different vectors in different contexts",
      "Fine-tuning: take a pretrained BERT/GPT, train a small head for your task",
    ],
    interviewQA: [
      { q: "What is TF-IDF and when do you use it?", a: "TF = how often a word appears in a document. IDF = log(N/df) — penalizes words appearing in many documents (common words). TF-IDF = TF × IDF. Use for information retrieval and traditional ML text classification." },
      { q: "What is the difference between Word2Vec and BERT embeddings?", a: "Word2Vec: one fixed vector per word regardless of context ('bank' always has same vector). BERT: contextual embeddings — the vector for 'bank' changes based on surrounding words (financial bank vs riverbank)." },
      { q: "What is BPE (Byte-Pair Encoding)?", a: "A subword tokenization algorithm. Merges the most frequent pairs of characters/subwords iteratively. 'unhappiness' → 'un', 'happiness' or 'un', 'happy', 'ness'. Handles unknown words and reduces vocabulary size. Used by GPT/BERT." },
      { q: "How do you fine-tune a pretrained language model?", a: "Add a task-specific head (linear layer) on top of BERT/GPT. For classification: take [CLS] token embedding → linear → softmax. Freeze early layers (optional), train on your labeled data with a small learning rate (2e-5 to 5e-5)." },
    ],
    code: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sentiment analysis pipeline
texts = [
    "This movie was absolutely wonderful and inspiring",
    "Terrible film, complete waste of time",
    "I loved every moment of this masterpiece",
    "Boring and disappointing throughout",
    "Outstanding performance by the entire cast",
    "Dreadful script and poor direction",
    "A delightful and heartwarming experience",
    "Painfully slow and utterly forgettable",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=1000,
        sublinear_tf=True,    # apply log(tf) scaling
    )),
    ('clf', LogisticRegression(random_state=42))
])
pipeline.fit(texts, labels)

# Test on new text
new_reviews = ["Absolutely brilliant film!", "Complete garbage movie"]
preds = pipeline.predict(new_reviews)
probs = pipeline.predict_proba(new_reviews)

for text, pred, prob in zip(new_reviews, preds, probs):
    sentiment = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    print(f"{sentiment} ({prob[pred]:.0%} confidence): {text}")`,
  },
  {
    id: 19, emoji: "🎮", title: "Reinforcement Learning", level: "Advanced",
    tagline: "Learning through trial, error, and rewards",
    analogy: "🐕 Training a Dog: A dog performs an action (sits). You give it a treat (reward) or scold (negative reward). Over time, it learns which actions in which situations lead to treats. RL is this — an agent takes actions in an environment, receives rewards, and learns to maximize total reward over time.",
    explanation: "RL trains an Agent to make sequential decisions in an Environment. The Agent observes a State, takes an Action, receives a Reward, transitions to a new State. The goal is to learn a Policy (state → action mapping) that maximizes cumulative reward. Q-Learning, Policy Gradient, and Actor-Critic are the main algorithms.",
    keyPoints: [
      "Agent, Environment, State, Action, Reward, Policy (the 6 key terms)",
      "Q-value: Q(s,a) = expected cumulative reward from state s, taking action a",
      "Bellman Equation: Q(s,a) = r + γ × max Q(s', a')",
      "Exploration vs Exploitation: ε-greedy balances trying new things vs using known good actions",
      "Deep Q-Network (DQN): neural network approximates Q-values",
    ],
    interviewQA: [
      { q: "What is the Bellman Equation?", a: "Q(s,a) = r + γ × max_a' Q(s', a'). The value of being in state s and taking action a equals the immediate reward r plus the discounted value of the best action in the next state. γ (gamma) is discount factor (0=myopic, 1=far-sighted)." },
      { q: "What is the exploration vs exploitation dilemma?", a: "Exploitation: use current best-known policy (maximize immediate reward). Exploration: try new actions (may find better long-term rewards). ε-greedy: with probability ε explore randomly, with probability 1-ε exploit. Decay ε over time." },
      { q: "What is Policy Gradient vs Value-Based RL?", a: "Value-Based (Q-Learning, DQN): learn value function Q(s,a), derive policy from it. Best for discrete actions. Policy Gradient (REINFORCE, PPO): directly learn a stochastic policy π(a|s). Better for continuous action spaces." },
      { q: "What are the key challenges in RL?", a: "Sparse rewards (long time before reward), sample inefficiency (millions of interactions needed), non-stationarity (distribution shifts as policy changes), credit assignment (which action caused later reward), and catastrophic forgetting." },
    ],
    code: `import numpy as np

# Q-Learning on a simple GridWorld
# State: position (0-7), Action: Left(0), Right(1)
# Goal: reach state 7 (+10 reward), avoid state 3 (-5 reward)

Q = np.zeros((8, 2))  # Q-table: 8 states × 2 actions
rewards = {7: 10, 3: -5}

# Hyperparameters
alpha   = 0.1   # learning rate
gamma   = 0.9   # discount factor  
epsilon = 0.3   # exploration rate

for episode in range(1000):
    state = np.random.randint(1, 6)  # start anywhere in middle
    
    for step in range(20):
        # ε-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(2)   # explore!
        else:
            action = np.argmax(Q[state])    # exploit!
        
        # Transition
        next_state = max(0, state - 1) if action == 0 else min(7, state + 1)
        reward = rewards.get(next_state, -0.1)  # small cost per step
        
        # Q-Learning update (Bellman equation)
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
        if next_state in [3, 7]:  # terminal states
            break

print("Learned Q-Table:")
print("State | Left  | Right | Best Action")
print("-" * 40)
for s in range(8):
    best = "←" if np.argmax(Q[s]) == 0 else "→"
    tag = " (GOAL)" if s == 7 else " (TRAP)" if s == 3 else ""
    print(f"  {s}   | {Q[s,0]:5.2f} | {Q[s,1]:5.2f} | {best}{tag}")`,
  },
  {
    id: 20, emoji: "⚙️", title: "Feature Engineering", level: "Intermediate",
    tagline: "Garbage in, garbage out — transform data into gold",
    analogy: "🍳 Cooking Ingredients: A raw potato vs crispy fries — same ingredient, but preparation makes all the difference. Feature engineering is transforming raw data (dates, text, numbers) into forms that make patterns obvious to ML models. It's often what separates good models from great ones.",
    explanation: "Feature Engineering creates, transforms, and selects features to improve model performance. Includes: handling missing values, encoding categoricals, scaling numerics, creating interaction features, extracting components from dates/text, and selecting the most informative features using statistical tests or model-based importance.",
    keyPoints: [
      "Missing values: impute (mean/median/mode/model) or flag + impute",
      "Categorical encoding: one-hot, label, target, frequency, embedding",
      "Numeric scaling: StandardScaler (z-score), MinMaxScaler, RobustScaler",
      "Feature creation: interactions, polynomial, date components, aggregations",
      "Feature selection: correlation filter, mutual information, recursive elimination",
    ],
    interviewQA: [
      { q: "When should you use StandardScaler vs MinMaxScaler?", a: "StandardScaler (z-score): when data has outliers or when algorithm assumes normal distribution (SVM, LR, NN). MinMaxScaler [0,1]: when you need bounded output (e.g., image pixels), no severe outliers. RobustScaler: best with outliers (uses median/IQR)." },
      { q: "What is Target Encoding?", a: "Replace each category with the mean target value for that category. E.g., if users from 'NYC' convert at 0.35 rate, encode NYC as 0.35. Risk: target leakage and overfitting on small categories. Fix: use cross-validation or Bayesian smoothing." },
      { q: "How do you handle high-cardinality categorical features?", a: "Options: (1) Target encoding with smoothing, (2) Frequency/count encoding, (3) Group rare categories as 'Other', (4) Embedding layers (neural networks), (5) Feature hashing (random projection to fixed size)." },
      { q: "What is the difference between feature selection and dimensionality reduction?", a: "Feature selection: pick a subset of original features (interpretable). Methods: filter (correlation), wrapper (RFE), embedded (Lasso). Dimensionality reduction: transform to new features (PCA, t-SNE) — less interpretable but can capture complex relationships." },
    ],
    code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Raw messy data
df = pd.DataFrame({
    'age':    [25, np.nan, 35, 45, 28],
    'income': [50000, 80000, np.nan, 120000, 65000],
    'city':   ['NYC', 'LA', 'NYC', 'Chicago', None],
    'tenure': [2, 5, 1, 8, 3],
    'churn':  [0, 0, 1, 0, 1],
})

X = df.drop('churn', axis=1)
y = df['churn']

# Define feature groups
numeric_features = ['age', 'income', 'tenure']
categorical_features = ['city']

# Build preprocessing pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # fill NaN with median
    ('scaler',  StandardScaler())                   # standardize
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # fill NaN
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))    # encode
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

X_transformed = preprocessor.fit_transform(X)
print(f"Shape before: {X.shape}")
print(f"Shape after:  {X_transformed.shape}")
print("✅ NaN values handled, features scaled, categoricals encoded")`,
  },
  {
    id: 21, emoji: "📊", title: "Evaluation Metrics", level: "Intermediate",
    tagline: "Measuring what actually matters",
    analogy: "⚖️ Medical Test Accuracy: A test for a rare disease can be '99% accurate' by just saying 'no disease' to everyone (since 99% of people don't have it). That's useless! Precision, Recall, F1, AUC — these metrics tell you the true story about model performance, especially with imbalanced data.",
    explanation: "Different problems need different metrics. Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC. Regression: MAE, MSE, RMSE, R². Clustering: Silhouette, Davies-Bouldin. Choosing the right metric is as important as choosing the right model — they directly encode what 'success' means for your problem.",
    keyPoints: [
      "Accuracy: misleading for imbalanced datasets",
      "Precision: of all predicted positives, how many are truly positive?",
      "Recall: of all actual positives, how many did we catch?",
      "F1 Score: harmonic mean of Precision and Recall",
      "ROC-AUC: how well the model separates classes across all thresholds",
    ],
    interviewQA: [
      { q: "When should you use Precision vs Recall?", a: "High Precision: when false positives are costly (spam filter — don't mark real emails as spam). High Recall: when false negatives are costly (cancer detection — don't miss any cancer). F1: balance both when both matter equally." },
      { q: "What is the ROC curve and AUC?", a: "ROC plots TPR (Recall) vs FPR (1-Specificity) at all decision thresholds. AUC = area under curve. AUC=1: perfect model. AUC=0.5: random guess. AUC=0: perfectly wrong. Useful because it's threshold-independent." },
      { q: "When is RMSE better than MAE?", a: "RMSE penalizes large errors more (squared). Use RMSE when large errors are especially bad (e.g., financial predictions). Use MAE when you want equal treatment of all errors and have outliers (MAE is more robust to outliers)." },
      { q: "What is PR-AUC and when is it better than ROC-AUC?", a: "PR-AUC = area under Precision-Recall curve. Better than ROC-AUC for highly imbalanced datasets (e.g., fraud detection with 0.1% fraud rate). ROC-AUC can look falsely optimistic because it considers the large TN pool." },
    ],
    code: `from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, mean_absolute_error, 
    mean_squared_error, r2_score
)
import numpy as np

# --- Classification Metrics ---
y_true = np.array([0,0,0,0,0,0,0,1,1,1])  # imbalanced: 7 negative, 3 positive
y_pred = np.array([0,0,0,0,0,0,1,0,1,1])  # model predictions
y_prob = np.array([0.1,0.2,0.1,0.15,0.1,0.3,0.7,0.4,0.8,0.9])

print("=== Classification Metrics ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2%}  ← misleading with imbalance!")
print(f"Precision: {precision_score(y_true, y_pred):.2%}  (of predicted pos, how many right)")
print(f"Recall:    {recall_score(y_true, y_pred):.2%}  (of actual pos, how many caught)")
print(f"F1 Score:  {f1_score(y_true, y_pred):.2%}  (harmonic mean)")
print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.2%}")
print(f"PR-AUC:    {average_precision_score(y_true, y_prob):.2%}")

# --- Regression Metrics ---
print("\\n=== Regression Metrics ===")
y_true_r = np.array([3, 2, 5, 7, 4, 6])
y_pred_r = np.array([2.8, 2.2, 4.9, 7.5, 3.8, 6.1])

print(f"MAE:  {mean_absolute_error(y_true_r, y_pred_r):.3f}  (avg absolute error)")
print(f"RMSE: {mean_squared_error(y_true_r, y_pred_r, squared=False):.3f}  (punishes big errors)")
print(f"R²:   {r2_score(y_true_r, y_pred_r):.3f}  (1=perfect, 0=just guessing mean)")`,
  },
];

const LEVEL_CONFIG = {
  Beginner:     { color: "#10B981", bg: "#064E3B", label: "🟢 Beginner" },
  Intermediate: { color: "#F59E0B", bg: "#451A03", label: "🟡 Intermediate" },
  Advanced:     { color: "#F43F5E", bg: "#4C0519", label: "🔴 Advanced" },
};

export default function MLBoss() {
  const [selected, setSelected] = useState(TOPICS[0]);
  const [completed, setCompleted] = useState(new Set());
  const [search, setSearch] = useState("");
  const [activeTab, setActiveTab] = useState("learn");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const filtered = useMemo(() =>
    TOPICS.filter(t =>
      t.title.toLowerCase().includes(search.toLowerCase()) ||
      t.level.toLowerCase().includes(search.toLowerCase())
    ), [search]);

  const toggleComplete = (id) => {
    setCompleted(prev => {
      const n = new Set(prev);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  };

  const progress = Math.round((completed.size / TOPICS.length) * 100);
  const cfg = LEVEL_CONFIG[selected.level];

  return (
    <div style={{
      display: "flex", height: "100vh", background: "#0F172A",
      fontFamily: "'Inter', system-ui, sans-serif", color: "#E2E8F0", overflow: "hidden"
    }}>

      {/* ─── SIDEBAR ─── */}
      <div style={{
        width: sidebarOpen ? 280 : 0, minWidth: sidebarOpen ? 280 : 0,
        background: "#1E293B", borderRight: "1px solid #334155",
        display: "flex", flexDirection: "column", transition: "width 0.3s, min-width 0.3s",
        overflow: "hidden"
      }}>
        {/* Logo */}
        <div style={{ padding: "20px 16px 12px", borderBottom: "1px solid #334155" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <div style={{
              background: "linear-gradient(135deg, #6366F1, #8B5CF6)",
              borderRadius: 10, padding: "6px 10px", fontSize: 20
            }}>🧬</div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 16, color: "#F1F5F9" }}>ML Boss</div>
              <div style={{ fontSize: 11, color: "#94A3B8" }}>Complete ML/DL Guide</div>
            </div>
          </div>

          {/* Progress bar */}
          <div style={{ background: "#0F172A", borderRadius: 6, height: 6, overflow: "hidden" }}>
            <div style={{
              height: "100%", borderRadius: 6,
              background: "linear-gradient(90deg, #6366F1, #10B981)",
              width: `${progress}%`, transition: "width 0.5s"
            }} />
          </div>
          <div style={{ fontSize: 11, color: "#94A3B8", marginTop: 4 }}>
            {completed.size}/{TOPICS.length} completed · {progress}%
          </div>

          {/* Search */}
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="🔍 Search topics..."
            style={{
              marginTop: 10, width: "100%", background: "#0F172A",
              border: "1px solid #334155", borderRadius: 8,
              padding: "7px 10px", color: "#E2E8F0", fontSize: 13,
              outline: "none", boxSizing: "border-box"
            }}
          />
        </div>

        {/* Topic list */}
        <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
          {filtered.map(topic => {
            const isSelected = selected.id === topic.id;
            const isDone = completed.has(topic.id);
            const lc = LEVEL_CONFIG[topic.level];
            return (
              <div
                key={topic.id}
                onClick={() => { setSelected(topic); setActiveTab("learn"); }}
                style={{
                  padding: "9px 16px", cursor: "pointer", display: "flex",
                  alignItems: "center", gap: 10, transition: "background 0.15s",
                  background: isSelected ? "#0F172A" : "transparent",
                  borderLeft: isSelected ? `3px solid #6366F1` : "3px solid transparent"
                }}
              >
                <span style={{ fontSize: 18, flexShrink: 0 }}>{topic.emoji}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 13, fontWeight: isSelected ? 600 : 400,
                    color: isSelected ? "#F1F5F9" : "#CBD5E1",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap"
                  }}>
                    {topic.title}
                  </div>
                  <div style={{
                    fontSize: 10, color: lc.color, fontWeight: 500, marginTop: 1
                  }}>{topic.level}</div>
                </div>
                {isDone && (
                  <span style={{ color: "#10B981", fontSize: 14, flexShrink: 0 }}>✓</span>
                )}
              </div>
            );
          })}
        </div>

        {/* Level legend */}
        <div style={{ padding: "12px 16px", borderTop: "1px solid #334155", fontSize: 11 }}>
          {Object.entries(LEVEL_CONFIG).map(([level, lc]) => (
            <div key={level} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: lc.color }} />
              <span style={{ color: "#94A3B8" }}>{level}: {TOPICS.filter(t=>t.level===level).length} topics</span>
            </div>
          ))}
        </div>
      </div>

      {/* ─── MAIN CONTENT ─── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Top bar */}
        <div style={{
          background: "#1E293B", borderBottom: "1px solid #334155",
          padding: "12px 24px", display: "flex", alignItems: "center", gap: 16, flexShrink: 0
        }}>
          <button
            onClick={() => setSidebarOpen(p => !p)}
            style={{
              background: "#334155", border: "none", color: "#E2E8F0",
              borderRadius: 6, padding: "5px 10px", cursor: "pointer", fontSize: 14
            }}
          >
            {sidebarOpen ? "◀" : "▶"}
          </button>

          <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 24 }}>{selected.emoji}</span>
            <div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#F1F5F9" }}>{selected.title}</div>
              <div style={{ fontSize: 12, color: "#94A3B8" }}>{selected.tagline}</div>
            </div>
            <div style={{
              marginLeft: 8, background: cfg.bg, color: cfg.color,
              border: `1px solid ${cfg.color}40`, borderRadius: 20,
              padding: "3px 12px", fontSize: 12, fontWeight: 600
            }}>
              {cfg.label}
            </div>
          </div>

          <button
            onClick={() => toggleComplete(selected.id)}
            style={{
              background: completed.has(selected.id) ? "#064E3B" : "#1E3A5F",
              border: `1px solid ${completed.has(selected.id) ? "#10B981" : "#3B82F6"}`,
              color: completed.has(selected.id) ? "#10B981" : "#60A5FA",
              borderRadius: 8, padding: "7px 16px", cursor: "pointer",
              fontSize: 13, fontWeight: 600, transition: "all 0.2s"
            }}
          >
            {completed.has(selected.id) ? "✅ Completed" : "Mark Complete"}
          </button>
        </div>

        {/* Tab bar */}
        <div style={{
          background: "#1E293B", borderBottom: "1px solid #334155",
          display: "flex", padding: "0 24px", flexShrink: 0
        }}>
          {[
            { id: "learn",     label: "📚 Learn" },
            { id: "interview", label: "🎤 Interview Q&A" },
            { id: "code",      label: "💻 Code" },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                background: "none", border: "none",
                borderBottom: activeTab === tab.id ? "2px solid #6366F1" : "2px solid transparent",
                color: activeTab === tab.id ? "#A5B4FC" : "#94A3B8",
                padding: "12px 16px", cursor: "pointer", fontSize: 14,
                fontWeight: activeTab === tab.id ? 600 : 400, transition: "all 0.2s"
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content area */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>

          {/* LEARN TAB */}
          {activeTab === "learn" && (
            <div style={{ maxWidth: 800, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>

              {/* Analogy card */}
              <div style={{
                background: "linear-gradient(135deg, #1E3A5F22, #312E8122)",
                border: "1px solid #3B82F640",
                borderRadius: 16, padding: 24
              }}>
                <div style={{ fontSize: 12, color: "#60A5FA", fontWeight: 700, letterSpacing: 1, marginBottom: 10, textTransform: "uppercase" }}>
                  💡 Real-World Analogy
                </div>
                <div style={{ fontSize: 15, lineHeight: 1.7, color: "#CBD5E1" }}>
                  {selected.analogy}
                </div>
              </div>

              {/* Plain English explanation */}
              <div style={{
                background: "#1E293B", border: "1px solid #334155",
                borderRadius: 16, padding: 24
              }}>
                <div style={{ fontSize: 12, color: "#A78BFA", fontWeight: 700, letterSpacing: 1, marginBottom: 10, textTransform: "uppercase" }}>
                  🧠 How It Actually Works
                </div>
                <div style={{ fontSize: 15, lineHeight: 1.8, color: "#CBD5E1" }}>
                  {selected.explanation}
                </div>
              </div>

              {/* Key Points */}
              <div style={{
                background: "#1E293B", border: "1px solid #334155",
                borderRadius: 16, padding: 24
              }}>
                <div style={{ fontSize: 12, color: "#34D399", fontWeight: 700, letterSpacing: 1, marginBottom: 14, textTransform: "uppercase" }}>
                  ✅ Key Points to Remember
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {selected.keyPoints.map((point, i) => (
                    <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                      <div style={{
                        width: 24, height: 24, borderRadius: 6, flexShrink: 0,
                        background: "linear-gradient(135deg, #6366F1, #8B5CF6)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 11, fontWeight: 700, color: "white"
                      }}>
                        {i + 1}
                      </div>
                      <div style={{ fontSize: 14, lineHeight: 1.6, color: "#CBD5E1", paddingTop: 3 }}>
                        {point}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick nav */}
              <div style={{ display: "flex", gap: 12, marginTop: 4 }}>
                {selected.id > 1 && (
                  <button
                    onClick={() => { setSelected(TOPICS[selected.id - 2]); setActiveTab("learn"); }}
                    style={{
                      background: "#1E293B", border: "1px solid #334155",
                      color: "#94A3B8", borderRadius: 8, padding: "10px 18px",
                      cursor: "pointer", fontSize: 14, flex: 1
                    }}
                  >
                    ← {TOPICS[selected.id - 2].emoji} {TOPICS[selected.id - 2].title}
                  </button>
                )}
                {selected.id < TOPICS.length && (
                  <button
                    onClick={() => { setSelected(TOPICS[selected.id]); setActiveTab("learn"); }}
                    style={{
                      background: "linear-gradient(135deg, #4F46E5, #7C3AED)",
                      border: "none", color: "white", borderRadius: 8,
                      padding: "10px 18px", cursor: "pointer", fontSize: 14,
                      flex: 1, fontWeight: 600
                    }}
                  >
                    {TOPICS[selected.id].emoji} {TOPICS[selected.id].title} →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* INTERVIEW TAB */}
          {activeTab === "interview" && (
            <div style={{ maxWidth: 800, margin: "0 auto", display: "flex", flexDirection: "column", gap: 16 }}>
              <div style={{
                background: "#1E3A5F22", border: "1px solid #3B82F640",
                borderRadius: 12, padding: "14px 20px", fontSize: 14, color: "#93C5FD"
              }}>
                🎤 These are real interview questions with clear, concise answers. Study them well!
              </div>
              {selected.interviewQA.map((item, i) => (
                <InterviewCard key={i} index={i + 1} question={item.q} answer={item.a} />
              ))}
            </div>
          )}

          {/* CODE TAB */}
          {activeTab === "code" && (
            <div style={{ maxWidth: 860, margin: "0 auto" }}>
              <div style={{
                background: "#0D1117", border: "1px solid #334155",
                borderRadius: 16, overflow: "hidden"
              }}>
                <div style={{
                  background: "#161B22", padding: "12px 20px",
                  display: "flex", alignItems: "center", gap: 10,
                  borderBottom: "1px solid #30363D"
                }}>
                  <div style={{ display: "flex", gap: 6 }}>
                    {["#FF5F57", "#FFBD2E", "#28CA41"].map((c, i) => (
                      <div key={i} style={{ width: 12, height: 12, borderRadius: "50%", background: c }} />
                    ))}
                  </div>
                  <span style={{ fontSize: 13, color: "#8B949E", marginLeft: 6 }}>
                    {selected.title.toLowerCase().replace(/\s+/g, '_')}.py
                  </span>
                </div>
                <pre style={{
                  margin: 0, padding: "24px", overflowX: "auto",
                  fontSize: "13px", lineHeight: "1.7", color: "#E2E8F0",
                  fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"
                }}>
                  <CodeHighlight code={selected.code} />
                </pre>
              </div>
              <div style={{
                marginTop: 16, background: "#1E293B", border: "1px solid #334155",
                borderRadius: 12, padding: "14px 20px", fontSize: 13, color: "#94A3B8"
              }}>
                💡 <strong style={{ color: "#F1F5F9" }}>Tip:</strong> Copy this code and run it in a Jupyter Notebook or Google Colab. Install missing packages with <code style={{ background: "#0F172A", padding: "2px 6px", borderRadius: 4, color: "#A78BFA" }}>pip install sklearn xgboost lightgbm catboost torch</code>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function InterviewCard({ index, question, answer }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      background: "#1E293B", border: `1px solid ${open ? "#6366F1" : "#334155"}`,
      borderRadius: 12, overflow: "hidden", transition: "border-color 0.2s"
    }}>
      <div
        onClick={() => setOpen(p => !p)}
        style={{
          padding: "16px 20px", cursor: "pointer",
          display: "flex", alignItems: "flex-start", gap: 14
        }}
      >
        <div style={{
          width: 28, height: 28, borderRadius: 8, flexShrink: 0,
          background: "linear-gradient(135deg, #4F46E5, #7C3AED)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, fontWeight: 700, color: "white"
        }}>
          Q{index}
        </div>
        <div style={{ flex: 1, fontSize: 14, fontWeight: 500, color: "#F1F5F9", lineHeight: 1.6, paddingTop: 4 }}>
          {question}
        </div>
        <span style={{ color: "#6366F1", fontSize: 18, flexShrink: 0, paddingTop: 2 }}>
          {open ? "▲" : "▼"}
        </span>
      </div>
      {open && (
        <div style={{
          padding: "0 20px 18px 62px",
          background: "#0F172A", borderTop: "1px solid #334155"
        }}>
          <div style={{
            paddingTop: 14, fontSize: 14, lineHeight: 1.75, color: "#94A3B8"
          }}>
            <span style={{ color: "#10B981", fontWeight: 700, marginRight: 6 }}>A:</span>
            {answer}
          </div>
        </div>
      )}
    </div>
  );
}

function CodeHighlight({ code }) {
  const keywords = ['import', 'from', 'def', 'class', 'return', 'for', 'in', 'if', 'else', 'elif', 'True', 'False', 'None', 'and', 'or', 'not', 'with', 'as', 'try', 'except', 'print', 'super', 'self'];
  const lines = code.split('\n');
  return (
    <>
      {lines.map((line, i) => {
        const isComment = line.trimStart().startsWith('#');
        if (isComment) {
          return (
            <div key={i} style={{ color: "#6A737D" }}>{line}</div>
          );
        }
        // Colorize strings and numbers simply
        let colored = line
          .replace(/("""[\s\S]*?"""|'''[\s\S]*?'''|"[^"]*"|'[^']*')/g,
            '<STRING>$1</STRING>')
          .replace(/\b(\d+\.?\d*)\b/g, '<NUM>$1</NUM>');

        return (
          <div key={i}>
            <span dangerouslySetInnerHTML={{
              __html: colored
                .replace(/<STRING>(.*?)<\/STRING>/g, '<span style="color:#A5C261">$1</span>')
                .replace(/<NUM>(.*?)<\/NUM>/g, '<span style="color:#6897BB">$1</span>')
                .replace(new RegExp(`\\b(${keywords.join('|')})\\b`, 'g'),
                  '<span style="color:#CC7832;font-weight:600">$1</span>')
            }} />
          </div>
        );
      })}
    </>
  );
}
