"""Retrain the phishing model.

Run from the folder that contains url_features.py:
    python train_model.py dataset_phishing.csv
Creates phishing_pipeline.joblib (one file: features + model together).
"""
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from url_features import extract_matrix

path = sys.argv[1] if len(sys.argv) > 1 else 'dataset_phishing.csv'
df = pd.read_csv(path).drop_duplicates(subset='url').dropna()
X = df[['url']]
y = (df['status'] == 'phishing').astype(int)  # phishing = 1, legitimate = 0

# Split FIRST so the test set never influences feature building
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


def features():
    return ColumnTransformer([
        ('chars', TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                                  max_features=30000, sublinear_tf=True), 'url'),
        ('hand', Pipeline([('f', FunctionTransformer(extract_matrix)),
                           ('s', StandardScaler())]), 'url'),
    ])


candidates = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                            random_state=42),
}
try:  # optional: pip install xgboost
    from xgboost import XGBClassifier
    candidates['XGBoost'] = XGBClassifier(n_estimators=300, max_depth=8,
                                          learning_rate=0.1, n_jobs=-1,
                                          eval_metric='logloss')
except ImportError:
    print('xgboost not installed, skipping it.')

# Choose the best model using cross-validation on the TRAINING data only
scores = {}
for name, clf in candidates.items():
    pipe = Pipeline([('features', features()), ('clf', clf)])
    scores[name] = cross_val_score(pipe, X_train, y_train, cv=3,
                                   scoring='f1').mean()
    print(f'{name}: CV F1 = {scores[name]:.4f}')

best_name = max(scores, key=scores.get)
best = Pipeline([('features', features()), ('clf', candidates[best_name])])
best.fit(X_train, y_train)

# Evaluate ONCE on the untouched test set
pred = best.predict(X_test)
print(f'\nBest model: {best_name}')
print(classification_report(y_test, pred, target_names=['legitimate', 'phishing']))
print('Confusion matrix [[TN FP] [FN TP]]:')
print(confusion_matrix(y_test, pred))

joblib.dump(best, 'phishing_pipeline.joblib')
print('\nSaved phishing_pipeline.joblib')
