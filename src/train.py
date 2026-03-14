# Hyperparameter grid search with cross-validation
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_model.save_model('checkpoints/xgb_best_v1.json')
print(f"Best AUC: {grid_search.best_score_:.4f} | Params: {grid_search.best_params_}")