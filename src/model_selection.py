"""Optional historical model comparison, adapted from secondary notebook cells 22/43/45/47.

Not run during portfolio packaging. This preserves the original preprocessing
order, including fitting the scaler before cross-validation. For new evaluation,
move preprocessing inside each fold and use temporal/customer-aware splits.
Dependencies are imported only when the optional experiment is requested.
"""
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from .pipeline import HOME, REGRESSION


def compare_and_tune(frame, n_trials=30):
    """Run the original comparison design; returned models stay in memory.

    The original Optuna sampler was unseeded. New trial paths/results need not
    match the stored trial log. No claim that 30 trials finished in the source.
    """
    import optuna
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    home = frame[frame['사업장_시도'].isin(HOME)]
    x = home[REGRESSION].copy()
    # This secondary notebook transformed five amount columns, unlike the
    # main notebook baseline, which transformed all seven features.
    for col in REGRESSION[:5]:
        x[col] = np.log1p(x[col])
    y = home['예대마진']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    constructors = {'RandomForest': RandomForestRegressor, 'XGBoost': XGBRegressor, 'LightGBM': LGBMRegressor}
    scores = {}
    for name, constructor in constructors.items():
        extra = {'verbose': -1} if name == 'LightGBM' else {}
        model = constructor(random_state=42, **extra)
        scores[name] = float(cross_val_score(model, xtrain, ytrain, cv=5, scoring='r2').mean())
    winner = max(scores, key=scores.get)

    def objective(trial):
        if winner == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            }
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
            }
            if winner == 'LightGBM':
                params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)
        extra = {'verbose': -1} if winner == 'LightGBM' else {}
        model = constructors[winner](**params, random_state=42, **extra)
        return float(cross_val_score(model, xtrain, ytrain, cv=5, scoring='r2').mean())

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    extra = {'verbose': -1} if winner == 'LightGBM' else {}
    model = constructors[winner](**study.best_params, random_state=42, **extra)
    model.fit(xtrain, ytrain)
    prediction = model.predict(xtest)
    return {'winner': winner, 'cv_r2': scores, 'best_params': study.best_params,
            'best_cv_r2': float(study.best_value), 'test_r2': float(r2_score(ytest, prediction)),
            'test_rmse': float(np.sqrt(mean_squared_error(ytest, prediction)))}
