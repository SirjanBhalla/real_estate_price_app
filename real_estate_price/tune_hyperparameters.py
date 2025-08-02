from sklearn.model_selection import GridSearchCV
from real_estate_price.pipeline import real_estate_pipeline
from real_estate_price.data_management import load_and_prepare_data
from real_estate_price import config
import joblib
import os

def tune_hyperparameters():
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()


    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [5, 8, 12, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['auto', 'sqrt', 'log2']
    }

  
    grid_search = GridSearchCV(
        estimator=real_estate_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2
    )

    
    grid_search.fit(X_train, y_train)

    
    print("Best hyperparameters found:")
    print(grid_search.best_params_)
    print(f"Best CV MAE score: {-grid_search.best_score_:.2f}")

    
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    best_model_path = config.TRAINED_MODEL_DIR / "real_estate_pipeline_best.joblib"
    joblib.dump(grid_search.best_estimator_, best_model_path)
    print(f"Best model saved to: {best_model_path}")

if __name__ == '__main__':
    tune_hyperparameters()
