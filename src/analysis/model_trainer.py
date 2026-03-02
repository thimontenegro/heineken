"""
ModelTrainer: Classe para treinar e avaliar modelos.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import product


@dataclass
class ModelMetrics:
    """Métricas de avaliação do modelo."""
    model_name: str
    cv_r2: float
    test_r2: float
    test_mae: float
    test_rmse: float
    best_params: Optional[Dict] = None


class ModelTrainer:
    """Classe para treinar e avaliar modelos."""

    def __init__(self, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: pd.Series, y_test: pd.Series,
                 n_folds: int = 5, random_state: int = 42):
        """
        Inicializa o ModelTrainer.

        Args:
            X_train: Features de treino
            X_test: Features de teste
            y_train: Target de treino
            y_test: Target de teste
            n_folds: Número de folds para cross-validation
            random_state: Seed para reprodutibilidade
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_folds = n_folds
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        self.results: List[ModelMetrics] = []
        self.trained_models: Dict[str, object] = {}

    def train_ridge(self, alpha: float = 1000) -> ModelMetrics:
        """Treina modelo Ridge (baseline linear)."""
        print("\n Treinando Ridge...")

        # Ridge precisa de scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        model = Ridge(alpha=alpha, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=self.kfold, scoring='r2')

        model.fit(X_train_scaled, self.y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = ModelMetrics(
            model_name='Ridge (Baseline)',
            cv_r2=cv_scores.mean(),
            test_r2=r2_score(self.y_test, y_pred),
            test_mae=mean_absolute_error(self.y_test, y_pred),
            test_rmse=np.sqrt(mean_squared_error(self.y_test, y_pred))
        )

        self.results.append(metrics)
        self.trained_models['Ridge'] = model
        print(f"   CV R²: {metrics.cv_r2:.4f} | Test R²: {metrics.test_r2:.4f}")
        return metrics

    def train_xgboost(self, use_early_stopping: bool = True, use_grid_search: bool = True) -> ModelMetrics:
        """Treina XGBoost com Grid Search."""
        print("\n Treinando XGBoost...")

        if use_grid_search:
            print("   Executando Grid Search...")
            # Grid de parâmetros
            param_grid = {
                'n_estimators': [600],
                'learning_rate': [0.03],
                'max_depth': [2, 3],
                'subsample': [0.7],
                'colsample_bytree': [0.7],
                'reg_lambda': [5, 10],
                'reg_alpha': [0.5, 1.0],
            }

            # Buscar melhores parâmetros
            best_score = -np.inf
            best_params = {}

            param_combinations = list(product(*param_grid.values()))
            for params in param_combinations:
                param_dict = dict(zip(param_grid.keys(), params))
                model = XGBRegressor(**param_dict, random_state=self.random_state, verbosity=0)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2')
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_params = param_dict

            print(f"   Melhores parâmetros: {best_params}")
        else:
            best_params = {
                'n_estimators': 600,
                'learning_rate': 0.03,
                'max_depth': 3,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_lambda': 5,
                'reg_alpha': 0.5
            }
            model = XGBRegressor(**best_params, random_state=self.random_state, verbosity=0)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2')
            best_score = cv_scores.mean()

        # Treinar modelo final com early stopping
        if use_early_stopping:
            X_tr, X_val, y_tr, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.15, random_state=self.random_state
            )
            model = XGBRegressor(**best_params, random_state=self.random_state,
                                 verbosity=0, early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model = XGBRegressor(**best_params, random_state=self.random_state, verbosity=0)
            model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        metrics = ModelMetrics(
            model_name='XGBoost',
            cv_r2=best_score,
            test_r2=r2_score(self.y_test, y_pred),
            test_mae=mean_absolute_error(self.y_test, y_pred),
            test_rmse=np.sqrt(mean_squared_error(self.y_test, y_pred)),
            best_params=best_params
        )

        self.results.append(metrics)
        self.trained_models['XGBoost'] = model
        print(f"   CV R²: {metrics.cv_r2:.4f} | Test R²: {metrics.test_r2:.4f}")
        return metrics

    def train_catboost(self, use_grid_search: bool = True, use_early_stopping: bool = True) -> ModelMetrics:
        """Treina CatBoost com Grid Search."""
        print("\n Treinando CatBoost com Grid Search...")

        if use_grid_search:
            # Grid de parâmetros otimizado (reduzido para performance)
            param_grid = {
                'iterations': [600, 800],
                'learning_rate': [0.02, 0.03, 0.05],
                'depth': [3, 4, 5],
                'l2_leaf_reg': [5, 10, 20],
                'subsample': [0.7, 0.8, 0.9],
                'random_strength': [1, 3]
            }

            # Buscar melhores parâmetros
            best_score = -np.inf
            best_params = {}

            # Para acelerar, fazer amostragem do grid
            param_combinations = list(product(*param_grid.values()))
            total_combinations = len(param_combinations)
            print(f"   Total de combinações: {total_combinations}")

            # Limitar a 30 combinações para performance
            if total_combinations > 30:
                np.random.seed(self.random_state)
                sample_indices = np.random.choice(total_combinations, size=30, replace=False)
                param_combinations = [param_combinations[i] for i in sample_indices]
                print(f"   Amostrando 30 combinações...")

            for i, params in enumerate(param_combinations):
                param_dict = dict(zip(param_grid.keys(), params))
                model = CatBoostRegressor(
                    **param_dict,
                    random_state=self.random_state,
                    verbose=0
                )
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2')
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_params = param_dict

            print(f"   Melhores parâmetros: {best_params}")
        else:
            best_params = {
                'iterations': 600,
                'learning_rate': 0.03,
                'depth': 3,
                'l2_leaf_reg': 5,
                'subsample': 0.7
            }

            model = CatBoostRegressor(**best_params, random_state=self.random_state, verbose=0)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=self.kfold, scoring='r2')
            best_score = cv_scores.mean()

        # Treinar modelo final
        if use_early_stopping:
            X_tr, X_val, y_tr, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.15, random_state=self.random_state
            )
            model = CatBoostRegressor(
                **best_params,
                random_state=self.random_state,
                verbose=0,
                early_stopping_rounds=50
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        else:
            model = CatBoostRegressor(**best_params, random_state=self.random_state, verbose=0)
            model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        metrics = ModelMetrics(
            model_name='CatBoost',
            cv_r2=best_score,
            test_r2=r2_score(self.y_test, y_pred),
            test_mae=mean_absolute_error(self.y_test, y_pred),
            test_rmse=np.sqrt(mean_squared_error(self.y_test, y_pred)),
            best_params=best_params
        )

        self.results.append(metrics)
        self.trained_models['CatBoost'] = model
        print(f"   CV R²: {metrics.cv_r2:.4f} | Test R²: {metrics.test_r2:.4f}")
        return metrics

    def train_lightgbm(self) -> ModelMetrics:
        """Treina LightGBM com regularização."""
        print("\n Treinando LightGBM...")

        model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=8,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5,
            reg_alpha=0.5,
            min_child_samples=20,
            random_state=self.random_state,
            verbose=-1
        )

        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=self.kfold, scoring='r2')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        metrics = ModelMetrics(
            model_name='LightGBM',
            cv_r2=cv_scores.mean(),
            test_r2=r2_score(self.y_test, y_pred),
            test_mae=mean_absolute_error(self.y_test, y_pred),
            test_rmse=np.sqrt(mean_squared_error(self.y_test, y_pred))
        )

        self.results.append(metrics)
        self.trained_models['LightGBM'] = model
        print(f"   CV R²: {metrics.cv_r2:.4f} | Test R²: {metrics.test_r2:.4f}")
        return metrics

    def train_gradient_boosting(self) -> ModelMetrics:
        """Treina Gradient Boosting com regularização."""
        print("\n Treinando GradientBoosting...")

        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.7,
            max_features=0.7,
            random_state=self.random_state,
            validation_fraction=0.15,
            n_iter_no_change=50,
            tol=1e-4
        )

        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=self.kfold, scoring='r2')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        metrics = ModelMetrics(
            model_name='GradientBoosting',
            cv_r2=cv_scores.mean(),
            test_r2=r2_score(self.y_test, y_pred),
            test_mae=mean_absolute_error(self.y_test, y_pred),
            test_rmse=np.sqrt(mean_squared_error(self.y_test, y_pred))
        )

        self.results.append(metrics)
        self.trained_models['GradientBoosting'] = model
        print(f"   CV R²: {metrics.cv_r2:.4f} | Test R²: {metrics.test_r2:.4f}")
        return metrics

    def train_all(self, catboost_grid_search: bool = True) -> List[ModelMetrics]:
        """
        Treina todos os modelos.

        Args:
            catboost_grid_search: Se True, usa Grid Search para CatBoost
        """
        self.train_ridge()
        self.train_xgboost()
        self.train_catboost(use_grid_search=catboost_grid_search)
        self.train_lightgbm()
        self.train_gradient_boosting()
        return self.results

    def get_results_df(self) -> pd.DataFrame:
        """Retorna DataFrame com resultados."""
        data = [{
            'Model': m.model_name,
            'CV R²': m.cv_r2,
            'Test R²': m.test_r2,
            'Test MAE': m.test_mae,
            'Test RMSE': m.test_rmse
        } for m in self.results]
        return pd.DataFrame(data).sort_values('Test R²', ascending=False)

    def get_best_model(self) -> Tuple[str, object]:
        """Retorna o melhor modelo."""
        results_df = self.get_results_df()
        best_name = results_df.iloc[0]['Model']

        # Mapear nome para modelo
        name_map = {
            'Ridge (Baseline)': 'Ridge',
            'XGBoost': 'XGBoost',
            'CatBoost': 'CatBoost',
            'LightGBM': 'LightGBM',
            'GradientBoosting': 'GradientBoosting'
        }

        model_key = name_map.get(best_name, 'XGBoost')
        return best_name, self.trained_models.get(model_key)

    def clear_results(self) -> None:
        """Limpa resultados anteriores."""
        self.results = []
        self.trained_models = {}

