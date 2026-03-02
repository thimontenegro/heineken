"""
AmstelAnalysis: Classe principal que orquestra toda a análise.
"""

import pandas as pd
from typing import Optional

from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .data_preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer
from .model_explainer import ModelExplainer
from .results_visualizer import ResultsVisualizer
from .eda_analyzer import EDAAnalyzer


class AmstelAnalysis:
    """Classe principal que orquestra toda a análise."""

    def __init__(self, filepath: str, product: str = 'AMST'):
        """
        Inicializa a análise.

        Args:
            filepath: Caminho para o arquivo CSV
            product: Código do produto (AMST ou HNK)
        """
        self.filepath = filepath
        self.product = product
        self.loader: Optional[DataLoader] = None
        self.engineer: Optional[FeatureEngineer] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.trainer: Optional[ModelTrainer] = None
        self.explainer: Optional[ModelExplainer] = None
        self.top10_trainer: Optional[ModelTrainer] = None
        self.eda: Optional[EDAAnalyzer] = None

        # Resultados dos experimentos
        self.results_without_temporal: Optional[pd.DataFrame] = None
        self.results_with_temporal: Optional[pd.DataFrame] = None

    def run(self, include_temporal: bool = True, catboost_grid_search: bool = True) -> None:
        """
        Executa o pipeline completo de análise.

        Args:
            include_temporal: Se True, inclui variáveis temporais
            catboost_grid_search: Se True, usa Grid Search para CatBoost
        """
        experiment_name = "COM" if include_temporal else "SEM"

        print("="*70)
        print(f"🍺 ANÁLISE AMSTEL - {experiment_name} Variáveis Temporais")
        print("="*70)

        # 1. Carregar dados
        print("\n FASE 1: Carregamento de Dados")
        print("-"*70)
        self.loader = DataLoader(self.filepath)
        self.loader.load()
        self.loader.filter_product(self.product)

        # 2. Feature Engineering
        print("\n FASE 2: Feature Engineering")
        print("-"*70)
        self.engineer = FeatureEngineer(self.loader.df)

        if include_temporal:
            self.engineer.create_temporal_features()

        self.engineer.create_process_features()

        # 3. Análise Exploratória de Dados (EDA)
        suffix = "_with_temporal" if include_temporal else "_without_temporal"
        print("\n📊 FASE 3: Análise Exploratória de Dados (EDA)")
        print("-"*70)
        self.eda = EDAAnalyzer(self.engineer.get_dataframe())
        feature_cols_for_eda = (
            DataPreprocessor.ORIGINAL_FEATURES +
            self.engineer.get_engineered_features()
        )
        self.eda.run_full_eda(feature_cols_for_eda, suffix)

        # 4. Preparar dados
        print("\n📊 FASE 4: Preparação dos Dados")
        print("-"*70)
        self.preprocessor = DataPreprocessor(self.engineer.get_dataframe())
        self.preprocessor.set_features(self.engineer.get_engineered_features())
        self.preprocessor.clean_data().split_data().preprocess()
        self.preprocessor.print_target_stats()

        print(f"\n Features utilizadas: {len(self.preprocessor.get_feature_cols())}")
        if include_temporal:
            print(f"   Temporais: {self.engineer.get_temporal_features()}")
        print(f"   Processo: {self.engineer.get_process_features()}")

        # 5. Treinar modelos
        print("\n📊 FASE 5: Treinamento de Modelos")
        print("-"*70)
        X_train, X_test, y_train, y_test = self.preprocessor.get_data()
        self.trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        self.trainer.train_all(catboost_grid_search=catboost_grid_search)

        # 6. Comparar modelos
        print("\n" + "="*70)
        print("📊 COMPARAÇÃO DOS MODELOS")
        print("="*70)
        results_df = self.trainer.get_results_df()
        print("\n" + results_df.to_string(index=False))

        ResultsVisualizer.plot_model_comparison(results_df, f'model_comparison{suffix}.png', show=False)

        # 7. SHAP para melhor modelo
        print("\n📊 FASE 6: Explicabilidade (SHAP)")
        print("-"*70)
        best_name, best_model = self.trainer.get_best_model()
        print(f"Melhor modelo: {best_name}")

        self.explainer = ModelExplainer(best_model, X_test, self.preprocessor.get_feature_cols())
        self.explainer.compute_shap()
        self.explainer.print_top_features(10)
        self.explainer.plot_summary(f"SHAP - {best_name} ({experiment_name} Temporal)",
                                    f'shap_best_model{suffix}.png', show=False)

        # 8. Modelo com Top 10 features
        print("\n" + "="*70)
        print("📊 FASE 7: Modelo com Top 10 Features")
        print("="*70)
        self._train_top10_models(suffix, catboost_grid_search)

        return results_df

    def run_experiments(self, catboost_grid_search: bool = True) -> None:
        """
        Executa dois experimentos: sem e com variáveis temporais.

        Args:
            catboost_grid_search: Se True, usa Grid Search para CatBoost
        """
        print("\n" + "="*70)
        print(" EXPERIMENTO 1: SEM Variáveis Temporais")
        print("="*70)
        self.results_without_temporal = self.run(include_temporal=False,
                                                  catboost_grid_search=catboost_grid_search)

        print("\n\n")
        print("="*70)
        print(" EXPERIMENTO 2: COM Variáveis Temporais")
        print("="*70)
        self.results_with_temporal = self.run(include_temporal=True,
                                               catboost_grid_search=catboost_grid_search)

        # Comparação final
        self._print_experiment_comparison()

    def _train_top10_models(self, suffix: str, catboost_grid_search: bool) -> None:
        """Treina modelos com top 10 features."""
        top_10_features = self.explainer.get_top_features(10)
        feature_cols = self.preprocessor.get_feature_cols()

        print("\n Top 10 Features selecionadas:")
        for i, feat in enumerate(top_10_features, 1):
            print(f"   {i:2d}. {feat}")

        # Obter índices das top 10 features
        top_10_indices = [feature_cols.index(f) for f in top_10_features]

        X_train, X_test, y_train, y_test = self.preprocessor.get_data()
        X_train_top10 = X_train[:, top_10_indices]
        X_test_top10 = X_test[:, top_10_indices]

        print(f"\nDataset reduzido: {X_train_top10.shape[1]} features")

        # Treinar modelos (sem grid search para Top 10, mais rápido)
        self.top10_trainer = ModelTrainer(X_train_top10, X_test_top10, y_train, y_test)
        self.top10_trainer.train_ridge()
        self.top10_trainer.train_xgboost(use_early_stopping=True, use_grid_search=False)
        self.top10_trainer.train_catboost(use_grid_search=False, use_early_stopping=True)

        # Comparar resultados
        print("\n" + "-"*70)
        print("COMPARAÇÃO: Todas Features vs Top 10")
        print("-"*70)

        all_results = self.trainer.get_results_df()
        top10_results = self.top10_trainer.get_results_df()
        top10_results['Model'] = top10_results['Model'].apply(lambda x: f"{x} (Top 10)")

        print("\n Resultados com Top 10 Features:")
        print(top10_results.to_string(index=False))

        # Plot comparativo
        ResultsVisualizer.plot_feature_comparison(all_results, top10_results,
                                                   f'comparison_all_vs_top10{suffix}.png', show=False)

        # SHAP para Top 10 (usar XGBoost ou CatBoost, não Ridge)
        # Obter modelo de árvore para SHAP
        if 'XGBoost' in self.top10_trainer.trained_models:
            best_model_t10 = self.top10_trainer.trained_models['XGBoost']
            best_name_t10 = 'XGBoost'
        elif 'CatBoost' in self.top10_trainer.trained_models:
            best_model_t10 = self.top10_trainer.trained_models['CatBoost']
            best_name_t10 = 'CatBoost'
        else:
            print("⚠Nenhum modelo de árvore disponível para SHAP")
            return

        explainer_t10 = ModelExplainer(best_model_t10, X_test_top10, top_10_features)
        explainer_t10.compute_shap()
        explainer_t10.plot_summary(f"SHAP - Top 10 ({best_name_t10})",
                                    f'shap_top10{suffix}.png', show=False)

    def _print_experiment_comparison(self) -> None:
        """Imprime comparação entre os dois experimentos."""
        print("\n\n")
        print("="*70)
        print("COMPARAÇÃO FINAL: SEM vs COM Variáveis Temporais")
        print("="*70)

        # Plot comparativo
        ResultsVisualizer.plot_experiment_comparison(
            self.results_without_temporal,
            self.results_with_temporal,
            'experiment_comparison.png',
            show=False
        )

        # Tabela comparativa
        print("\n Resultados SEM Variáveis Temporais:")
        print(self.results_without_temporal.to_string(index=False))

        print("\n Resultados COM Variáveis Temporais:")
        print(self.results_with_temporal.to_string(index=False))

        # Calcular melhoria
        print("\n MELHORIA com Variáveis Temporais:")
        models = ['Ridge', 'XGBoost', 'CatBoost', 'LightGBM', 'GradientBoosting']

        print(f"\n{'Modelo':<20} {'Test R² (Sem)':<15} {'Test R² (Com)':<15} {'Melhoria':<15}")
        print("-"*65)

        for m in models:
            mask_without = self.results_without_temporal['Model'].str.contains(m)
            mask_with = self.results_with_temporal['Model'].str.contains(m)

            if mask_without.any() and mask_with.any():
                r2_without = self.results_without_temporal[mask_without]['Test R²'].values[0]
                r2_with = self.results_with_temporal[mask_with]['Test R²'].values[0]

                if r2_without != 0:
                    improvement = ((r2_with - r2_without) / abs(r2_without)) * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"

                print(f"{m:<20} {r2_without:<15.4f} {r2_with:<15.4f} {improvement_str:<15}")

        print("\n" + "="*70)
        print(" EXPERIMENTOS COMPLETOS!")
        print("="*70)

        print("""
📁 Arquivos gerados:
   - model_comparison_without_temporal.png
   - model_comparison_with_temporal.png
   - shap_best_model_without_temporal.png
   - shap_best_model_with_temporal.png
   - comparison_all_vs_top10_without_temporal.png
   - comparison_all_vs_top10_with_temporal.png
   - experiment_comparison.png
""")

