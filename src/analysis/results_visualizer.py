"""
ResultsVisualizer: Classe para visualizar resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class ResultsVisualizer:
    """Classe para visualizar resultados."""

    @staticmethod
    def plot_model_comparison(results_df: pd.DataFrame,
                               save_path: Optional[str] = None,
                               show: bool = True) -> None:
        """
        Plota comparação de modelos.

        Args:
            results_df: DataFrame com resultados
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(results_df))
        width = 0.35

        # R² Score
        axes[0].bar(x - width/2, results_df['CV R²'], width, label='CV R²', color='#2E86AB')
        axes[0].bar(x + width/2, results_df['Test R²'], width, label='Test R²', color='#E94F37')
        axes[0].set_xlabel('Modelo')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('CV R² vs Test R²')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # MAE
        axes[1].bar(results_df['Model'], results_df['Test MAE'], color='#5DA5DA')
        axes[1].set_xlabel('Modelo')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Test MAE por Modelo')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_experiment_comparison(results_without_temporal: pd.DataFrame,
                                   results_with_temporal: pd.DataFrame,
                                   save_path: Optional[str] = None,
                                   show: bool = True) -> None:
        """
        Plota comparação entre experimentos com e sem variáveis temporais.

        Args:
            results_without_temporal: Resultados sem variáveis temporais
            results_with_temporal: Resultados com variáveis temporais
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Extrair modelos comuns
        models = ['Ridge', 'XGBoost', 'CatBoost', 'LightGBM', 'GradientBoosting']
        x = np.arange(len(models))
        width = 0.35

        # Extrair métricas
        cv_without = []
        cv_with = []
        test_without = []
        test_with = []

        for m in models:
            # Sem temporal
            mask_without = results_without_temporal['Model'].str.contains(m)
            if mask_without.any():
                cv_without.append(results_without_temporal[mask_without]['CV R²'].values[0])
                test_without.append(results_without_temporal[mask_without]['Test R²'].values[0])
            else:
                cv_without.append(0)
                test_without.append(0)

            # Com temporal
            mask_with = results_with_temporal['Model'].str.contains(m)
            if mask_with.any():
                cv_with.append(results_with_temporal[mask_with]['CV R²'].values[0])
                test_with.append(results_with_temporal[mask_with]['Test R²'].values[0])
            else:
                cv_with.append(0)
                test_with.append(0)

        # CV R²
        axes[0].bar(x - width/2, cv_without, width, label='Sem Temporal', color='#2E86AB')
        axes[0].bar(x + width/2, cv_with, width, label='Com Temporal', color='#E94F37')
        axes[0].set_ylabel('CV R²')
        axes[0].set_title('CV R²: Sem vs Com Variáveis Temporais')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Test R²
        axes[1].bar(x - width/2, test_without, width, label='Sem Temporal', color='#2E86AB')
        axes[1].bar(x + width/2, test_with, width, label='Com Temporal', color='#E94F37')
        axes[1].set_ylabel('Test R²')
        axes[1].set_title('Test R²: Sem vs Com Variáveis Temporais')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].legend()
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_feature_comparison(all_results: pd.DataFrame,
                                 top10_results: pd.DataFrame,
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> None:
        """
        Plota comparação entre todas features vs top 10.

        Args:
            all_results: Resultados com todas as features
            top10_results: Resultados com top 10 features
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models = ['Ridge', 'XGBoost', 'CatBoost']
        x = np.arange(len(models))
        width = 0.35

        # Extrair dados
        cv_all = []
        cv_top10 = []
        test_all = []
        test_top10 = []

        for m in models:
            mask_all = all_results['Model'].str.contains(m)
            mask_top10 = top10_results['Model'].str.contains(m)

            cv_all.append(all_results[mask_all]['CV R²'].values[0] if mask_all.any() else 0)
            cv_top10.append(top10_results[mask_top10]['CV R²'].values[0] if mask_top10.any() else 0)
            test_all.append(all_results[mask_all]['Test R²'].values[0] if mask_all.any() else 0)
            test_top10.append(top10_results[mask_top10]['Test R²'].values[0] if mask_top10.any() else 0)

        # CV R²
        axes[0].bar(x - width/2, cv_all, width, label='Todas Features', color='#2E86AB')
        axes[0].bar(x + width/2, cv_top10, width, label='Top 10 Features', color='#E94F37')
        axes[0].set_ylabel('CV R²')
        axes[0].set_title('CV R²: Todas vs Top 10')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].legend()
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Test R²
        axes[1].bar(x - width/2, test_all, width, label='Todas Features', color='#2E86AB')
        axes[1].bar(x + width/2, test_top10, width, label='Top 10 Features', color='#E94F37')
        axes[1].set_ylabel('Test R²')
        axes[1].set_title('Test R²: Todas vs Top 10')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

