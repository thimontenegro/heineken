"""
ModelExplainer: Classe para explicar modelos com SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import List, Optional


class ModelExplainer:
    """Classe para explicar modelos com SHAP."""

    def __init__(self, model, X_test: np.ndarray, feature_cols: List[str]):
        """
        Inicializa o ModelExplainer.

        Args:
            model: Modelo treinado
            X_test: Features de teste
            feature_cols: Lista de nomes das features
        """
        self.model = model
        self.X_test = X_test
        self.feature_cols = feature_cols
        self.shap_values: Optional[np.ndarray] = None
        self.shap_importance: Optional[pd.DataFrame] = None

    def compute_shap(self) -> 'ModelExplainer':
        """
        Calcula valores SHAP.

        Returns:
            self para encadeamento
        """
        print("\nCalculando valores SHAP...")
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(self.X_test)

        self.shap_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'SHAP Importance': np.abs(self.shap_values).mean(0)
        }).sort_values('SHAP Importance', ascending=False)

        print("SHAP calculado!")
        return self

    def get_top_features(self, n: int = 10) -> List[str]:
        """
        Retorna top N features por importância SHAP.

        Args:
            n: Número de features

        Returns:
            Lista de nomes das features
        """
        if self.shap_importance is None:
            self.compute_shap()
        return self.shap_importance.head(n)['Feature'].tolist()

    def get_shap_importance(self) -> pd.DataFrame:
        """
        Retorna DataFrame com importância SHAP.

        Returns:
            DataFrame com features e importância
        """
        if self.shap_importance is None:
            self.compute_shap()
        return self.shap_importance

    def print_top_features(self, n: int = 10) -> None:
        """
        Imprime top N features.

        Args:
            n: Número de features
        """
        if self.shap_importance is None:
            self.compute_shap()

        print(f"\n Top {n} Features (SHAP):")
        for i, row in self.shap_importance.head(n).iterrows():
            bar = "█" * int(row['SHAP Importance'] * 10 / self.shap_importance['SHAP Importance'].max())
            print(f"   {row['Feature']:30s} {row['SHAP Importance']:.4f} {bar}")

    def plot_summary(self, title: str = "SHAP Feature Importance",
                     save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plota summary SHAP.

        Args:
            title: Título do gráfico
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        if self.shap_values is None:
            self.compute_shap()

        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_cols)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(self.shap_values, X_test_df, plot_type="bar", show=False)
        plt.title(f'{title}', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_beeswarm(self, title: str = "SHAP Beeswarm",
                      save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plota beeswarm SHAP.

        Args:
            title: Título do gráfico
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        if self.shap_values is None:
            self.compute_shap()

        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_cols)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(self.shap_values, X_test_df, show=False)
        plt.title(f'{title}', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

