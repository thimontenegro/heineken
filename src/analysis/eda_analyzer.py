"""
EDAAnalyzer: Classe para Análise Exploratória de Dados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


class EDAAnalyzer:
    """Classe para realizar Análise Exploratória de Dados (EDA)."""

    def __init__(self, df: pd.DataFrame, target: str = 'Color'):
        """
        Inicializa o EDAAnalyzer.

        Args:
            df: DataFrame com os dados
            target: Nome da coluna target
        """
        self.df = df.copy()
        self.target = target

    def plot_target_distribution(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plota a distribuição da variável target.

        Args:
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Histograma
        axes[0].hist(self.df[self.target].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#2E86AB')
        axes[0].axvline(self.df[self.target].mean(), color='red', linestyle='--', label=f'Média: {self.df[self.target].mean():.2f}')
        axes[0].axvline(self.df[self.target].median(), color='orange', linestyle='--', label=f'Mediana: {self.df[self.target].median():.2f}')
        axes[0].set_xlabel(self.target)
        axes[0].set_ylabel('Frequência')
        axes[0].set_title(f'Distribuição de {self.target}')
        axes[0].legend()

        # Boxplot
        axes[1].boxplot(self.df[self.target].dropna(), vert=True)
        axes[1].set_ylabel(self.target)
        axes[1].set_title(f'Boxplot de {self.target}')

        # QQ Plot
        from scipy import stats
        stats.probplot(self.df[self.target].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title(f'QQ Plot de {self.target}')

        plt.suptitle(f'📊 Análise da Distribuição do Target ({self.target})', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_feature_distributions(self, feature_cols: List[str],
                                    save_path: Optional[str] = None,
                                    show: bool = True) -> None:
        """
        Plota a distribuição de todas as features.

        Args:
            feature_cols: Lista de features para plotar
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        # Filtrar apenas features que existem
        available_features = [f for f in feature_cols if f in self.df.columns]
        n_features = len(available_features)

        # Calcular grid
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, col in enumerate(available_features):
            ax = axes[i]
            data = self.df[col].dropna()

            ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='#5DA5DA')
            ax.set_title(col, fontsize=9)
            ax.set_xlabel('')
            ax.tick_params(axis='both', labelsize=7)

            # Adicionar estatísticas
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1, alpha=0.7)

        # Esconder eixos vazios
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('📊 Distribuição das Features', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_correlation_matrix(self, feature_cols: List[str],
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> pd.DataFrame:
        """
        Plota a matriz de correlação.

        Args:
            feature_cols: Lista de features
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico

        Returns:
            DataFrame com a matriz de correlação
        """
        # Filtrar apenas features que existem e adicionar target
        available_features = [f for f in feature_cols if f in self.df.columns]
        cols_to_use = available_features + [self.target]

        # Calcular correlação
        corr_matrix = self.df[cols_to_use].corr()

        # Plot
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, square=True,
                    linewidths=0.5, annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.8})

        plt.title('📊 Matriz de Correlação', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return corr_matrix

    def plot_correlation_with_target(self, feature_cols: List[str],
                                      save_path: Optional[str] = None,
                                      show: bool = True) -> pd.Series:
        """
        Plota correlação das features com o target.

        Args:
            feature_cols: Lista de features
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico

        Returns:
            Series com correlações
        """
        # Filtrar apenas features que existem
        available_features = [f for f in feature_cols if f in self.df.columns]

        # Calcular correlação com target
        correlations = self.df[available_features].corrwith(self.df[self.target])
        correlations = correlations.sort_values(key=abs, ascending=True)

        # Plot
        plt.figure(figsize=(10, max(6, len(correlations) * 0.3)))
        colors = ['#E94F37' if x < 0 else '#2E86AB' for x in correlations]

        plt.barh(correlations.index, correlations.values, color=colors)
        plt.xlabel(f'Correlação com {self.target}')
        plt.title(f'📊 Correlação das Features com {self.target}', fontsize=14, pad=15)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Adicionar valores
        for i, (idx, val) in enumerate(correlations.items()):
            plt.text(val + 0.01 if val >= 0 else val - 0.01, i, f'{val:.3f}',
                     va='center', ha='left' if val >= 0 else 'right', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return correlations

    def plot_scatter_vs_target(self, feature_cols: List[str], n_top: int = 6,
                                save_path: Optional[str] = None,
                                show: bool = True) -> None:
        """
        Plota scatter plots das top features vs target.

        Args:
            feature_cols: Lista de features
            n_top: Número de top features para plotar
            save_path: Caminho para salvar o gráfico
            show: Se True, mostra o gráfico
        """
        # Calcular correlação com target
        available_features = [f for f in feature_cols if f in self.df.columns]
        correlations = self.df[available_features].corrwith(self.df[self.target])
        top_features = correlations.abs().sort_values(ascending=False).head(n_top).index.tolist()

        # Plot
        n_cols = 3
        n_rows = (n_top + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(top_features):
            ax = axes[i]
            ax.scatter(self.df[col], self.df[self.target], alpha=0.5, s=20, color='#2E86AB')

            # Linha de tendência
            z = np.polyfit(self.df[col].dropna(),
                          self.df.loc[self.df[col].notna(), self.target], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.df[col].min(), self.df[col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            corr = correlations[col]
            ax.set_xlabel(col)
            ax.set_ylabel(self.target)
            ax.set_title(f'{col}\n(corr: {corr:.3f})', fontsize=10)

        # Esconder eixos vazios
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f'📊 Top {n_top} Features vs {self.target}', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot salvo em: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def print_summary_stats(self, feature_cols: List[str]) -> pd.DataFrame:
        """
        Imprime estatísticas resumidas das features.

        Args:
            feature_cols: Lista de features

        Returns:
            DataFrame com estatísticas
        """
        available_features = [f for f in feature_cols if f in self.df.columns]
        stats = self.df[available_features + [self.target]].describe().T
        stats['missing'] = self.df[available_features + [self.target]].isnull().sum()
        stats['missing_pct'] = (stats['missing'] / len(self.df) * 100).round(2)
        stats['skewness'] = self.df[available_features + [self.target]].skew()

        print("\n📊 Estatísticas Resumidas:")
        print("="*80)
        print(stats[['count', 'mean', 'std', 'min', 'max', 'missing', 'missing_pct', 'skewness']].to_string())

        return stats

    def run_full_eda(self, feature_cols: List[str], suffix: str = '') -> None:
        """
        Executa EDA completa.

        Args:
            feature_cols: Lista de features
            suffix: Sufixo para nomes dos arquivos
        """
        print("\n" + "="*70)
        print("📊 ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
        print("="*70)

        # 1. Estatísticas resumidas
        self.print_summary_stats(feature_cols)

        # 2. Distribuição do target
        print("\n📊 Distribuição do Target:")
        self.plot_target_distribution(f'eda_target_distribution{suffix}.png', show=False)

        # 3. Distribuição das features
        print("\n📊 Distribuição das Features:")
        self.plot_feature_distributions(feature_cols, f'eda_feature_distributions{suffix}.png', show=False)

        # 4. Correlação com target
        print("\n📊 Correlação com Target:")
        correlations = self.plot_correlation_with_target(feature_cols, f'eda_correlation_target{suffix}.png', show=False)

        # Mostrar top correlações
        print("\n🎯 Top 10 Features mais correlacionadas com o target:")
        top_corr = correlations.abs().sort_values(ascending=False).head(10)
        for feat, corr in top_corr.items():
            sign = '+' if correlations[feat] > 0 else '-'
            print(f"   {feat:35s}: {sign}{corr:.4f}")

        # 5. Matriz de correlação
        print("\n📊 Matriz de Correlação:")
        self.plot_correlation_matrix(feature_cols, f'eda_correlation_matrix{suffix}.png', show=False)

        # 6. Scatter plots
        print("\n📊 Scatter Plots (Top 6 features vs Target):")
        self.plot_scatter_vs_target(feature_cols, n_top=6, save_path=f'eda_scatter_plots{suffix}.png', show=False)

        print("\n✅ EDA completa!")

