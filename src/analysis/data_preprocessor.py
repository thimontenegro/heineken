"""
DataPreprocessor: Classe para preparar dados para modelagem.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Optional


class DataPreprocessor:
    """Classe para preparar dados para modelagem."""

    ORIGINAL_FEATURES = [
        'Roast amount (kg)', '1st malt amount (kg)', '2nd malt amount (kg)',
        'MT - Temperature', 'MT - Time', 'WK - Temperature', 'WK - Steam', 'WK - Time',
        'Total cold wort', 'pH', 'Extract', 'WOC - Time', 'WHP Transfer - Time',
        'WHP Rest - Time', 'Roast color', '1st malt color', '2nd malt color'
    ]

    def __init__(self, df: pd.DataFrame, target: str = 'Color'):
        """
        Inicializa o DataPreprocessor.

        Args:
            df: DataFrame com os dados
            target: Nome da coluna target
        """
        self.df = df
        self.target = target
        self.feature_cols: List[str] = []
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_processed: Optional[np.ndarray] = None
        self.X_test_processed: Optional[np.ndarray] = None
        self.imputer: Optional[SimpleImputer] = None

    def set_features(self, engineered_features: List[str]) -> 'DataPreprocessor':
        """
        Define as features a serem usadas.

        Args:
            engineered_features: Lista de features adicionais criadas

        Returns:
            self para encadeamento
        """
        self.feature_cols = self.ORIGINAL_FEATURES + engineered_features
        return self

    def set_features_list(self, feature_cols: List[str]) -> 'DataPreprocessor':
        """
        Define a lista completa de features.

        Args:
            feature_cols: Lista completa de features

        Returns:
            self para encadeamento
        """
        self.feature_cols = feature_cols
        return self

    def clean_data(self, color_min: float = 0, color_max: float = 50) -> 'DataPreprocessor':
        """
        Limpa dados removendo valores inválidos.

        Args:
            color_min: Valor mínimo válido para cor
            color_max: Valor máximo válido para cor

        Returns:
            self para encadeamento
        """
        # Remover linhas sem target
        self.df = self.df.dropna(subset=[self.target]).copy()

        # Filtrar valores impossíveis de cor
        self.df = self.df[(self.df[self.target] >= color_min) & (self.df[self.target] <= color_max)]

        # Tratar valores negativos em features que devem ser positivas
        for col in self.feature_cols:
            if col in self.df.columns:
                keywords = ['amount', 'time', 'temperature', 'color', 'extract', 'wort', 'malt']
                if any(kw in col.lower() for kw in keywords):
                    mask = self.df[col] < 0
                    if mask.any():
                        self.df.loc[mask, col] = np.nan

        print(f"Dados limpos: {len(self.df)} registros")
        return self

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> 'DataPreprocessor':
        """
        Divide dados em treino e teste.

        Args:
            test_size: Proporção do conjunto de teste
            random_state: Seed para reprodutibilidade

        Returns:
            self para encadeamento
        """
        # Filtrar apenas features que existem no DataFrame
        available_features = [f for f in self.feature_cols if f in self.df.columns]

        X = self.df[available_features].copy()
        y = self.df[self.target].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Atualizar feature_cols para apenas as disponíveis
        self.feature_cols = available_features

        print(f"Split: Treino={len(self.X_train)} | Teste={len(self.X_test)}")
        return self

    def preprocess(self, strategy: str = 'median') -> 'DataPreprocessor':
        """
        Aplica pré-processamento (imputação).

        Args:
            strategy: Estratégia de imputação ('mean' ou 'median')

        Returns:
            self para encadeamento
        """
        self.imputer = SimpleImputer(strategy=strategy)
        self.X_train_processed = self.imputer.fit_transform(self.X_train)
        self.X_test_processed = self.imputer.transform(self.X_test)

        print(f"Pré-processamento: Imputação por {strategy}")
        return self

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Retorna dados processados."""
        return self.X_train_processed, self.X_test_processed, self.y_train, self.y_test

    def get_feature_cols(self) -> List[str]:
        """Retorna lista de features."""
        return self.feature_cols

    def print_target_stats(self) -> None:
        """Imprime estatísticas do target."""
        y = self.df[self.target]
        print(f"\nEstatísticas do Target ({self.target}):")
        print(f"   Média: {y.mean():.2f}")
        print(f"   Mediana: {y.median():.2f}")
        print(f"   Desvio Padrão: {y.std():.2f}")
        print(f"   Skewness: {y.skew():.4f}")

