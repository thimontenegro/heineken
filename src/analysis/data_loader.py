"""
DataLoader: Classe para carregar e filtrar dados.
"""

import pandas as pd
from typing import Optional


class DataLoader:
    """Classe para carregar e filtrar dados."""

    def __init__(self, filepath: str):
        """
        Inicializa o DataLoader.

        Args:
            filepath: Caminho para o arquivo CSV
        """
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Carrega dados do CSV."""
        self.df = pd.read_csv(self.filepath)
        print(f" Dataset carregado: {len(self.df)} registros")
        return self.df

    def filter_product(self, product: str) -> pd.DataFrame:
        """
        Filtra por produto.

        Args:
            product: Código do produto (AMST ou HNK)

        Returns:
            DataFrame filtrado
        """
        if self.df is None:
            self.load()
        self.df = self.df[self.df['Product'] == product].copy()
        print(f" Filtrado para {product}: {len(self.df)} registros")
        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        """Retorna o DataFrame."""
        if self.df is None:
            self.load()
        return self.df

