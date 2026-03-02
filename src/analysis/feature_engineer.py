"""
FeatureEngineer: Classe para criar novas features.
"""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """Classe para criar novas features."""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa o FeatureEngineer.

        Args:
            df: DataFrame com os dados
        """
        self.df = df.copy()
        self.temporal_features: List[str] = []
        self.process_features: List[str] = []

    def create_temporal_features(self) -> 'FeatureEngineer':
        """
        Cria features temporais a partir de Date/Time.

        Features criadas:
        - hour (0-23)
        - day_of_week (0=Segunda, 6=Domingo)
        - day_of_month (1-31)
        - month (1-12)
        - is_weekend (0 ou 1)
        - shift (0=manhã, 1=tarde, 2=noite)

        Returns:
            self para encadeamento
        """
        if 'Date/Time' not in self.df.columns:
            print("⚠Coluna Date/Time não encontrada")
            return self

        self.df['DateTime'] = pd.to_datetime(self.df['Date/Time'], format='%m/%d/%Y %H:%M')

        # Extrair componentes
        self.df['hour'] = self.df['DateTime'].dt.hour
        self.df['day_of_week'] = self.df['DateTime'].dt.dayofweek
        self.df['day_of_month'] = self.df['DateTime'].dt.day
        self.df['month'] = self.df['DateTime'].dt.month
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)

        # Turno
        self.df['shift'] = self.df['hour'].apply(
            lambda h: 0 if 6 <= h < 14 else (1 if 14 <= h < 22 else 2)
        )

        self.temporal_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'shift']
        print(f"Features temporais criadas: {self.temporal_features}")
        return self

    def create_process_features(self) -> 'FeatureEngineer':
        """
        Cria features de processo.

        Features criadas:
        - total_malt = 1st_malt + 2nd_malt + roast_amount
        - roast_ratio = roast_amount / total_malt
        - thermal_load = MT_Temp * MT_Time
        - wk_energy = WK_Temperature * WK_Time
        - extract_per_malt = Extract / total_malt

        Returns:
            self para encadeamento
        """
        # total_malt
        self.df['total_malt'] = (
            self.df['1st malt amount (kg)'].fillna(0) +
            self.df['2nd malt amount (kg)'].fillna(0) +
            self.df['Roast amount (kg)'].fillna(0)
        )

        # roast_ratio
        self.df['roast_ratio'] = self.df['Roast amount (kg)'] / self.df['total_malt']
        self.df['roast_ratio'] = self.df['roast_ratio'].replace([np.inf, -np.inf], np.nan)

        # thermal_load
        self.df['thermal_load'] = self.df['MT - Temperature'] * self.df['MT - Time']

        # wk_energy
        self.df['wk_energy'] = self.df['WK - Temperature'] * self.df['WK - Time']

        # extract_per_malt
        self.df['extract_per_malt'] = self.df['Extract'] / self.df['total_malt']
        self.df['extract_per_malt'] = self.df['extract_per_malt'].replace([np.inf, -np.inf], np.nan)

        self.process_features = ['total_malt', 'roast_ratio', 'thermal_load', 'wk_energy', 'extract_per_malt']
        print(f"Features de processo criadas: {self.process_features}")
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """Retorna o DataFrame com as novas features."""
        return self.df

    def get_temporal_features(self) -> List[str]:
        """Retorna lista de features temporais."""
        return self.temporal_features

    def get_process_features(self) -> List[str]:
        """Retorna lista de features de processo."""
        return self.process_features

    def get_engineered_features(self) -> List[str]:
        """Retorna lista de todas as features criadas."""
        return self.temporal_features + self.process_features

