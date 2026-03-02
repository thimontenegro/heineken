import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analysis import AmstelAnalysis

import warnings
warnings.filterwarnings('ignore')


def main():
    """Executa os dois experimentos."""

    print("="*70)
    print("🍺 HEINEKEN DATA SCIENCE USE CASE")
    print("="*70)

    # Criar análise
    analysis = AmstelAnalysis(
        filepath='data/Heineken - Data Science Use Case 3.csv',
        product='AMST'
    )

    # Executar os dois experimentos
    analysis.run_experiments()

    return analysis


if __name__ == "__main__":
    analysis = main()

