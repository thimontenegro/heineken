#!/usr/bin/env python
"""
Experimento 1: SEM variáveis temporais
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analysis import AmstelAnalysis
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔬 EXPERIMENTO 1: SEM Variáveis Temporais")
print("="*70)

analysis = AmstelAnalysis(
    filepath='data/Heineken - Data Science Use Case 3.csv',
    product='AMST'
)

results = analysis.run(include_temporal=False, catboost_grid_search=True)
print("\n✅ Experimento 1 completo!")

