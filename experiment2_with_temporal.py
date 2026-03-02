#!/usr/bin/env python
"""
Experimento 2: COM variáveis temporais
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analysis import AmstelAnalysis
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔬 EXPERIMENTO 2: COM Variáveis Temporais")
print("="*70)

analysis = AmstelAnalysis(
    filepath='data/Heineken - Data Science Use Case 3.csv',
    product='AMST'
)

results = analysis.run(include_temporal=True, catboost_grid_search=True)
print("\n✅ Experimento 2 completo!")

