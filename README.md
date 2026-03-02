# Heineken Data Science Use Case

## 📋 Descrição

Projeto de análise preditiva para dados da Heineken, utilizando modelos de Machine Learning (XGBoost, CatBoost, Random Forest) com Grid Search para otimização de hiperparâmetros. O projeto compara o desempenho de modelos com e sem variáveis temporais para prever a variável alvo.

## 📁 Estrutura do Projeto


## ⚙️ Instalação

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd heineken

### 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate
### 3. Instale as dependências
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn shap
### Como executar
#### via jupyter
jupyter notebook notebooks/notebook.ipynb
### Via Script Python
import sys
sys.path.append('src/')
from analysis import AmstelAnalysis

analysis = AmstelAnalysis(
    filepath='data/Heineken - Data Science Use Case 3.csv',
    product='AMST'
)
analysis.run_experiments()
