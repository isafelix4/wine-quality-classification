# 🍷 Wine Quality Classification — Tech Challenge (POSTECH Fase 2)

Projeto de Machine Learning para prever a **qualidade de vinhos tintos** a partir de suas
características físico-químicas. A nota de qualidade (0–10) é transformada em uma
**classificação binária**:

- **Alta Qualidade:** nota ≥ 7
- **Baixa/Média Qualidade:** nota < 7

## 📊 Principais resultados

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | 0,913 | 0,800 | 0,500 | **0,615** | **0,910** |
| Gradient Boosting | 0,900 | 0,667 | 0,562 | 0,610 | 0,879 |
| SVM (RBF) | 0,817 | 0,411 | 0,719 | 0,523 | 0,870 |
| Regressão Logística | 0,799 | 0,379 | 0,688 | 0,489 | 0,850 |

> **Modelo recomendado: Random Forest** — melhor F1 e ROC-AUC, robusto a outliers.

## 🔑 Insights da análise

- **~14%** das amostras são de Alta Qualidade (159 de 1.143): base **desbalanceada**.
- **Teor alcoólico** é o fator de maior influência positiva (correlação +0,48 com a qualidade).
- **Acidez volátil** é o principal fator negativo (correlação −0,41): quanto maior, pior o vinho.
- **Ácido cítrico** e **sulfatos** também contribuem positivamente.
- Por causa do desbalanceamento, priorizamos **F1 e ROC-AUC** em vez da acurácia.

## 📁 Estrutura do repositório

```
wine-quality-classification/
├── data/
│   └── WineQT.csv                          # Base de dados utilizada
├── notebooks/
│   └── wine_quality_classification.ipynb   # Análise e modelagem completas
├── src/
│   ├── preprocessing.py                    # Funções auxiliares
│   └── run_analysis.py                     # Pipeline ponta-a-ponta (gera results/)
├── results/                                # Gráficos e métricas gerados
├── requirements.txt
└── README.md
```

## 🚀 Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2a. Rodar o notebook (recomendado)
jupyter notebook notebooks/wine_quality_classification.ipynb

# 2b. OU rodar a pipeline completa via script (gera tudo em results/)
python src/run_analysis.py
```

## 🔬 Pipeline (6 etapas)

1. **Compreensão do problema** — definição da variável alvo binária (≥7).
2. **EDA** — distribuições, correlações justificadas, outliers e balanceamento das classes.
3. **Pré-processamento** — checagem de faltantes, padronização e split estratificado.
4. **Modelagem** — Regressão Logística, Random Forest, Gradient Boosting e SVM.
5. **Avaliação** — Acurácia, Precisão, Recall, F1, ROC-AUC, matriz de confusão e curvas ROC.
6. **Interpretação** — importância das variáveis e implicações para a produção.

## 🛠️ Tecnologias

Python · pandas · numpy · scikit-learn · matplotlib · seaborn

## 👥 Equipe

- (preencher com os integrantes do grupo)
