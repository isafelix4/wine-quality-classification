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
│   ├── correlacao_features_derivadas.csv   # Correlação das features testadas (não incorporadas)
│   └── limiar_precisao_recall_rf.csv       # Precisão/Recall/F1 por limiar (Random Forest)
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
2. **EDA** — distribuições, correlações justificadas, outliers, consistência física dos valores
   (faixas plausíveis para vinho) e balanceamento das classes.
3. **Pré-processamento** — checagem de faltantes, avaliação de features derivadas (testadas e
   descartadas por não superarem as variáveis originais), padronização e split estratificado.
4. **Modelagem** — Regressão Logística, Random Forest, Gradient Boosting e SVM.
5. **Avaliação** — Acurácia, Precisão, Recall, F1, ROC-AUC, matriz de confusão, curvas ROC,
   leitura de custo de negócio (Falso Positivo x Falso Negativo) e ajuste de limiar do Random Forest.
6. **Interpretação** — importância das variáveis e implicações para a produção.

## 💡 Leitura de negócio: limiar de decisão

O Random Forest, no limiar padrão (0,5), identifica apenas metade dos vinhos de Alta Qualidade
(recall = 0,50). Reduzir esse limiar aumenta o recall às custas de precisão:

| Limiar | Precisão | Recall | F1 |
|---|---|---|---|
| 0,5 (padrão) | 0,800 | 0,500 | 0,615 |
| 0,4 | 0,750 | 0,656 | **0,700** |
| 0,3 | 0,579 | 0,688 | 0,629 |
| 0,2 | 0,469 | 0,719 | 0,568 |

O limiar 0,4 chama atenção: mantém precisão razoável (0,75) e já eleva o F1 para 0,70 — melhor
que o padrão 0,5. A escolha do ponto de operação ideal depende de qual erro custa mais caro para
a operação: deixar passar um vinho bom (Falso Negativo) ou rotular um vinho comum como especial
(Falso Positivo).

## 🛠️ Tecnologias

Python · pandas · numpy · scikit-learn · matplotlib · seaborn

## 👥 Equipe

- Isabele Cristina Felix Santos (rm371797)
- Thais Marcondes Narbonne (rm372559)
- Tiago Antônio dos Santos (rm373581)
