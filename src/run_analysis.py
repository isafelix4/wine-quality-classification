"""
Tech Challenge - Classificacao da Qualidade de Vinhos
Pipeline completa: EDA -> pre-processamento -> modelagem -> avaliacao -> interpretacao
Gera todos os graficos e metricas na pasta results/ e imprime um resumo em JSON.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, ConfusionMatrixDisplay,
    precision_recall_curve
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

summary = {}

# ----------------------------------------------------------------------
# 1. COMPREENSAO DO PROBLEMA
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
df = pd.read_csv(os.path.join(BASE, "data", "WineQT.csv"))
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
if "id" in df.columns:
    df = df.drop(columns=["id"])

df["target"] = (df["quality"] >= 7).astype(int)
features = [c for c in df.columns if c not in ["quality", "target"]]

summary["n_amostras"] = int(df.shape[0])
summary["n_features"] = len(features)
summary["nulos"] = int(df.isnull().sum().sum())
summary["duplicados"] = int(df.duplicated().sum())
vc = df["target"].value_counts().sort_index()
summary["classe_baixa_media"] = int(vc[0])
summary["classe_alta"] = int(vc[1])
summary["pct_alta"] = round(100 * df["target"].mean(), 1)

# ----------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------
# 2.1 distribuicao da nota original
plt.figure()
sns.countplot(x="quality", data=df, hue="quality", palette="viridis", legend=False)
plt.title("Distribuicao da nota de qualidade (original)")
plt.xlabel("Qualidade"); plt.ylabel("Contagem")
plt.tight_layout(); plt.savefig(f"{RESULTS}/01_dist_quality.png", dpi=120); plt.close()

# 2.2 balanceamento das classes
plt.figure()
ax = sns.countplot(x="target", data=df, hue="target", palette="Set2", legend=False)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Baixa/Media (<7)", "Alta (>=7)"])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.title("Balanceamento das classes"); plt.xlabel(""); plt.ylabel("Contagem")
plt.tight_layout(); plt.savefig(f"{RESULTS}/02_balanceamento_classes.png", dpi=120); plt.close()

# 2.3 distribuicoes das variaveis
df[features].hist(bins=30, figsize=(15, 12), color="steelblue", edgecolor="black")
plt.suptitle("Distribuicao das variaveis fisico-quimicas", fontsize=14)
plt.tight_layout(); plt.savefig(f"{RESULTS}/03_distribuicoes_features.png", dpi=120); plt.close()

# 2.4 matriz de correlacao
plt.figure(figsize=(12, 9))
corr = df[features + ["quality"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": .8})
plt.title("Matriz de correlacao")
plt.tight_layout(); plt.savefig(f"{RESULTS}/04_matriz_correlacao.png", dpi=120); plt.close()

corr_quality = df[features].corrwith(df["quality"]).sort_values(ascending=False)
corr_quality.to_csv(f"{RESULTS}/correlacao_com_qualidade.csv")
summary["correlacao_com_qualidade"] = {k: round(v, 3) for k, v in corr_quality.items()}

# 2.5 comparacao de medias por classe
medias = df.groupby("target")[features].mean().T
medias.columns = ["Baixa/Media", "Alta"]
medias["diff_%"] = (100 * (medias["Alta"] - medias["Baixa/Media"]) / medias["Baixa/Media"]).round(1)
medias = medias.round(3)
medias.to_csv(f"{RESULTS}/medias_por_classe.csv")
summary["medias_por_classe"] = medias.to_dict(orient="index")

# 2.6 boxplots (outliers)
n = len(features); cols = 4; rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(y=df[col], ax=axes[i], color="salmon")
    axes[i].set_title(col)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.suptitle("Boxplots - deteccao de outliers", fontsize=14)
plt.tight_layout(); plt.savefig(f"{RESULTS}/05_boxplots_outliers.png", dpi=120); plt.close()

def contar_outliers_iqr(serie):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return int(((serie < q1 - 1.5*iqr) | (serie > q3 + 1.5*iqr)).sum())

outliers = {c: contar_outliers_iqr(df[c]) for c in features}
summary["outliers_por_variavel"] = dict(sorted(outliers.items(), key=lambda x: -x[1]))

# distribuicao de alcohol por classe (insight chave)
plt.figure()
sns.kdeplot(data=df, x="alcohol", hue="target", fill=True, common_norm=False,
            palette={0: "salmon", 1: "seagreen"})
plt.title("Teor alcoolico por classe (0=Baixa/Media, 1=Alta)")
plt.tight_layout(); plt.savefig(f"{RESULTS}/06_alcohol_por_classe.png", dpi=120); plt.close()

# 2.7 consistencia fisica dos valores (faixas plausiveis para vinho)
faixas_plausiveis = {
    "ph": (2.5, 4.5),
    "alcohol": (8.0, 15.0),
    "density": (0.98, 1.01),
    "volatile_acidity": (0.0, 2.0),
}
fora_da_faixa = {}
for col, (minimo, maximo) in faixas_plausiveis.items():
    fora_da_faixa[col] = int(((df[col] < minimo) | (df[col] > maximo)).sum())
summary["valores_fora_faixa_plausivel"] = fora_da_faixa

# ----------------------------------------------------------------------
# 3. PRE-PROCESSAMENTO
# ----------------------------------------------------------------------

# 3.1 feature engineering: testar variaveis derivadas antes de decidir usa-las
df_fe = df.copy()
df_fe["acidez_total"] = df_fe["fixed_acidity"] + df_fe["volatile_acidity"]
df_fe["alcool_densidade"] = df_fe["alcohol"] / df_fe["density"]
novas_features = ["acidez_total", "alcool_densidade"]
corr_novas = df_fe[novas_features].corrwith(df_fe["quality"])
corr_novas.to_csv(f"{RESULTS}/correlacao_features_derivadas.csv")
summary["correlacao_features_derivadas"] = {k: round(v, 3) for k, v in corr_novas.items()}
# Decisao: nenhuma das duas supera claramente a variavel original que a compoe
# (ver correlacao_com_qualidade.csv), entao NAO sao incorporadas ao modelo final.

if df[features].isnull().sum().sum() > 0:
    df[features] = df[features].fillna(df[features].median())

X = df[features]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
summary["treino"] = int(X_train.shape[0])
summary["teste"] = int(X_test.shape[0])

# ----------------------------------------------------------------------
# 4. MODELOS
# ----------------------------------------------------------------------
modelos = {
    "Regressao Logistica": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE))]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
}
for nome, m in modelos.items():
    m.fit(X_train, y_train)

# validacao cruzada (F1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}
for nome, m in modelos.items():
    s = cross_val_score(m, X_train, y_train, cv=cv, scoring="f1")
    cv_scores[nome] = [round(s.mean(), 3), round(s.std(), 3)]
summary["cv_f1"] = cv_scores

# ----------------------------------------------------------------------
# 5. AVALIACAO
# ----------------------------------------------------------------------
def avaliar(nome, modelo):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    return {
        "Modelo": nome,
        "Acuracia": round(accuracy_score(y_test, y_pred), 3),
        "Precisao": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1": round(f1_score(y_test, y_pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 3),
    }

resultados = [avaliar(n, m) for n, m in modelos.items()]
df_res = pd.DataFrame(resultados).set_index("Modelo").sort_values("F1", ascending=False)
df_res.to_csv(f"{RESULTS}/comparacao_modelos.csv")
summary["comparacao_modelos"] = df_res.reset_index().to_dict(orient="records")
melhor = df_res.index[0]
summary["melhor_modelo"] = melhor

# relatorios de classificacao
with open(f"{RESULTS}/relatorios_classificacao.txt", "w") as f:
    for nome, m in modelos.items():
        f.write("="*60 + "\n" + nome + "\n" + "="*60 + "\n")
        f.write(classification_report(y_test, m.predict(X_test),
                target_names=["Baixa/Media", "Alta"]))
        f.write("\n\n")

# matrizes de confusao
fig, axes = plt.subplots(1, len(modelos), figsize=(5*len(modelos), 4))
for ax, (nome, m) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, m.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["Baixa/Media", "Alta"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(nome)
plt.tight_layout(); plt.savefig(f"{RESULTS}/07_matrizes_confusao.png", dpi=120); plt.close()

# curvas ROC
plt.figure(figsize=(8, 6))
for nome, m in modelos.items():
    y_proba = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{nome} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Aleatorio")
plt.xlabel("Taxa de Falsos Positivos"); plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curvas ROC"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{RESULTS}/08_curvas_roc.png", dpi=120); plt.close()

# tabela comparativa como imagem
fig, ax = plt.subplots(figsize=(9, 2.2))
ax.axis("off")
t = ax.table(cellText=df_res.reset_index().values,
             colLabels=df_res.reset_index().columns, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
for j in range(len(df_res.columns)+1):
    t[0, j].set_facecolor("#40466e"); t[0, j].set_text_props(color="white")
plt.title("Comparacao de modelos (conjunto de teste)", pad=20)
plt.tight_layout(); plt.savefig(f"{RESULTS}/09_tabela_comparacao.png", dpi=120, bbox_inches="tight"); plt.close()

# 5.5 ajuste de limiar (threshold) do Random Forest: precisao/recall por limiar
rf_model = modelos["Random Forest"]
probs_rf = rf_model.predict_proba(X_test)[:, 1]
prec_curve, rec_curve, thr_curve = precision_recall_curve(y_test, probs_rf)

plt.figure()
plt.plot(thr_curve, prec_curve[:-1], label="Precisao")
plt.plot(thr_curve, rec_curve[:-1], label="Recall")
plt.axvline(0.5, color="gray", linestyle="--", label="Limiar padrao (0,5)")
plt.xlabel("Limiar de decisao"); plt.ylabel("Score")
plt.title("Precisao e Recall por limiar - Random Forest"); plt.legend()
plt.tight_layout(); plt.savefig(f"{RESULTS}/12_limiar_precisao_recall_rf.png", dpi=120); plt.close()

limiares_teste = [0.5, 0.4, 0.3, 0.2]
tabela_limiar = []
for limiar in limiares_teste:
    yp_l = (probs_rf >= limiar).astype(int)
    tabela_limiar.append({
        "Limiar": limiar,
        "Precisao": round(precision_score(y_test, yp_l), 3),
        "Recall": round(recall_score(y_test, yp_l), 3),
        "F1": round(f1_score(y_test, yp_l), 3),
    })
df_limiar = pd.DataFrame(tabela_limiar)
df_limiar.to_csv(f"{RESULTS}/limiar_precisao_recall_rf.csv", index=False)
summary["limiar_precisao_recall_rf"] = tabela_limiar

# ----------------------------------------------------------------------
# 6. INTERPRETACAO
# ----------------------------------------------------------------------
rf = modelos["Random Forest"].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
imp.to_csv(f"{RESULTS}/importancia_features_rf.csv")
summary["importancia_rf"] = {k: round(v, 3) for k, v in imp.items()}

plt.figure(figsize=(9, 6))
sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="viridis", legend=False)
plt.title("Importancia das variaveis - Random Forest"); plt.xlabel("Importancia")
plt.tight_layout(); plt.savefig(f"{RESULTS}/10_importancia_features_rf.png", dpi=120); plt.close()

logreg = modelos["Regressao Logistica"].named_steps["clf"]
coefs = pd.Series(logreg.coef_[0], index=features).sort_values()
summary["coeficientes_logreg"] = {k: round(v, 3) for k, v in coefs.sort_values(ascending=False).items()}

plt.figure(figsize=(9, 6))
colors = ["crimson" if c < 0 else "seagreen" for c in coefs.values]
sns.barplot(x=coefs.values, y=coefs.index, palette=colors, hue=coefs.index, legend=False)
plt.title("Coeficientes - Regressao Logistica"); plt.xlabel("Coeficiente (efeito na prob. de Alta Qualidade)")
plt.tight_layout(); plt.savefig(f"{RESULTS}/11_coeficientes_logreg.png", dpi=120); plt.close()

# salva resumo
with open(f"{RESULTS}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
