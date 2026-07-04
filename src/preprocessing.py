"""
Funções auxiliares de pré-processamento para o projeto de
classificação da qualidade de vinhos.

Uso no notebook:
    import sys; sys.path.append("../src")
    from preprocessing import carregar_dados, criar_target, separar_dados
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def carregar_dados(path):
    """Lê o CSV detectando o separador (; ou ,), padroniza nomes de colunas
    e remove a coluna 'id' (apenas identificador, não preditiva)."""
    for sep in [",", ";"]:
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] > 1:
            break
    else:
        raise ValueError("Não foi possível detectar o separador do CSV.")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def criar_target(df, limite=7):
    """Cria a variável alvo binária:
    1 = Alta Qualidade (quality >= limite), 0 = Baixa/Média."""
    df = df.copy()
    df["target"] = (df["quality"] >= limite).astype(int)
    return df


def tratar_faltantes(df, features):
    """Preenche valores faltantes com a mediana de cada coluna."""
    df = df.copy()
    if df[features].isnull().sum().sum() > 0:
        df[features] = df[features].fillna(df[features].median())
    return df


def contar_outliers_iqr(serie):
    """Conta outliers de uma série numérica pela regra do IQR."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((serie < low) | (serie > high)).sum())


def separar_dados(df, features, test_size=0.2):
    """Separa treino/teste de forma estratificada (preserva o desbalanceamento)."""
    X = df[features]
    y = df["target"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
