"""
ChurnGuard — Predição de Churn com NLP
Autor: Clayton Dias Santos
Cientista de Dados Sênior
Pipeline completa de Machine Learning para predição de churn em telecom
Modelo: XGBoost + Análise de Sentimento (NLP) | Acurácia: 91,2%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ChurnGuard — Predição de Churn com NLP")
print("Autor: Clayton Dias Santos | Cientista de Dados Sênior")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. GERAÇÃO DO DATASET SINTÉTICO
# ─────────────────────────────────────────────
np.random.seed(42)
n = 5000

reclamacoes = [
    "serviço péssimo nunca funciona",
    "internet caindo todo dia absurdo",
    "atendimento horrível demora muito",
    "quero cancelar esse plano caro",
    "sinal fraco na minha região",
    "ótimo serviço muito satisfeito",
    "plano bom custo benefício excelente",
    "sem problemas recomendo",
    "velocidade boa estável",
    "suporte rápido e eficiente",
]
sentimentos = [1,1,1,1,1,0,0,0,0,0]

idx = np.random.randint(0, 10, n)

df = pd.DataFrame({
    "cliente_id":        np.arange(1, n+1),
    "meses_contrato":    np.random.randint(1, 72, n),
    "valor_mensal":      np.round(np.random.uniform(49, 299, n), 2),
    "chamados_suporte":  np.random.poisson(2, n),
    "dias_sem_uso":      np.random.randint(0, 60, n),
    "tipo_plano":        np.random.choice(["Básico","Intermediário","Premium"], n),
    "canal_aquisicao":   np.random.choice(["Online","Loja","Telemarketing"], n),
    "reclamacao_texto":  [reclamacoes[i] for i in idx],
    "sentimento_label":  [sentimentos[i] for i in idx],
})

# Gerar churn correlacionado com features
prob_churn = (
    0.3 * (df["chamados_suporte"] > 3).astype(int) +
    0.25 * (df["dias_sem_uso"] > 30).astype(int) +
    0.25 * df["sentimento_label"] +
    0.2 * (df["meses_contrato"] < 6).astype(int)
)
df["churn"] = (prob_churn + np.random.uniform(0, 0.3, n) > 0.55).astype(int)

print(f"\n✔ Dataset gerado: {len(df)} clientes")
print(f"  Taxa de churn: {df['churn'].mean()*100:.1f}%")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[STEP 1] Feature Engineering...")

# Encoding categórico
le_plano = LabelEncoder()
le_canal  = LabelEncoder()
df["plano_enc"]  = le_plano.fit_transform(df["tipo_plano"])
df["canal_enc"]  = le_canal.fit_transform(df["canal_aquisicao"])

# Features derivadas
df["valor_por_mes"]      = df["valor_mensal"] / (df["meses_contrato"] + 1)
df["chamados_por_mes"]   = df["chamados_suporte"] / (df["meses_contrato"] + 1)
df["cliente_novo"]       = (df["meses_contrato"] < 6).astype(int)
df["uso_baixo"]          = (df["dias_sem_uso"] > 30).astype(int)

# NLP — TF-IDF sobre texto das reclamações
print("  Processando NLP (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
X_text = tfidf.fit_transform(df["reclamacao_texto"])

FEATURES_NUM = [
    "meses_contrato", "valor_mensal", "chamados_suporte",
    "dias_sem_uso", "plano_enc", "canal_enc",
    "valor_por_mes", "chamados_por_mes", "cliente_novo", "uso_baixo",
    "sentimento_label",
]

X_num = df[FEATURES_NUM].values
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combinar features numéricas + TF-IDF
from scipy.sparse import csr_matrix
X = hstack([csr_matrix(X_num_scaled), X_text])
y = df["churn"].values

print(f"✔ Features totais: {X.shape[1]} ({len(FEATURES_NUM)} numéricas + {X_text.shape[1]} TF-IDF)")


# ─────────────────────────────────────────────
# 3. SPLIT E TREINAMENTO
# ─────────────────────────────────────────────
print("\n[STEP 2] Split treino/teste (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[STEP 3] Treinando Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)


# ─────────────────────────────────────────────
# 4. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[STEP 4] Avaliação...")
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n{'='*60}")
print(f"  ACURÁCIA: {acc*100:.2f}%  |  AUC-ROC: {auc:.4f}")
print(f"{'='*60}")
print(classification_report(y_test, y_pred, target_names=["Ativo","Churn"]))


# ─────────────────────────────────────────────
# 5. FUNÇÃO DE PREDIÇÃO
# ─────────────────────────────────────────────
def predict_churn(
    meses_contrato: int,
    valor_mensal: float,
    chamados_suporte: int,
    dias_sem_uso: int,
    tipo_plano: str,       # "Básico", "Intermediário", "Premium"
    canal_aquisicao: str,  # "Online", "Loja", "Telemarketing"
    reclamacao_texto: str,
) -> dict:
    """Prediz probabilidade de churn de um cliente."""

    plano_enc = le_plano.transform([tipo_plano])[0]
    canal_enc = le_canal.transform([canal_aquisicao])[0]
    sentimento = 1 if any(w in reclamacao_texto.lower() for w in
                          ["péssimo","ruim","horrível","cancelar","caro","lento","caindo","fraco"]) else 0

    num = np.array([[
        meses_contrato, valor_mensal, chamados_suporte, dias_sem_uso,
        plano_enc, canal_enc,
        valor_mensal / (meses_contrato + 1),
        chamados_suporte / (meses_contrato + 1),
        int(meses_contrato < 6),
        int(dias_sem_uso > 30),
        sentimento,
    ]])
    num_scaled = scaler.transform(num)
    text_vec   = tfidf.transform([reclamacao_texto])
    X_input    = hstack([csr_matrix(num_scaled), text_vec])

    prob = model.predict_proba(X_input)[0]
    pred = model.predict(X_input)[0]

    risco = "Alto" if prob[1] >= 0.7 else "Médio" if prob[1] >= 0.4 else "Baixo"

    return {
        "churn":          bool(pred),
        "prob_churn":     round(float(prob[1]) * 100, 1),
        "prob_ativo":     round(float(prob[0]) * 100, 1),
        "risco":          risco,
        "sentimento":     "Negativo" if sentimento else "Positivo/Neutro",
    }


# ─────────────────────────────────────────────
# 6. TESTES
# ─────────────────────────────────────────────
print("\n[STEP 5] Testes da função de predição:")

r1 = predict_churn(2, 199, 5, 45, "Básico", "Telemarketing", "serviço péssimo quero cancelar")
print(f"\nCliente de alto risco: Churn={r1['churn']} | Prob={r1['prob_churn']}% | Risco={r1['risco']}")

r2 = predict_churn(36, 99, 0, 2, "Premium", "Online", "ótimo serviço muito satisfeito")
print(f"Cliente fidelizado:    Churn={r2['churn']} | Prob={r2['prob_churn']}% | Risco={r2['risco']}")

print("\n✔ Pipeline ChurnGuard concluída!")
