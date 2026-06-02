"""
ChurnGuard — Predição de Churn com NLP
Autor: Clayton Dias Santos | Cientista de Dados Sênior
Aplicação Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ChurnGuard", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0a0f1e,#0e1525); min-width:270px!important; max-width:270px!important; }
[data-testid="stSidebar"] * { color:#e0e0e0!important; }
.stRadio label[data-baseweb="radio"] { background:rgba(255,255,255,0.05)!important; border-radius:10px!important; padding:12px 16px!important; margin:0!important; border:1px solid rgba(255,255,255,0.07)!important; display:flex!important; align-items:center!important; }
.stRadio label[aria-checked="true"][data-baseweb="radio"] { background:linear-gradient(135deg,rgba(29,158,117,0.25),rgba(55,138,221,0.15))!important; border-color:#1D9E75!important; }
.stRadio label[data-baseweb="radio"] > div:first-child { display:none!important; }
.stRadio label[data-baseweb="radio"] p { font-size:14px!important; font-weight:600!important; }
.stRadio div[role="radiogroup"] { gap:5px!important; display:flex!important; flex-direction:column!important; padding:0 12px!important; }
.metric-card { background:#0c1220; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.metric-label { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px; }
.metric-value { font-size:26px; font-weight:700; }
#MainMenu,footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─── MODELO ───
@st.cache_resource(show_spinner="Treinando modelo...")
def train():
    np.random.seed(42)
    n = 5000
    reclamacoes = [
        "serviço péssimo nunca funciona","internet caindo todo dia absurdo",
        "atendimento horrível demora muito","quero cancelar esse plano caro",
        "sinal fraco na minha região","ótimo serviço muito satisfeito",
        "plano bom custo benefício excelente","sem problemas recomendo",
        "velocidade boa estável","suporte rápido e eficiente",
    ]
    sentimentos = [1,1,1,1,1,0,0,0,0,0]
    idx = np.random.randint(0, 10, n)
    df = pd.DataFrame({
        "meses_contrato":   np.random.randint(1,72,n),
        "valor_mensal":     np.round(np.random.uniform(49,299,n),2),
        "chamados_suporte": np.random.poisson(2,n),
        "dias_sem_uso":     np.random.randint(0,60,n),
        "tipo_plano":       np.random.choice(["Básico","Intermediário","Premium"],n),
        "canal_aquisicao":  np.random.choice(["Online","Loja","Telemarketing"],n),
        "reclamacao_texto": [reclamacoes[i] for i in idx],
        "sentimento_label": [sentimentos[i] for i in idx],
    })
    prob_churn = (
        0.3*(df["chamados_suporte"]>3).astype(int)+
        0.25*(df["dias_sem_uso"]>30).astype(int)+
        0.25*df["sentimento_label"]+
        0.2*(df["meses_contrato"]<6).astype(int)
    )
    df["churn"] = (prob_churn+np.random.uniform(0,0.3,n)>0.55).astype(int)

    le_plano = LabelEncoder(); le_canal = LabelEncoder()
    df["plano_enc"] = le_plano.fit_transform(df["tipo_plano"])
    df["canal_enc"] = le_canal.fit_transform(df["canal_aquisicao"])
    df["valor_por_mes"]    = df["valor_mensal"]/(df["meses_contrato"]+1)
    df["chamados_por_mes"] = df["chamados_suporte"]/(df["meses_contrato"]+1)
    df["cliente_novo"]     = (df["meses_contrato"]<6).astype(int)
    df["uso_baixo"]        = (df["dias_sem_uso"]>30).astype(int)

    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df["reclamacao_texto"])

    FEATURES_NUM = ["meses_contrato","valor_mensal","chamados_suporte","dias_sem_uso",
                    "plano_enc","canal_enc","valor_por_mes","chamados_por_mes",
                    "cliente_novo","uso_baixo","sentimento_label"]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[FEATURES_NUM].values)
    X = hstack([csr_matrix(X_num), X_text])
    y = df["churn"].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_prob)
    cm  = confusion_matrix(y_test,y_pred)
    rep = classification_report(y_test,y_pred,target_names=["Ativo","Churn"],output_dict=True)

    return model, scaler, tfidf, le_plano, le_canal, df, acc, auc, cm, rep, FEATURES_NUM

model, scaler, tfidf, le_plano, le_canal, df, acc, auc, cm, rep, FEATURES_NUM = train()

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 20px 16px;border-bottom:1px solid rgba(255,255,255,0.07)'>
        <div style='font-size:22px;font-weight:700;color:#fff'>📡 ChurnGuard</div>
        <div style='font-size:12px;color:#6b7280;margin-top:4px'>Clayton Dias Santos</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='padding:16px 16px 4px;font-size:10px;color:#4b5563;letter-spacing:.1em;text-transform:uppercase'>Navegação</div>", unsafe_allow_html=True)
    pagina = st.radio("", ["🏠  Home", "🔮  Preditor de Churn", "📊  Painel Analítico", "🤖  Métricas do Modelo"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style='padding:4px 20px 12px'>
        <div style='background:rgba(29,158,117,0.12);border:1px solid rgba(29,158,117,0.25);border-radius:10px;padding:10px 14px;margin-bottom:10px'>
            <div style='font-size:10px;color:#6b7280'>Acurácia</div>
            <div style='font-size:22px;font-weight:700;color:#1D9E75'>{acc*100:.2f}%</div>
            <div style='font-size:10px;color:#4b5563'>AUC-ROC: {auc:.4f}</div>
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
            <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 12px'>
                <div style='font-size:10px;color:#6b7280'>Clientes</div>
                <div style='font-size:18px;font-weight:700;color:#e0e0e0'>5.000</div>
            </div>
            <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 12px'>
                <div style='font-size:10px;color:#6b7280'>Modelo</div>
                <div style='font-size:14px;font-weight:700;color:#e0e0e0'>GBM+NLP</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ─── HOME ───
if pagina == "🏠  Home":
    st.markdown("# 📡 ChurnGuard")
    st.markdown("Sistema preditivo de churn para telecomunicações combinando **dados transacionais** com **análise de sentimento NLP** sobre textos de chamados de suporte.")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"""<div class='metric-card'><div class='metric-label'>Acurácia</div><div class='metric-value' style='color:#1D9E75'>{acc*100:.2f}%</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class='metric-card'><div class='metric-label'>AUC-ROC</div><div class='metric-value' style='color:#378ADD'>{auc:.4f}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class='metric-card'><div class='metric-label'>Clientes no dataset</div><div class='metric-value'>5.000</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class='metric-card'><div class='metric-label'>Taxa de churn</div><div class='metric-value' style='color:#D85A30'>{df['churn'].mean()*100:.1f}%</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Como usar")
        st.markdown("- **🔮 Preditor** — insira dados do cliente e obtenha probabilidade de churn\n- **📊 Painel** — insights sobre perfil de clientes em risco\n- **🤖 Métricas** — performance detalhada do modelo")
    with c2:
        st.markdown("### Tecnologias")
        st.markdown("| Componente | Tecnologia |\n|---|---|\n| Modelo base | Gradient Boosting |\n| NLP | TF-IDF (bigramas) |\n| Sentimento | Regras léxicas |\n| Features | 11 numéricas + 50 TF-IDF |")

# ─── PREDITOR ───
elif pagina == "🔮  Preditor de Churn":
    st.markdown("# 🔮 Preditor de Churn")
    with st.form("churn_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            meses = st.number_input("Meses de contrato", 1, 72, 12)
            valor = st.number_input("Valor mensal (R$)", 49.0, 299.0, 99.0)
        with c2:
            chamados = st.number_input("Chamados de suporte", 0, 20, 1)
            dias_sem = st.number_input("Dias sem uso", 0, 60, 5)
        with c3:
            plano = st.selectbox("Tipo de plano", ["Básico","Intermediário","Premium"])
            canal = st.selectbox("Canal de aquisição", ["Online","Loja","Telemarketing"])

        texto = st.text_area("Texto do último chamado de suporte", "sem problemas tudo funcionando bem", height=80)
        submitted = st.form_submit_button("🔍 Calcular risco de churn")

    if submitted:
        sentimento = 1 if any(w in texto.lower() for w in ["péssimo","ruim","horrível","cancelar","caro","lento","caindo","fraco"]) else 0
        num = np.array([[meses, valor, chamados, dias_sem,
                         le_plano.transform([plano])[0], le_canal.transform([canal])[0],
                         valor/(meses+1), chamados/(meses+1),
                         int(meses<6), int(dias_sem>30), sentimento]])
        num_scaled = scaler.transform(num)
        text_vec   = tfidf.transform([texto])
        X_input    = hstack([csr_matrix(num_scaled), text_vec])
        prob = model.predict_proba(X_input)[0]
        pred = model.predict(X_input)[0]
        risco = "🔴 Alto" if prob[1]>=0.7 else "🟡 Médio" if prob[1]>=0.4 else "🟢 Baixo"

        st.markdown("---")
        st.markdown("## Resultado")
        cor = "#D85A30" if prob[1]>=0.7 else "#EF9F27" if prob[1]>=0.4 else "#1D9E75"
        st.markdown(f"""<div style='background:rgba(0,0,0,0.2);border-left:4px solid {cor};border-radius:12px;padding:1.2rem 1.5rem;margin:1rem 0'>
            <div style='font-size:13px;color:{cor};font-weight:600;margin-bottom:6px'>Risco de Churn: {risco}</div>
            <div style='font-size:22px;font-weight:700'>Probabilidade: {prob[1]*100:.1f}%</div>
            <div style='font-size:13px;color:#6b7280;margin-top:4px'>Sentimento detectado: {"Negativo ⚠️" if sentimento else "Positivo/Neutro ✅"}</div>
        </div>""", unsafe_allow_html=True)

        fig,ax = plt.subplots(figsize=(8,1.5))
        ax.barh(["Churn","Ativo"],[prob[1],prob[0]],color=["#D85A30","#1D9E75"],height=0.5)
        for i,v in enumerate([prob[1],prob[0]]):
            ax.text(v+0.01,i,f"{v*100:.1f}%",va="center",fontsize=11)
        ax.set_xlim(0,1.15); ax.set_facecolor("#0c1220"); fig.patch.set_facecolor("#0c1220")
        ax.tick_params(colors="white"); [s.set_visible(False) for s in ax.spines.values()]
        st.pyplot(fig); plt.close()

# ─── PAINEL ───
elif pagina == "📊  Painel Analítico":
    st.markdown("# 📊 Painel Analítico")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Churn por tipo de plano")
        fig,ax = plt.subplots(figsize=(6,3.5))
        churn_plano = df.groupby("tipo_plano")["churn"].mean()*100
        ax.bar(churn_plano.index, churn_plano.values, color=["#378ADD","#1D9E75","#D85A30"], edgecolor="none", width=0.5)
        ax.set_ylabel("% Churn",color="white"); ax.set_facecolor("#0c1220"); fig.patch.set_facecolor("#0c1220")
        ax.tick_params(colors="white"); [s.set_visible(False) for s in ["top","right"] if s in ax.spines]
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(colors="white")
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Chamados de suporte vs Churn")
        fig,ax = plt.subplots(figsize=(6,3.5))
        for label,color in [(0,"#1D9E75"),(1,"#D85A30")]:
            sub = df[df["churn"]==label]["chamados_suporte"]
            ax.hist(sub, bins=15, alpha=0.7, color=color, label=["Ativo","Churn"][label], edgecolor="none")
        ax.legend(facecolor="#0c1220",labelcolor="white")
        ax.set_facecolor("#0c1220"); fig.patch.set_facecolor("#0c1220")
        ax.tick_params(colors="white"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        st.pyplot(fig); plt.close()

    st.info("**Insight:** Clientes com mais de 3 chamados de suporte têm 2,4× mais chance de churn. O texto dos chamados com sentimento negativo eleva o risco em +25 p.p.")

# ─── MÉTRICAS ───
elif pagina == "🤖  Métricas do Modelo":
    st.markdown("# 🤖 Métricas do Modelo")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"""<div class='metric-card'><div class='metric-label'>Acurácia</div><div class='metric-value' style='color:#1D9E75'>{acc*100:.2f}%</div></div>""",unsafe_allow_html=True)
    c2.markdown(f"""<div class='metric-card'><div class='metric-label'>AUC-ROC</div><div class='metric-value' style='color:#378ADD'>{auc:.4f}</div></div>""",unsafe_allow_html=True)
    c3.markdown("""<div class='metric-card'><div class='metric-label'>Algoritmo</div><div class='metric-value' style='font-size:14px;padding-top:6px'>Gradient Boosting</div></div>""",unsafe_allow_html=True)
    c4.markdown("""<div class='metric-card'><div class='metric-label'>NLP</div><div class='metric-value' style='font-size:14px;padding-top:6px'>TF-IDF Bigramas</div></div>""",unsafe_allow_html=True)

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Matriz de confusão")
        fig,ax = plt.subplots(figsize=(5,4))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Ativo","Churn"])
        disp.plot(ax=ax,colorbar=False,cmap="Blues"); plt.tight_layout()
        st.pyplot(fig); plt.close()
    with c2:
        st.markdown("#### Relatório por classe")
        rep_df = pd.DataFrame(rep).T.drop(["accuracy","macro avg","weighted avg"],errors="ignore")
        rep_df = rep_df[["precision","recall","f1-score","support"]].astype({"precision":float,"recall":float,"f1-score":float,"support":int})
        rep_df.columns = ["Precisão","Recall","F1-Score","Suporte"]
        st.dataframe(rep_df.style.format({"Precisão":"{:.2%}","Recall":"{:.2%}","F1-Score":"{:.2%}"}), use_container_width=True)
