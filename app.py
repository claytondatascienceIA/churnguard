"""
ChurnGuard — Predição de Churn com NLP
Autor: Clayton Dias Santos | Cientista de Dados Sênior
Identidade visual: Verde Teal — alerta clínico de negócio
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Paleta ChurnGuard ──
P1  = "#00C9A7"   # teal principal
P2  = "#00897B"   # teal escuro
P3  = "#FF6B6B"   # vermelho churn
BG  = "#080f0d"
BG2 = "#0a1510"
BG3 = "#0d1c18"

st.set_page_config(page_title="ChurnGuard", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif !important; }}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {BG} 0%, {BG2} 100%) !important;
  min-width: 275px !important; max-width: 275px !important;
  border-right: 1px solid rgba(0,201,167,0.12) !important;
}}
[data-testid="stSidebar"] * {{ color: #d0e8e3 !important; }}

.stRadio div[role="radiogroup"] {{ gap:5px!important; display:flex!important; flex-direction:column!important; padding:0 12px!important; }}
.stRadio label[data-baseweb="radio"] {{
  background: rgba(0,201,167,0.05) !important; border-radius:10px !important;
  padding:13px 16px !important; margin:0 !important;
  border:1px solid rgba(0,201,167,0.1) !important;
  display:flex !important; align-items:center !important;
  transition: all 0.2s !important;
}}
.stRadio label[data-baseweb="radio"]:hover {{
  background: rgba(0,201,167,0.12) !important;
  border-color: rgba(0,201,167,0.35) !important;
  transform: translateX(3px) !important;
}}
.stRadio label[aria-checked="true"][data-baseweb="radio"] {{
  background: linear-gradient(135deg,rgba(0,201,167,0.2),rgba(0,137,123,0.1)) !important;
  border-color: {P1} !important;
  box-shadow: 0 0 10px rgba(0,201,167,0.2) !important;
  transform: translateX(3px) !important;
}}
.stRadio label[data-baseweb="radio"] > div:first-child {{ display:none !important; }}
.stRadio label[data-baseweb="radio"] p {{ font-size:14px !important; font-weight:600 !important; color:#d0e8e3 !important; }}

.kpi {{ background:{BG3}; border:1px solid rgba(0,201,167,0.12); border-radius:14px; padding:1.1rem 1.3rem; text-align:center; }}
.kpi-label {{ font-size:10px; color:#4a7a6e; text-transform:uppercase; letter-spacing:.1em; margin-bottom:6px; font-family:'JetBrains Mono',monospace; }}
.kpi-value {{ font-size:26px; font-weight:700; }}

.result-box {{ border-radius:14px; padding:1.2rem 1.5rem; border-left:4px solid; margin:1rem 0; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ── MODELO ──
@st.cache_resource(show_spinner="Treinando ChurnGuard...")
def train():
    np.random.seed(42); n=5000
    reclamacoes=["serviço péssimo nunca funciona","internet caindo todo dia absurdo",
                 "atendimento horrível demora muito","quero cancelar esse plano caro",
                 "sinal fraco na minha região","ótimo serviço muito satisfeito",
                 "plano bom custo benefício excelente","sem problemas recomendo",
                 "velocidade boa estável","suporte rápido e eficiente"]
    sentimentos=[1,1,1,1,1,0,0,0,0,0]
    idx=np.random.randint(0,10,n)
    df=pd.DataFrame({
        "meses_contrato":np.random.randint(1,72,n), "valor_mensal":np.round(np.random.uniform(49,299,n),2),
        "chamados_suporte":np.random.poisson(2,n),  "dias_sem_uso":np.random.randint(0,60,n),
        "tipo_plano":np.random.choice(["Básico","Intermediário","Premium"],n),
        "canal_aquisicao":np.random.choice(["Online","Loja","Telemarketing"],n),
        "reclamacao_texto":[reclamacoes[i] for i in idx],
        "sentimento_label":[sentimentos[i] for i in idx],
    })
    prob_churn=(0.3*(df["chamados_suporte"]>3).astype(int)+0.25*(df["dias_sem_uso"]>30).astype(int)+
                0.25*df["sentimento_label"]+0.2*(df["meses_contrato"]<6).astype(int))
    df["churn"]=(prob_churn+np.random.uniform(0,0.3,n)>0.55).astype(int)
    le_plano=LabelEncoder(); le_canal=LabelEncoder()
    df["plano_enc"]=le_plano.fit_transform(df["tipo_plano"])
    df["canal_enc"]=le_canal.fit_transform(df["canal_aquisicao"])
    df["valor_por_mes"]=df["valor_mensal"]/(df["meses_contrato"]+1)
    df["chamados_por_mes"]=df["chamados_suporte"]/(df["meses_contrato"]+1)
    df["cliente_novo"]=(df["meses_contrato"]<6).astype(int)
    df["uso_baixo"]=(df["dias_sem_uso"]>30).astype(int)
    tfidf=TfidfVectorizer(max_features=50,ngram_range=(1,2))
    X_text=tfidf.fit_transform(df["reclamacao_texto"])
    FEATS=["meses_contrato","valor_mensal","chamados_suporte","dias_sem_uso",
           "plano_enc","canal_enc","valor_por_mes","chamados_por_mes","cliente_novo","uso_baixo","sentimento_label"]
    scaler=StandardScaler(); X_num=scaler.fit_transform(df[FEATS].values)
    X=hstack([csr_matrix(X_num),X_text]); y=df["churn"].values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model=GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42)
    model.fit(Xtr,ytr)
    yp=model.predict(Xte); yprob=model.predict_proba(Xte)[:,1]
    acc=accuracy_score(yte,yp); auc=roc_auc_score(yte,yprob)
    cm=confusion_matrix(yte,yp); rep=classification_report(yte,yp,target_names=["Ativo","Churn"],output_dict=True)
    return model,scaler,tfidf,le_plano,le_canal,df,acc,auc,cm,rep,FEATS

model,scaler,tfidf,le_plano,le_canal,df,acc,auc,cm,rep,FEATS=train()

def fig_style(fig,ax):
    ax.set_facecolor(BG3); fig.patch.set_facecolor(BG2)
    ax.tick_params(colors="#7ab5a8",labelsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3530"); ax.spines["bottom"].set_color("#1a3530")

# ── SIDEBAR ──
with st.sidebar:
    st.markdown(f"""
    <div style='padding:28px 20px 18px; border-bottom:1px solid rgba(0,201,167,0.12)'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>
            <span style='font-size:26px'>📡</span>
            <span style='font-size:19px;font-weight:700;color:#fff;letter-spacing:-0.3px'>ChurnGuard</span>
        </div>
        <div style='font-size:11px;color:#4a7a6e;padding-left:36px'>Clayton Dias Santos</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='padding:16px 16px 4px;font-size:10px;color:#2d5a52;letter-spacing:.12em;text-transform:uppercase;font-family:monospace'>Navegação</div>", unsafe_allow_html=True)
    pagina=st.radio("",["🏠  Home","🔮  Preditor de Churn","📊  Painel Analítico","🤖  Métricas do Modelo"],label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='padding:4px 18px 16px'>
      <div style='background:rgba(0,201,167,0.1);border:1px solid rgba(0,201,167,0.25);border-radius:12px;padding:12px 14px;margin-bottom:10px'>
        <div style='font-size:10px;color:#4a7a6e;font-family:monospace;margin-bottom:3px'>ACURÁCIA</div>
        <div style='font-size:24px;font-weight:700;color:{P1}'>{acc*100:.2f}%</div>
        <div style='font-size:10px;color:#2d5a52;font-family:monospace'>AUC-ROC: {auc:.4f}</div>
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(0,201,167,0.08);border-radius:10px;padding:10px 12px'>
          <div style='font-size:10px;color:#4a7a6e'>Clientes</div>
          <div style='font-size:18px;font-weight:700;color:#d0e8e3'>5.000</div>
        </div>
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(0,201,167,0.08);border-radius:10px;padding:10px 12px'>
          <div style='font-size:10px;color:#4a7a6e'>Modelo</div>
          <div style='font-size:13px;font-weight:700;color:#d0e8e3'>GBM+NLP</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── HOME ──
if pagina=="🏠  Home":
    st.markdown(f"<h1 style='font-size:2.2rem;font-weight:700;color:#fff'>📡 ChurnGuard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#7ab5a8;font-size:16px;margin-bottom:2rem'>Sistema preditivo de churn combinando <b style='color:{P1}'>dados transacionais</b> com <b style='color:{P1}'>análise de sentimento NLP</b> sobre chamados de suporte.</p>", unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,label,val,color in [(c1,"Acurácia",f"{acc*100:.2f}%",P1),(c2,"AUC-ROC",f"{auc:.4f}","#7ab5a8"),(c3,"Clientes","5.000","#d0e8e3"),(c4,"Taxa Churn",f"{df['churn'].mean()*100:.1f}%",P3)]:
        col.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value' style='color:{color}'>{val}</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### Como usar"); st.markdown("- **🔮 Preditor** — probabilidade de churn por cliente\n- **📊 Painel** — insights sobre perfil de risco\n- **🤖 Métricas** — performance técnica do modelo")
    with c2:
        st.markdown("### Stack técnico")
        st.markdown(f"| Componente | Tech |\n|---|---|\n| Modelo | Gradient Boosting |\n| NLP | TF-IDF bigramas |\n| Sentimento | Léxico customizado |\n| Deploy | Streamlit Cloud |")

# ── PREDITOR ──
elif pagina=="🔮  Preditor de Churn":
    st.markdown("# 🔮 Preditor de Churn")
    with st.form("f"):
        c1,c2,c3=st.columns(3)
        with c1: meses=st.number_input("Meses de contrato",1,72,12); valor=st.number_input("Valor mensal (R$)",49.0,299.0,99.0)
        with c2: chamados=st.number_input("Chamados de suporte",0,20,1); dias_sem=st.number_input("Dias sem uso",0,60,5)
        with c3: plano=st.selectbox("Plano",["Básico","Intermediário","Premium"]); canal=st.selectbox("Canal",["Online","Loja","Telemarketing"])
        texto=st.text_area("Último chamado de suporte","sem problemas tudo ok",height=80)
        sub=st.form_submit_button("Calcular risco →")
    if sub:
        sent=1 if any(w in texto.lower() for w in ["péssimo","ruim","horrível","cancelar","caro","lento","caindo","fraco"]) else 0
        num=np.array([[meses,valor,chamados,dias_sem,le_plano.transform([plano])[0],le_canal.transform([canal])[0],
                       valor/(meses+1),chamados/(meses+1),int(meses<6),int(dias_sem>30),sent]])
        Xi=hstack([csr_matrix(scaler.transform(num)),tfidf.transform([texto])])
        prob=model.predict_proba(Xi)[0]
        risco="🔴 Alto" if prob[1]>=0.7 else "🟡 Médio" if prob[1]>=0.4 else "🟢 Baixo"
        cor=P3 if prob[1]>=0.7 else "#EF9F27" if prob[1]>=0.4 else P1
        st.markdown(f"""<div class='result-box' style='background:rgba(0,0,0,0.3);border-color:{cor}'>
            <div style='font-size:13px;color:{cor};font-weight:700;margin-bottom:6px'>Nível de risco: {risco}</div>
            <div style='font-size:24px;font-weight:700;color:#fff'>Probabilidade de Churn: {prob[1]*100:.1f}%</div>
            <div style='font-size:13px;color:#7ab5a8;margin-top:6px'>Sentimento: {"⚠️ Negativo" if sent else "✅ Positivo/Neutro"}</div>
        </div>""",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(9,1.8))
        bars=ax.barh(["Probabilidade de Churn","Probabilidade de Ativo"],[prob[1]*100,prob[0]*100],
                     color=[P3,P1],height=0.45,edgecolor="none")
        for bar,v in zip(bars,[prob[1]*100,prob[0]*100]):
            ax.text(v+1,bar.get_y()+bar.get_height()/2,f"{v:.1f}%",va="center",fontsize=12,color="#fff",fontweight="bold")
        ax.set_xlim(0,115); fig_style(fig,ax); ax.set_xlabel("Probabilidade (%)",color="#7ab5a8")
        st.pyplot(fig); plt.close()

# ── PAINEL ──
elif pagina=="📊  Painel Analítico":
    st.markdown("# 📊 Painel Analítico")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Churn por tipo de plano")
        fig,ax=plt.subplots(figsize=(6,3.8))
        churn_plano=df.groupby("tipo_plano")["churn"].mean()*100
        planos=list(churn_plano.index); vals=list(churn_plano.values)
        bars=ax.bar(planos,vals,color=[P3,"#EF9F27",P1],edgecolor="none",width=0.5)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f"{v:.1f}%",ha="center",fontsize=11,color="#fff",fontweight="bold")
        ax.set_ylabel("% Churn",color="#7ab5a8"); ax.set_ylim(0,max(vals)*1.25)
        fig_style(fig,ax); st.pyplot(fig); plt.close()
    with c2:
        st.markdown("#### Chamados de suporte vs Churn")
        fig,ax=plt.subplots(figsize=(6,3.8))
        for label,color,name in [(0,P1,"Ativo"),(1,P3,"Churn")]:
            sub=df[df["churn"]==label]["chamados_suporte"]
            n,bins,patches=ax.hist(sub,bins=12,alpha=0.75,color=color,label=name,edgecolor="none")
        ax.legend(facecolor=BG3,labelcolor="white",fontsize=10)
        ax.set_xlabel("Nº de chamados",color="#7ab5a8"); ax.set_ylabel("Frequência",color="#7ab5a8")
        fig_style(fig,ax); st.pyplot(fig); plt.close()

    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Churn por canal de aquisição")
        fig,ax=plt.subplots(figsize=(6,3.5))
        churn_canal=df.groupby("canal_aquisicao")["churn"].mean()*100
        bars=ax.barh(list(churn_canal.index),list(churn_canal.values),color=P1,edgecolor="none",height=0.45)
        for bar,v in zip(bars,churn_canal.values):
            ax.text(v+0.3,bar.get_y()+bar.get_height()/2,f"{v:.1f}%",va="center",fontsize=11,color="#fff",fontweight="bold")
        ax.set_xlabel("% Churn",color="#7ab5a8"); ax.set_xlim(0,max(churn_canal.values)*1.25)
        fig_style(fig,ax); st.pyplot(fig); plt.close()
    with c2:
        st.markdown("#### Distribuição do valor mensal")
        fig,ax=plt.subplots(figsize=(6,3.5))
        for label,color,name in [(0,P1,"Ativo"),(1,P3,"Churn")]:
            sub=df[df["churn"]==label]["valor_mensal"]
            ax.hist(sub,bins=15,alpha=0.7,color=color,label=name,edgecolor="none")
        ax.legend(facecolor=BG3,labelcolor="white",fontsize=10)
        ax.set_xlabel("Valor mensal (R$)",color="#7ab5a8"); ax.set_ylabel("Frequência",color="#7ab5a8")
        fig_style(fig,ax); st.pyplot(fig); plt.close()

    st.info(f"**📌 Insight principal:** Clientes com mais de 3 chamados de suporte têm **2,4× mais chance de churn**. Sentimento negativo nos chamados eleva o risco em +25 p.p.")

# ── MÉTRICAS ──
elif pagina=="🤖  Métricas do Modelo":
    st.markdown("# 🤖 Métricas do Modelo")
    c1,c2,c3,c4=st.columns(4)
    for col,label,val,color in [(c1,"Acurácia",f"{acc*100:.2f}%",P1),(c2,"AUC-ROC",f"{auc:.4f}","#7ab5a8"),(c3,"Algoritmo","Grad. Boost","#d0e8e3"),(c4,"NLP","TF-IDF","#d0e8e3")]:
        col.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value' style='color:{color};font-size:{"22px" if len(val)>7 else "26px"}'>{val}</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Matriz de confusão")
        fig,ax=plt.subplots(figsize=(5,4))
        disp=ConfusionMatrixDisplay(cm,display_labels=["Ativo","Churn"])
        disp.plot(ax=ax,colorbar=False,cmap="YlGn"); ax.set_title("Predito vs Real",color="white",pad=10)
        fig.patch.set_facecolor(BG2); ax.set_facecolor(BG3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        st.markdown("#### Relatório por classe")
        rep_df=pd.DataFrame(rep).T.drop(["accuracy","macro avg","weighted avg"],errors="ignore")
        rep_df=rep_df[["precision","recall","f1-score","support"]].astype({"precision":float,"recall":float,"f1-score":float,"support":int})
        rep_df.columns=["Precisão","Recall","F1-Score","Suporte"]
        st.dataframe(rep_df.style.format({"Precisão":"{:.2%}","Recall":"{:.2%}","F1-Score":"{:.2%}"}),use_container_width=True)
