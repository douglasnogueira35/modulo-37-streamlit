import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import io
from fpdf import FPDF

st.title("🤖 AutoML Universal")

# Upload
df_file = st.file_uploader("Carregue seu arquivo (CSV, Excel, SQLite, Feather)", 
                           type=["csv","xlsx","xls","db","sqlite","ftr"])

if df_file is not None:
    # Detectar formato
    if str(df_file.name).endswith(".csv"):
        df = pd.read_csv(df_file)
    elif str(df_file.name).endswith((".xlsx",".xls")):
        df = pd.read_excel(df_file)
    elif str(df_file.name).endswith((".db",".sqlite")):
        import sqlite3
        conn = sqlite3.connect(df_file.name)
        df = pd.read_sql("SELECT * FROM tabela", conn)  # ajuste conforme sua tabela
    elif str(df_file.name).endswith(".ftr"):
        df = pd.read_feather(df_file)

    st.success(f"✅ Arquivo carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")

    # ================================
    # Sidebar
    # ================================
    st.sidebar.header("⚙️ Configurações")

    alvo = st.sidebar.selectbox("🎯 Selecione a coluna alvo", df.columns)

    # Slider para quantidade de linhas
    num_linhas = st.sidebar.slider(
        "📊 Quantidade de linhas a usar",
        min_value=100,
        max_value=len(df),
        value=min(1000, len(df)),
        step=100
    )

    df_final = df.head(num_linhas)

    st.write(f"📊 Dados selecionados (primeiras {num_linhas} linhas):")
    st.dataframe(df_final)

    # ================================
    # Pré-processamento
    # ================================
    y = df_final[alvo]
    X = df_final.drop(columns=[alvo])

    if "data_ref" in X.columns:
        X["data_ref"] = pd.to_datetime(X["data_ref"], errors="coerce").astype(int) // 10**9

    X = pd.get_dummies(X, drop_first=True).fillna(0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Detectar problema
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        problema = "regressao"
        y = pd.to_numeric(y, errors="coerce").fillna(y.mean())
    else:
        problema = "classificacao"
        y = y.astype("category").cat.codes

    st.info(f"🔎 Detectado problema de **{problema.upper()}**")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelos
    if problema == "classificacao":
        modelos = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        }
    else:
        modelos = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor()
        }

    resultados = {}
    variaveis_importancia = {}

    # Treinamento
    for nome, modelo in modelos.items():
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            if problema == "classificacao":
                resultados[nome] = {
                    "Acurácia": accuracy_score(y_test, y_pred),
                    "F1-Score": f1_score(y_test, y_pred, average="weighted"),
                    "Precisão": precision_score(y_test, y_pred, average="weighted"),
                    "Recall": recall_score(y_test, y_pred, average="weighted")
                }
            else:
                resultados[nome] = {
                    "R²": r2_score(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAE": mean_absolute_error(y_test, y_pred)
                }

            # Importância das variáveis
            if hasattr(modelo, "feature_importances_"):
                imp = modelo.feature_importances_
            elif hasattr(modelo, "coef_"):
                coef = modelo.coef_[0] if len(modelo.coef_.shape) > 1 else modelo.coef_
                imp = np.abs(coef)
            else:
                imp = np.zeros(X.shape[1])

            df_imp = pd.DataFrame({"Variável": X.columns, "Importância": imp})
            variaveis_importancia[nome] = df_imp.sort_values("Importância", ascending=False)

        except Exception as e:
            resultados[nome] = f"Erro: {e}"

    # ================================
    # Relatório Final na Tela
    # ================================
    st.subheader("📑 Relatório Final")
    df_resultados = pd.DataFrame(resultados).T
    st.dataframe(df_resultados.style.highlight_max(axis=0, color="lightgreen"))

    st.subheader("📌 Importância das Variáveis")
    for modelo, df_imp in variaveis_importancia.items():
        st.markdown(f"**{modelo}**")
        st.dataframe(df_imp.head(10).style.background_gradient(cmap="Blues"))
# ================================
# Aba de Explicações e Insights
# ================================
st.subheader("📖 Explicações e Insights")

tab1, tab2 = st.tabs(["Por que este modelo?", "Insights de Negócio"])

with tab1:
    st.write("### Justificativa da escolha dos modelos")
    if problema == "classificacao":
        st.markdown("""
        - **Logistic Regression**: modelo simples e interpretável, útil para entender relações lineares entre variáveis.
        - **Random Forest Classifier**: combina várias árvores de decisão, robusto contra overfitting e captura relações não lineares.
        - **XGBClassifier**: algoritmo de boosting altamente eficiente, ótimo para dados complexos e competições de machine learning.
        """)
    else:
        st.markdown("""
        - **Linear Regression**: modelo básico e interpretável, bom para relações lineares.
        - **Random Forest Regressor**: captura interações complexas entre variáveis e é robusto contra ruído.
        - **XGBRegressor**: modelo de boosting que otimiza erros residuais, excelente para alta performance em regressão.
        """)

with tab2:
    st.write("### Insights de Negócio")
    if isinstance(df_resultados, pd.DataFrame):
        melhor_modelo = max(
            resultados.items(),
            key=lambda x: x[1][list(x[1].keys())[0]] if isinstance(x[1], dict) else -999
        )[0]
        st.info(f"O modelo que mais se destacou foi **{melhor_modelo}**.")

    st.markdown("""
    - Use o modelo com melhor desempenho para prever novos dados.
    - Analise as variáveis mais importantes para orientar decisões estratégicas.
    - Modelos complexos como XGBoost ajudam a identificar padrões ocultos.
    - Se o objetivo for reduzir erro, priorize o modelo com menor RMSE ou MAE.
    """)
    # ================================
    # Gráficos Comparativos
    # ================================
    st.subheader("📊 Gráficos Comparativos")

    if problema == "classificacao":
        for metrica, cor in zip(["Acurácia","F1-Score","Precisão","Recall"],
                                ["skyblue","orange","green","purple"]):
            valores = {m: resultados[m][metrica] for m in resultados if isinstance(resultados[m], dict)}
            fig, ax = plt.subplots()
            ax.bar(valores.keys(), valores.values(), color=cor)
            ax.set_title(f"Comparação de {metrica}")
            ax.set_ylabel(metrica)
            st.pyplot(fig)
    else:
        for metrica, cor in zip(["R²","RMSE","MAE"],
                                ["skyblue","orange","green"]):
            valores = {m: resultados[m][metrica] for m in resultados if isinstance(resultados[m], dict)}
            fig, ax = plt.subplots()
            ax.bar(valores.keys(), valores.values(), color=cor)
            ax.set_title(f"Comparação de {metrica}")
            ax.set_ylabel(metrica)
            st.pyplot(fig)

    # ================================
    # Exportação de Relatórios
    # ================================
    st.subheader("📥 Exportar Relatórios")

    # Botão CSV
    csv = df_resultados.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", csv, "relatorio_modelos.csv", "text/csv")

    # Função para nomes seguros de abas
    def safe_sheet_name(name: str) -> str:
        return ("Imp_" + name.replace(" ", "_"))[:31]

    # Botão Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_resultados.to_excel(writer, sheet_name="Resultados")
        for nome, df_imp in variaveis_importancia.items():
            df_imp.to_excel(writer, sheet_name=safe_sheet_name(nome))
    st.download_button("⬇️ Baixar Excel", buffer.getvalue(),
                       "relatorio_completo.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Botão PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Modelos", ln=True, align="C")

    def clean_text(text):
        return text.encode("latin-1", "ignore").decode("latin-1")

    for modelo, metricas in resultados.items():
        if isinstance(metricas, dict):
            for metrica, valor in metricas.items():
                pdf.cell(200, 10, txt=clean_text(f"{modelo} - {metrica}: {valor:.4f}"), ln=True, align="L")
        else:
            pdf.cell(200, 10, txt=clean_text(f"{modelo}: {metricas}"), ln=True, align="L")

    pdf.add_page()
pdf.cell(200, 10, txt="Importância das Variáveis", ln=True, align="C")

for modelo, df_imp in variaveis_importancia.items():
    pdf.cell(200, 10, txt=clean_text(f"{modelo}"), ln=True, align="L")
    for _, row in df_imp.head(10).iterrows():
        pdf.cell(200, 10, txt=clean_text(f"{row['Variável']}: {row['Importância']:.4f}"), ln=True, align="L")

# Exporta PDF
pdf_output = pdf.output(dest="S").encode("latin-1", "ignore")
st.download_button(
    label="⬇️ Baixar relatório em PDF",
    data=pdf_output,
    file_name="relatorio_completo.pdf",
    mime="application/pdf"
)