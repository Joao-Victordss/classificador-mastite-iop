import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHO_MODELO = RAIZ_PROJETO / "modelos" / "random_forest_mastite.pkl"

COLUNAS_ESPERADAS = [
    "Months_after_giving_birth",
    "IUFL",
    "EUFL",
    "IUFR",
    "EUFR",
    "IURL",
    "EURL",
    "IURR",
    "EURR",
    "Temperature",
]


@st.cache_resource
def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {CAMINHO_MODELO}. Treine primeiro o modelo."
        )
    with open(CAMINHO_MODELO, "rb") as arquivo_modelo:
        return pickle.load(arquivo_modelo)


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas_ausentes = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes no CSV enviado: {', '.join(colunas_ausentes)}. "
            "Use o layout de exemplo_entrada.csv."
        )
    return df[COLUNAS_ESPERADAS].copy()


def main() -> None:
    st.set_page_config(
        page_title="Classificador de risco de mastite em vacas leiteiras (IoT)",
        layout="centered",
    )
    st.title("Classificador de risco de mastite em vacas leiteiras (IoT)")
    st.write(
        "Ferramenta de apoio à decisão baseada em dados de sensores de úbere e temperatura. "
        "Não substitui a avaliação de um(a) veterinário(a)."
    )

    st.sidebar.header("Entrada de dados")
    arquivo = st.sidebar.file_uploader(
        "Envie um arquivo CSV com as leituras de sensores", type=["csv"]
    )

    if arquivo is None:
        st.info("Envie um CSV no formato de exemplo_entrada.csv para obter previsões.")
        return

    try:
        df_original = pd.read_csv(arquivo)
    except Exception as exc:
        st.error(f"Não foi possível ler o CSV enviado: {exc}")
        return

    st.subheader("Dados recebidos")
    st.dataframe(df_original)

    if "ID" not in df_original.columns:
        st.error("A coluna 'ID' é obrigatória para identificar cada animal.")
        return

    df_sem_id = df_original.drop(columns=["ID"])

    try:
        df_modelo = preparar_dados(df_sem_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        modelo = carregar_modelo()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    probabilidades = modelo.predict_proba(df_modelo)
    predicoes = modelo.predict(df_modelo)

    classes_modelo = list(modelo.classes_)
    idx_mastite = classes_modelo.index(0)
    idx_saudavel = classes_modelo.index(1) if 1 in classes_modelo else None

    df_resultado = df_original.copy()
    df_resultado["classe_prevista"] = [
        "Mastite" if p == 0 else "Saudável" for p in predicoes
    ]
    df_resultado["prob_mastite"] = probabilidades[:, idx_mastite]
    if idx_saudavel is not None:
        df_resultado["prob_saudavel"] = probabilidades[:, idx_saudavel]

    st.subheader("Resultados")
    st.dataframe(df_resultado)

    total = len(df_resultado)
    mastite = (df_resultado["classe_prevista"] == "Mastite").sum()
    saudavel = (df_resultado["classe_prevista"] == "Saudável").sum()

    st.markdown(
        f"**Total de animais:** {total} | "
        f"**Com risco de mastite:** {mastite} | "
        f"**Saudáveis:** {saudavel}"
    )


if __name__ == "__main__":
    main()
