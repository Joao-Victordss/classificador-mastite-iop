from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Constantes de caminhos
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_BASE_BALANCEADA = RAIZ_PROJETO / "dados" / "processado" / "mastite_iot_balanceado.csv"
CAMINHO_MODELO = RAIZ_PROJETO / "modelos" / "random_forest_mastite.pkl"


def carregar_dados(caminho: Path):
    """Carrega a base balanceada e separa features e rótulos."""
    df = pd.read_csv(caminho)
    if "classe" not in df.columns:
        raise ValueError("A base balanceada precisa conter a coluna 'classe'.")

    X = df.drop(columns=["classe"])
    y = df["classe"]
    return X, y


def treinar_e_avaliar(X, y):
    """Treina o RandomForest e imprime métricas básicas."""
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    modelo = RandomForestClassifier(
        criterion="entropy",
        n_estimators=100,
        random_state=1000,
    )

    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)

    print(f"Acurácia: {accuracy_score(y_teste, y_pred):.4f}")
    print("Matriz de confusão:")
    print(confusion_matrix(y_teste, y_pred))
    print("Relatório de classificação:")
    print(classification_report(y_teste, y_pred))

    return modelo


def salvar_modelo(modelo, caminho: Path) -> None:
    """Salva o modelo treinado em disco."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as arquivo_modelo:
        pickle.dump(modelo, arquivo_modelo)
    print(f"Modelo salvo em: {caminho}")


def main() -> None:
    X, y = carregar_dados(CAMINHO_BASE_BALANCEADA)
    modelo = treinar_e_avaliar(X, y)
    salvar_modelo(modelo, CAMINHO_MODELO)


if __name__ == "__main__":
    main()
