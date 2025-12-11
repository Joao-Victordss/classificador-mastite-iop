# classificador-mastite-iot

Projeto em Python para classificar risco de mastite em vacas leiteiras usando dados de sensores de úbere e temperatura. A metodologia de preparação de dados e escolha do modelo é inspirada no artigo “MasPA: A Machine Learning Application to Predict Risk of Mastitis in Cattle from AMS Sensor Data” (AgriEngineering, 2021, DOI: https://doi.org/10.3390/agriengineering3030037), adaptada para este repositório.

## Estrutura do projeto
```
classificador-mastite-iot/
├─ dados/
│  ├─ bruto/
│  │  └─ mastite_iot_bruto.csv        # substituir pelo CSV bruto original
│  └─ processado/                     # gerado pelos scripts
├─ modelos/                           # modelos treinados (.pkl)
├─ src/
│  ├─ dados/preparar_base.py          # prepara e balanceia os dados
│  └─ modelos/treinar_random_forest.py# treina e avalia o modelo
├─ exemplo_entrada.csv                # exemplo de entrada para inferência
├─ app_streamlit.py                   # app web para inferência
├─ requirements.txt
└─ README.md
```

## Preparação do ambiente
1. Criar e ativar o ambiente virtual (exemplo no Windows PowerShell):
   ```
   python -m venv .venv
   .\\.venv\\Scripts\\activate
   ```
2. Instalar as dependências:
   ```
   pip install -r requirements.txt
   ```

## Fluxo de uso
1. Coloque o CSV bruto em `dados/bruto/mastite_iot_bruto.csv` (mesmos nomes de features do artigo ou ajuste o mapa de renomeação em `src/dados/preparar_base.py`).  
   - Os dados brutos e processados **não são versionados** (`dados/` está no `.gitignore`); mantenha-os fora do repositório remoto.  
   - Baixe ou exporte o CSV bruto da fonte original (sensores/arquivo do estudo) e coloque manualmente nesse caminho.
2. Gere as bases tratada e balanceada:
   ```
   python src/dados/preparar_base.py
   ```
   - Saídas: `dados/processado/mastite_iot_tratado.csv` e `dados/processado/mastite_iot_balanceado.csv`.
3. Treine o modelo Random Forest (criterion entropy, 100 árvores):
   ```
   python src/modelos/treinar_random_forest.py
   ```
   - Saída: `modelos/random_forest_mastite.pkl`.
4. Suba o app web e envie um CSV no formato de `exemplo_entrada.csv`:
   ```
   streamlit run app_streamlit.py
   ```
   - O app remove a coluna `ID` antes de inferir, mostra classe prevista (0 = Mastite, 1 = Saudável) e probabilidade de mastite.

## Referências
- GHAFOOR, Naeem Abdul; SITKOWSKA, Beata. MasPA: a machine learning application to predict risk of mastitis in cattle from AMS sensor data. *AgriEngineering*, Basel, v. 3, n. 3, p. 575-583, 2021. DOI: 10.3390/agriengineering3030037.
- Repositório original relacionado: https://github.com/naeemmrz/MasPA.py
