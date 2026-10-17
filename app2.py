Python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Painel de Previsão de Energia Industrial",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Este painel interativo utiliza um modelo de **Regressão Linear** treinado com dados simulados para estimar o consumo de energia com base em parâmetros operacionais.")

# 1. Geração do Dataset e Treinamento do Modelo
@st.cache_data
def carregar_dados_e_modelo():
    np.random.seed(42)
    n_registros = 100  # Mais de 20 registros conforme solicitado
    
    # Gerando variáveis simuladas
    velocidade = np.random.uniform(10, 100, n_registros)  # m/min
    maquinas = np.random.randint(1, 25, n_registros)        # unidades
    temp = np.random.uniform(15, 40, n_registros)         # °C
    
    # Criando o alvo (Consumo_kWh) com uma fórmula lógica + ruído aleatório
    consumo = 40 + (2.2 * velocidade) + (14.5 * maquinas) + (1.1 * temp) + np.random.normal(0, 8, n_registros)
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade,
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': temp,
        'Consumo_kWh': consumo
    })
    
    # Treinamento do Modelo de Regressão
    X = df[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
    y = df['Consumo_kWh']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    return df, modelo

df_treino, modelo = carregar_dados_e_modelo()

# 2. Barra Lateral (Sidebar) com Controles Numéricos
st.sidebar.header("Parâmetros Operacionais")
st.sidebar.markdown("Ajuste os valores abaixo para simular a operação:")

vel_input = st.sidebar.slider(
    "Velocidade da Esteira (m/min)", 
    float(df_treino['Velocidade_Esteira'].min()), 
    float(df_treino['Velocidade_Esteira'].max()), 
    float(df_treino['Velocidade_Esteira'].mean())
)

maq_input = st.sidebar.slider(
    "Máquinas Ativas (unidades)", 
    int(df_treino['Maquinas_Ativas'].min()), 
    int(df_treino['Maquinas_Ativas'].max()), 
    int(df_treino['Maquinas_Ativas'].mean())
)

temp_input = st.sidebar.slider(
    "Temperatura Ambiente (°C)", 
    float(df_treino['Temp_Ambiente'].min()), 
    float(df_treino['Temp_Ambiente'].max()), 
    float(df_treino['Temp_Ambiente'].mean())
)

# 3. Predição baseada na entrada do usuário
entrada_usuario = pd.DataFrame({
    'Velocidade_Esteira': [vel_input],
    'Maquinas_Ativas': [maq_input],
    'Temp_Ambiente': [temp_input]
})

consumo_previsto = modelo.predict(entrada_usuario)[0]

# 4. Exibição na Tela Principal
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resultado da Previsão")
    st.markdown("Com base na configuração atual dos parâmetros de fábrica, o consumo estimado é:")
    
    # Exibição em destaque usando st.metric
    st.metric(
        label="Consumo Energético Estimado", 
        value=f"{consumo_previsto:.2f} kWh"
    )
    
    with st.expander("Visualizar amostra do dataset de treinamento"):
        st.dataframe(df_treino.head(10), use_container_width=True)

with col2:
    st.subheader("Contribuição de Cada Variável")
    st.markdown("Impacto absoluto de cada parâmetro na estimativa atual (Coeficiente × Valor):")
    
    # Cálculo da contribuição ponderada para a amostra atual
    variaveis = ['Velocidade da Esteira', 'Máquinas Ativas', 'Temp. Ambiente']
    contribuicao = modelo.coef_ * [vel_input, maq_input, temp_input]
    
    # Gráfico de barras com Matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.barh(variaveis, contribuicao, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xlabel("Contribuição para o Consumo (kWh)")
    ax.set_title("Análise de Impacto dos Atributos")
    ax.bar_label(barras, fmt='%.1f kWh', padding=3)
    
    st.pyplot(fig)
