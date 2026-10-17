import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Painel de Consumo de Energia Industrial",
    page_icon="⚡",
    layout="wide"
)

# 1. Geração do Dataset e Treinamento do Modelo (com Cache)
@st.cache_resource
def carregar_dados_e_modelo():
    np.random.seed(42)
    n_registros = 100  # Mais de 20 registros conforme solicitado
    
    # Gerando dados sintéticos
    velocidade = np.random.uniform(5.0, 30.0, n_registros)      # m/s
    maquinas = np.random.randint(1, 20, n_registros)             # unidades ativas
    temp_amb = np.random.uniform(15.0, 40.0, n_registros)       # °C
    
    # Criando o alvo 'Consumo_kWh' com base em uma relação lógica + ruído
    consumo = (velocidade * 12.5) + (maquinas * 45.2) + (temp_amb * 3.1) + np.random.normal(0, 10, n_registros)
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade,
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': temp_amb,
        'Consumo_kWh': consumo
    })
    
    # Treinando o modelo de Regressão Linear
    X = df[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
    y = df['Consumo_kWh']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    return modelo, df

modelo, df = carregar_dados_e_modelo()

# Título do Painel
st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Este painel utiliza Machine Learning (**Regressão Linear**) para estimar o consumo de energia com base em parâmetros operacionais da fábrica.")

# 2. Barra Lateral (Sidebar) com controles numéricos
st.sidebar.header("⚙️ Parâmetros de Entrada")

vel_input = st.sidebar.slider(
    "Velocidade da Esteira (m/s)", 
    float(df['Velocidade_Esteira'].min()), 
    float(df['Velocidade_Esteira'].max()), 
    float(df['Velocidade_Esteira'].mean())
)

maq_input = st.sidebar.slider(
    "Máquinas Ativas (unidades)", 
    int(df['Maquinas_Ativas'].min()), 
    int(df['Maquinas_Ativas'].max()), 
    int(df['Maquinas_Ativas'].mean())
)

temp_input = st.sidebar.slider(
    "Temperatura Ambiente (°C)", 
    float(df['Temp_Ambiente'].min()), 
    float(df['Temp_Ambiente'].max()), 
    float(df['Temp_Ambiente'].mean())
)

# Criando dataframe com os dados inseridos pelo usuário
entrada_usuario = pd.DataFrame({
    'Velocidade_Esteira': [vel_input],
    'Maquinas_Ativas': [maq_input],
    'Temp_Ambiente': [temp_input]
})

# Realizando a previsão
consumo_previsto = modelo.predict(entrada_usuario)[0]

# 3. Tela Principal: Exibição da Métrica e Gráfico
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Previsão Atual")
    st.metric(
        label="Consumo Estimado",
        value=f"{consumo_previsto:.2f} kWh"
    )
    st.info("Ajuste os valores na barra lateral para ver o impacto em tempo real na estimativa de consumo.")

with col2:
    st.subheader("Contribuição de Cada Variável")
    
    # Calculando a contribuição (Coeficiente * Valor inserido pelo usuário)
    features = ['Velocidade Esteira', 'Máquinas Ativas', 'Temp. Ambiente']
    contribuicoes = modelo.coef_ * [vel_input, maq_input, temp_input]
    
    # Plotando o gráfico de barras com Matplotlib
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.barh(features, contribuicoes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xlabel("Contribuição para o Consumo (kWh)")
    ax.set_title("Impacto Parcial na Predição Atual")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adicionar os valores nas barras
    for barra in barras:
        largura = barra.get_width()
        ax.text(largura + (2 if largura >= 0 else -15), barra.get_y() + barra.get_height()/2, 
                f'{largura:.1f} kWh', va='center', ha='left' if largura >= 0 else 'right', fontsize=9)
        
    st.pyplot(fig)

# Seção expansível para visualizar o dataset gerado
with st.expander("Visualizar Dataset de Treinamento (Amostra)"):
    st.dataframe(df.head(20))
