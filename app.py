import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
import base64
from io import BytesIO
import uuid

# Configuração da página
st.set_page_config(
    page_title="Análise de Resíduos e Energia",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para aplicar estilo CSS personalizado
def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #0080FF;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #0080FF;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #0080FF;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
        .highlight {
            background-color: #F0F8FF;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #0080FF;
            margin-bottom: 1rem;
        }
        .insight-box {
            background-color: #E6F9FF;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #00BFFF;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #FFF3E0;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #FF9800;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Funções para processamento de dados
def carregar_dados(file_path):
    """Carrega os dados do arquivo Excel e faz verificações iniciais"""
    try:
        df = pd.read_excel(file_path)
        st.success(f"Dados carregados com sucesso! Total de registros: {df.shape[0]}")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

def verificar_integridade(df):
    """Verifica a integridade dos dados"""
    if df is None:
        return None, None, None
    
    valores_ausentes = df.isnull().sum()
    duplicatas = df.duplicated().sum()
    tipos_dados = df.dtypes
    return valores_ausentes, duplicatas, tipos_dados

def limpar_dados(df):
    """Limpa os dados"""
    if df is None:
        return None
    
    df_limpo = df.copy()
    
    # Documentar alterações feitas
    alteracoes = []
    
    # Converter tipos de dados se necessário
    for col in df_limpo.select_dtypes(include=['object']).columns:
        try:
            # Tenta converter para numérico, mas não força se não for apropriado para todas as colunas object
            # Idealmente, colunas específicas seriam selecionadas para conversão numérica.
            # Esta tentativa genérica pode ser mantida, mas com cautela.
            pd.to_numeric(df_limpo[col]) # Teste de conversão
            # Se a conversão for bem-sucedida E desejada para esta coluna, descomente a linha abaixo
            # df_limpo[col] = pd.to_numeric(df_limpo[col])
            # alteracoes.append(f"Coluna '{col}' continha apenas números e foi considerada numérica.")
        except ValueError: # Especificar o erro esperado (ValueError para falha na conversão para numérico)
            pass # Mantém como objeto se não puder ser convertido para numérico
    
    # Tratar valores ausentes
    for col in df_limpo.select_dtypes(include=['float64', 'int64']).columns:
        if df_limpo[col].isnull().sum() > 0:
            valor_anterior = df_limpo[col].isnull().sum()
            df_limpo[col] = df_limpo[col].fillna(df_limpo[col].mean())
            alteracoes.append(f"Coluna '{col}': {valor_anterior} valores ausentes preenchidos com a média")
    
    for col in df_limpo.select_dtypes(include=['object']).columns:
        if df_limpo[col].isnull().sum() > 0:
            valor_anterior = df_limpo[col].isnull().sum()
            # Garante que haja pelo menos um modo antes de tentar acessá-lo
            if not df_limpo[col].mode().empty:
                df_limpo[col] = df_limpo[col].fillna(df_limpo[col].mode()[0])
                alteracoes.append(f"Coluna '{col}': {valor_anterior} valores ausentes preenchidos com a moda")
            else:
                # Se não houver modo (coluna só com NaNs, por exemplo), preenche com um placeholder
                df_limpo[col] = df_limpo[col].fillna("Desconhecido")
                alteracoes.append(f"Coluna '{col}': {valor_anterior} valores ausentes preenchidos com 'Desconhecido' (sem moda)")

    
    # Remover duplicatas se existirem
    if df_limpo.duplicated().sum() > 0:
        valor_anterior = df_limpo.duplicated().sum()
        df_limpo = df_limpo.drop_duplicates()
        alteracoes.append(f"Removidas {valor_anterior} linhas duplicadas")
    
    return df_limpo, alteracoes

def detectar_outliers(df, coluna, metodo='iqr'):
    """Detecta outliers em uma coluna"""
    if df is None or coluna not in df.columns or df[coluna].dtype not in ['float64', 'int64']:
        st.warning(f"Coluna '{coluna}' não é numérica ou não existe. Não é possível detectar outliers.")
        return None, None, None
    
    if metodo == 'iqr':
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
        return outliers, limite_inferior, limite_superior
    elif metodo == 'zscore':
        # Z-score só faz sentido para dados que se aproximam da normalidade
        # e pode ser sensível a amostras pequenas.
        if len(df[coluna].dropna()) < 3: # Z-score de 1 ou 2 pontos não é útil
             st.warning(f"Coluna '{coluna}' tem poucos valores para cálculo de Z-score confiável.")
             return None, None, None
        z_scores = np.abs(stats.zscore(df[coluna].dropna())) #dropna para evitar erro com NaNs
        outliers = df.loc[df[coluna].dropna().index[z_scores > 3]] # Mapeia de volta para o índice original
        return outliers, None, None # Z-score não define limites explícitos como IQR

def testar_normalidade(df, coluna):
    """Testa a normalidade de uma coluna"""
    if df is None or coluna not in df.columns or df[coluna].dtype not in ['float64', 'int64']:
        st.warning(f"Coluna '{coluna}' não é numérica ou não existe.")
        return (None, None), (None, None)
    
    dados_validos = df[coluna].dropna()
    if len(dados_validos) < 3: # Testes de normalidade precisam de um mínimo de amostras
        st.warning(f"Coluna '{coluna}' tem menos de 3 valores válidos para teste de normalidade.")
        return (None, None), (None, None)

    stat_sw, p_sw = None, None
    # Shapiro-Wilk é geralmente recomendado para amostras < 5000
    if len(dados_validos) >=3 and len(dados_validos) < 5000:
        stat_sw, p_sw = stats.shapiro(dados_validos)
    
    # Kolmogorov-Smirnov é mais geral
    stat_ks, p_ks = stats.kstest(dados_validos, 'norm', args=(dados_validos.mean(), dados_validos.std()))
    return (stat_sw, p_sw), (stat_ks, p_ks)

def realizar_teste_nao_parametrico(df, coluna_numerica, grupo_coluna):
    """Realiza testes não paramétricos"""
    if df is None or coluna_numerica not in df.columns or grupo_coluna not in df.columns:
        return None, None
    if df[coluna_numerica].dtype not in ['float64', 'int64']:
        st.warning(f"Coluna numérica '{coluna_numerica}' não é do tipo correto.")
        return None, None
    if df[grupo_coluna].nunique() < 2:
        st.warning(f"Coluna de grupo '{grupo_coluna}' tem menos de 2 grupos distintos.")
        return None, None

    grupos = df[grupo_coluna].unique()
    dados_grupos = [df[df[grupo_coluna] == grupo][coluna_numerica].dropna() for grupo in grupos]
    dados_grupos = [g for g in dados_grupos if len(g) > 0] # Remove grupos vazios

    if len(dados_grupos) < 2:
        st.warning("Não há grupos suficientes com dados para realizar o teste.")
        return None, None

    if len(dados_grupos) == 2:
        # Teste Mann-Whitney para dois grupos
        stat, p = stats.mannwhitneyu(dados_grupos[0], dados_grupos[1], alternative='two-sided')
        return "Mann-Whitney U", p
    elif len(dados_grupos) > 2:
        # Teste Kruskal-Wallis para mais de dois grupos
        try:
            stat, p = stats.kruskal(*dados_grupos)
            return "Kruskal-Wallis", p
        except ValueError as e: #Pode ocorrer se algum grupo tiver todos os valores iguais
            st.warning(f"Erro no teste Kruskal-Wallis: {e}")
            return None, None
    return None, None


def calcular_estatisticas(df, ano=None):
    """Calcula estatísticas descritivas dos dados"""
    if df is None:
        return None
    df_filtrado = df[df['Ano'] == ano] if ano is not None and 'Ano' in df.columns else df
    df_num = df_filtrado.select_dtypes(include=np.number) # Usa np.number para incluir int e float
    if df_num.empty:
        st.warning("Nenhuma coluna numérica encontrada para calcular estatísticas.")
        return None
    estatisticas = df_num.describe().T
    estatisticas['IQR'] = estatisticas['75%'] - estatisticas['25%']
    estatisticas['Assimetria'] = df_num.skew()
    estatisticas['Curtose'] = df_num.kurtosis()
    return estatisticas

def criar_boxplot(df, ano=None):
    """Cria boxplots para as variáveis numéricas por tipo de resíduo"""
    if df is None or 'Tipo de Resíduo' not in df.columns:
        return None, None, None
    df_filtrado = df[df['Ano'] == ano] if ano is not None and 'Ano' in df.columns else df
    
    fig_gerado, fig_reciclado, fig_porcentagem = None, None, None
    if 'Resíduo Gerado (ton)' in df_filtrado.columns:
        fig_gerado = px.box(df_filtrado, x='Tipo de Resíduo', y='Resíduo Gerado (ton)', color='Tipo de Resíduo', title='Distribuição de Resíduos Gerados por Tipo')
    if 'Resíduo Reciclado (ton)' in df_filtrado.columns:
        fig_reciclado = px.box(df_filtrado, x='Tipo de Resíduo', y='Resíduo Reciclado (ton)', color='Tipo de Resíduo', title='Distribuição de Resíduos Reciclados por Tipo')
    if '% Reciclagem' in df_filtrado.columns:
        fig_porcentagem = px.box(df_filtrado, x='Tipo de Resíduo', y='% Reciclagem', color='Tipo de Resíduo', title='Distribuição da Porcentagem de Reciclagem por Tipo')
    return fig_gerado, fig_reciclado, fig_porcentagem

def criar_histogramas(df, ano=None):
    """Cria histogramas para as variáveis numéricas"""
    if df is None:
        return None, None, None
    df_filtrado = df[df['Ano'] == ano] if ano is not None and 'Ano' in df.columns else df
    
    fig_gerado, fig_reciclado, fig_energia = None, None, None
    if 'Resíduo Gerado (ton)' in df_filtrado.columns:
        fig_gerado = px.histogram(df_filtrado, x='Resíduo Gerado (ton)', nbins=20, title='Distribuição de Resíduos Gerados', marginal='box')
    if 'Resíduo Reciclado (ton)' in df_filtrado.columns:
        fig_reciclado = px.histogram(df_filtrado, x='Resíduo Reciclado (ton)', nbins=20, title='Distribuição de Resíduos Reciclados', marginal='box')
    if 'Energia Gerada (kWh)' in df_filtrado.columns:
        fig_energia = px.histogram(df_filtrado, x='Energia Gerada (kWh)', nbins=20, title='Distribuição de Energia Gerada', marginal='box')
    return fig_gerado, fig_reciclado, fig_energia

def criar_scatter_plots(df, ano=None):
    """Cria gráficos de dispersão entre variáveis"""
    if df is None:
        return None, None
    df_filtrado = df[df['Ano'] == ano] if ano is not None and 'Ano' in df.columns else df
    
    fig_energia_emissoes, fig_gerado_reciclado = None, None
    required_cols_energia = ['Energia Gerada (kWh)', 'Emissões Evitadas (kg CO₂)', 'Tipo de Resíduo', 'Painéis Solares Instalados', 'Nome do Resíduo', 'Ano', 'Região']
    if all(col in df_filtrado.columns for col in required_cols_energia):
        fig_energia_emissoes = px.scatter(df_filtrado, x='Energia Gerada (kWh)', y='Emissões Evitadas (kg CO₂)', color='Tipo de Resíduo', size='Painéis Solares Instalados', hover_name='Nome do Resíduo', hover_data=['Ano', 'Região'], title='Relação entre Energia Gerada e Emissões Evitadas')
    
    required_cols_residuos = ['Resíduo Gerado (ton)', 'Resíduo Reciclado (ton)', 'Tipo de Resíduo', '% Reciclagem', 'Nome do Resíduo', 'Ano', 'Região']
    if all(col in df_filtrado.columns for col in required_cols_residuos):
        fig_gerado_reciclado = px.scatter(df_filtrado, x='Resíduo Gerado (ton)', y='Resíduo Reciclado (ton)', color='Tipo de Resíduo', size='% Reciclagem', hover_name='Nome do Resíduo', hover_data=['Ano', 'Região'], title='Relação entre Resíduo Gerado e Reciclado')
    return fig_energia_emissoes, fig_gerado_reciclado

def criar_series_temporais(df):
    """Cria gráficos de séries temporais"""
    if df is None or 'Ano' not in df.columns:
        return None, None, None
        
    agg_dict = {}
    if 'Resíduo Gerado (ton)' in df.columns: agg_dict['Resíduo Gerado (ton)'] = 'sum'
    if 'Resíduo Reciclado (ton)' in df.columns: agg_dict['Resíduo Reciclado (ton)'] = 'sum'
    if '% Reciclagem' in df.columns: agg_dict['% Reciclagem'] = 'mean'
    if 'Painéis Solares Instalados' in df.columns: agg_dict['Painéis Solares Instalados'] = 'sum'
    if 'Energia Gerada (kWh)' in df.columns: agg_dict['Energia Gerada (kWh)'] = 'sum'
    if 'Emissões Evitadas (kg CO₂)' in df.columns: agg_dict['Emissões Evitadas (kg CO₂)'] = 'sum'

    if not agg_dict: # Se nenhuma coluna relevante existir
        return None, None, None

    df_anual = df.groupby('Ano').agg(agg_dict).reset_index()

    fig_residuos, fig_energia, fig_reciclagem = None, None, None

    if 'Resíduo Gerado (ton)' in df_anual.columns and 'Resíduo Reciclado (ton)' in df_anual.columns:
        fig_residuos = go.Figure()
        fig_residuos.add_trace(go.Scatter(x=df_anual['Ano'], y=df_anual['Resíduo Gerado (ton)'], mode='lines+markers', name='Resíduo Gerado (ton)'))
        fig_residuos.add_trace(go.Scatter(x=df_anual['Ano'], y=df_anual['Resíduo Reciclado (ton)'], mode='lines+markers', name='Resíduo Reciclado (ton)'))
        fig_residuos.update_layout(title='Evolução de Resíduos ao Longo dos Anos', xaxis_title='Ano', yaxis_title='Toneladas')

    if 'Energia Gerada (kWh)' in df_anual.columns and 'Emissões Evitadas (kg CO₂)' in df_anual.columns:
        fig_energia = make_subplots(specs=[[{"secondary_y": True}]])
        fig_energia.add_trace(go.Scatter(x=df_anual['Ano'], y=df_anual['Energia Gerada (kWh)'], mode='lines+markers', name='Energia Gerada (kWh)'), secondary_y=False)
        fig_energia.add_trace(go.Scatter(x=df_anual['Ano'], y=df_anual['Emissões Evitadas (kg CO₂)'], mode='lines+markers', name='Emissões Evitadas (kg CO₂)'), secondary_y=True)
        fig_energia.update_layout(title='Evolução de Energia Gerada e Emissões Evitadas', xaxis_title='Ano')
        fig_energia.update_yaxes(title_text='Energia Gerada (kWh)', secondary_y=False)
        fig_energia.update_yaxes(title_text='Emissões Evitadas (kg CO₂)', secondary_y=True)

    if '% Reciclagem' in df_anual.columns:
        fig_reciclagem = px.line(df_anual, x='Ano', y='% Reciclagem', markers=True, title='Evolução da Porcentagem de Reciclagem ao Longo dos Anos')
    
    return fig_residuos, fig_energia, fig_reciclagem


def criar_mapa_calor(df):
    """Cria um mapa de calor para correlações entre variáveis numéricas"""
    if df is None:
        return None
    df_num = df.select_dtypes(include=np.number)
    if df_num.shape[1] < 2: # Precisa de pelo menos 2 colunas numéricas para correlação
        st.warning("Não há colunas numéricas suficientes para gerar um mapa de correlação.")
        return None
    correlacoes = df_num.corr()
    fig = px.imshow(correlacoes, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title='Mapa de Correlação entre Variáveis')
    return fig

def criar_analise_por_regiao(df, ano=None):
    """Cria gráficos para análise por região"""
    if df is None or 'Região' not in df.columns:
        return None, None, None
    df_filtrado = df[df['Ano'] == ano] if ano is not None and 'Ano' in df.columns else df
    
    agg_dict = {}
    if 'Resíduo Gerado (ton)' in df.columns: agg_dict['Resíduo Gerado (ton)'] = 'sum'
    if 'Resíduo Reciclado (ton)' in df.columns: agg_dict['Resíduo Reciclado (ton)'] = 'sum'
    if '% Reciclagem' in df.columns: agg_dict['% Reciclagem'] = 'mean'
    if 'Painéis Solares Instalados' in df.columns: agg_dict['Painéis Solares Instalados'] = 'sum'
    if 'Energia Gerada (kWh)' in df.columns: agg_dict['Energia Gerada (kWh)'] = 'sum'
    if 'Emissões Evitadas (kg CO₂)' in df.columns: agg_dict['Emissões Evitadas (kg CO₂)'] = 'sum'

    if not agg_dict:
        return None, None, None
        
    df_regiao = df_filtrado.groupby('Região').agg(agg_dict).reset_index()

    fig_residuos, fig_reciclagem, fig_energia = None, None, None

    if 'Resíduo Gerado (ton)' in df_regiao.columns and 'Resíduo Reciclado (ton)' in df_regiao.columns:
        fig_residuos = px.bar(df_regiao, x='Região', y=['Resíduo Gerado (ton)', 'Resíduo Reciclado (ton)'], barmode='group', title='Resíduos por Região')
    if '% Reciclagem' in df_regiao.columns:
        fig_reciclagem = px.bar(df_regiao, x='Região', y='% Reciclagem', title='Porcentagem de Reciclagem por Região', color='% Reciclagem', color_continuous_scale='Viridis')
    if 'Energia Gerada (kWh)' in df_regiao.columns:
        color_param = 'Painéis Solares Instalados' if 'Painéis Solares Instalados' in df_regiao.columns else None
        fig_energia = px.bar(df_regiao, x='Região', y='Energia Gerada (kWh)', title='Energia Gerada por Região', color=color_param, color_continuous_scale='Plasma' if color_param else None)
    
    return fig_residuos, fig_reciclagem, fig_energia


def criar_analise_por_tipo_residuo(df, ano=None):
    """Cria gráficos para análise por tipo de resíduo"""
    if df is None or 'Tipo de Resíduo' not in df.columns:
        return None, None, None
    df_filtrado = df[df['Ano'] == ano] if ano is not None and 'Ano' in df.columns else df
    
    agg_dict = {}
    if 'Resíduo Gerado (ton)' in df.columns: agg_dict['Resíduo Gerado (ton)'] = 'sum'
    if 'Resíduo Reciclado (ton)' in df.columns: agg_dict['Resíduo Reciclado (ton)'] = 'sum'
    if '% Reciclagem' in df.columns: agg_dict['% Reciclagem'] = 'mean'
    # As colunas abaixo não fazem sentido agregar por tipo de resíduo da mesma forma que por região/ano
    # 'Painéis Solares Instalados': 'sum',
    # 'Energia Gerada (kWh)': 'sum',
    # 'Emissões Evitadas (kg CO₂)': 'sum'

    if not agg_dict:
        return None, None, None
        
    df_tipo = df_filtrado.groupby('Tipo de Resíduo').agg(agg_dict).reset_index()

    fig_gerado, fig_reciclado, fig_reciclagem = None, None, None

    if 'Resíduo Gerado (ton)' in df_tipo.columns:
        fig_gerado = px.pie(df_tipo, values='Resíduo Gerado (ton)', names='Tipo de Resíduo', title='Distribuição de Resíduos Gerados por Tipo')
    if 'Resíduo Reciclado (ton)' in df_tipo.columns:
        fig_reciclado = px.pie(df_tipo, values='Resíduo Reciclado (ton)', names='Tipo de Resíduo', title='Distribuição de Resíduos Reciclados por Tipo')
    if '% Reciclagem' in df_tipo.columns:
        fig_reciclagem = px.bar(df_tipo, x='Tipo de Resíduo', y='% Reciclagem', title='Porcentagem de Reciclagem por Tipo', color='Tipo de Resíduo')
    
    return fig_gerado, fig_reciclado, fig_reciclagem


def gerar_insight_reciclagem(df):
    """Gera insights sobre reciclagem"""
    if df is None or 'Ano' not in df.columns or 'Tipo de Resíduo' not in df.columns:
        return None
    
    insights = {}
    
    # Insights anuais
    df_anual_agg = {}
    if 'Resíduo Gerado (ton)' in df.columns: df_anual_agg['Resíduo Gerado (ton)'] = 'sum'
    if 'Resíduo Reciclado (ton)' in df.columns: df_anual_agg['Resíduo Reciclado (ton)'] = 'sum'
    if '% Reciclagem' in df.columns: df_anual_agg['% Reciclagem'] = 'mean'

    if not df_anual_agg: return None
    df_anual = df.groupby('Ano').agg(df_anual_agg).reset_index()
    
    if '% Reciclagem' in df_anual.columns and not df_anual.empty:
        df_anual['Taxa Crescimento Reciclagem'] = df_anual['% Reciclagem'].pct_change() * 100
        insights['ano_maior_reciclagem'] = int(df_anual.loc[df_anual['% Reciclagem'].idxmax()]['Ano']) if not df_anual['% Reciclagem'].empty else 'N/A'
        insights['maior_reciclagem'] = df_anual['% Reciclagem'].max() if not df_anual['% Reciclagem'].empty else 0
    insights['df_anual'] = df_anual # Sempre retorna o df_anual, mesmo que parcial

    # Insights por tipo de resíduo
    df_tipo_agg = {}
    if '% Reciclagem' in df.columns: df_tipo_agg['% Reciclagem'] = 'mean'
    
    if df_tipo_agg:
        df_tipo = df.groupby('Tipo de Resíduo').agg(df_tipo_agg).reset_index()
        if '% Reciclagem' in df_tipo.columns and not df_tipo.empty:
            insights['tipo_maior_reciclagem'] = df_tipo.loc[df_tipo['% Reciclagem'].idxmax()]['Tipo de Resíduo'] if not df_tipo['% Reciclagem'].empty else 'N/A'
            insights['taxa_maior_reciclagem'] = df_tipo['% Reciclagem'].max() if not df_tipo['% Reciclagem'].empty else 0

    # Potencial de reciclagem
    if 'Resíduo Gerado (ton)' in df.columns and 'Resíduo Reciclado (ton)' in df.columns:
        insights['potencial'] = df['Resíduo Gerado (ton)'].sum() - df['Resíduo Reciclado (ton)'].sum()
    
    return insights


def gerar_insight_energia(df):
    """Gera insights sobre energia"""
    if df is None or 'Ano' not in df.columns or 'Região' not in df.columns:
        return None
    
    insights = {}
    
    # Insights anuais
    df_anual_agg = {}
    if 'Energia Gerada (kWh)' in df.columns: df_anual_agg['Energia Gerada (kWh)'] = 'sum'
    if 'Painéis Solares Instalados' in df.columns: df_anual_agg['Painéis Solares Instalados'] = 'sum'
    if 'Emissões Evitadas (kg CO₂)' in df.columns: df_anual_agg['Emissões Evitadas (kg CO₂)'] = 'sum'
    if 'Custo Médio Energia (R$/kWh)' in df.columns: df_anual_agg['Custo Médio Energia (R$/kWh)'] = 'mean'

    if not df_anual_agg: return None # Se não houver colunas relevantes para insights de energia
    df_anual = df.groupby('Ano').agg(df_anual_agg).reset_index()

    if 'Energia Gerada (kWh)' in df_anual.columns and 'Painéis Solares Instalados' in df_anual.columns and not df_anual.empty:
        # Evitar divisão por zero ou por NaN
        df_anual['Eficiência Painéis (kWh/painel)'] = np.where(
            (df_anual['Painéis Solares Instalados'].notna()) & (df_anual['Painéis Solares Instalados'] != 0),
            df_anual['Energia Gerada (kWh)'] / df_anual['Painéis Solares Instalados'],
            np.nan
        )

    if 'Energia Gerada (kWh)' in df_anual.columns and not df_anual.empty:
        df_anual['Taxa Crescimento Energia'] = df_anual['Energia Gerada (kWh)'].pct_change() * 100
        insights['ano_maior_energia'] = int(df_anual.loc[df_anual['Energia Gerada (kWh)'].idxmax()]['Ano']) if not df_anual['Energia Gerada (kWh)'].empty else 'N/A'
        insights['maior_energia'] = df_anual['Energia Gerada (kWh)'].max() if not df_anual['Energia Gerada (kWh)'].empty else 0
    
    insights['df_anual'] = df_anual

    # Insights por região
    df_regiao_agg = {}
    if 'Energia Gerada (kWh)' in df.columns: df_regiao_agg['Energia Gerada (kWh)'] = 'sum'
    if 'Painéis Solares Instalados' in df.columns: df_regiao_agg['Painéis Solares Instalados'] = 'sum'
    
    if df_regiao_agg:
        df_regiao = df.groupby('Região').agg(df_regiao_agg).reset_index()
        if 'Energia Gerada (kWh)' in df_regiao.columns and 'Painéis Solares Instalados' in df_regiao.columns and not df_regiao.empty:
            df_regiao['Eficiência Painéis (kWh/painel)'] = np.where(
                (df_regiao['Painéis Solares Instalados'].notna()) & (df_regiao['Painéis Solares Instalados'] != 0),
                df_regiao['Energia Gerada (kWh)'] / df_regiao['Painéis Solares Instalados'],
                np.nan
            )
            if not df_regiao['Eficiência Painéis (kWh/painel)'].dropna().empty:
                 insights['regiao_maior_eficiencia'] = df_regiao.loc[df_regiao['Eficiência Painéis (kWh/painel)'].idxmax()]['Região'] if not df_regiao['Eficiência Painéis (kWh/painel)'].dropna().empty else "N/A"

        if 'Energia Gerada (kWh)' in df_regiao.columns and not df_regiao.empty:
            insights['regiao_maior_energia'] = df_regiao.loc[df_regiao['Energia Gerada (kWh)'].idxmax()]['Região'] if not df_regiao['Energia Gerada (kWh)'].empty else 'N/A'
        insights['df_regiao'] = df_regiao
        
    return insights


def gerar_recomendacoes(df):
    """Gera recomendações baseadas nos dados"""
    if df is None:
        return None
    
    recomendacoes = {
        'tendencia_reciclagem': "Indeterminada",
        'tipo_menor_reciclagem': "N/A",
        'regiao_menor_reciclagem': "N/A",
        'tendencia_eficiencia': "Indeterminada" # Eficiência pode ser definida de várias formas
    }
    
    # Análise de tendências de reciclagem
    if 'Ano' in df.columns and '% Reciclagem' in df.columns:
        df_anual_rec = df.groupby('Ano')['% Reciclagem'].mean().reset_index()
        if len(df_anual_rec) > 1:
            # Simples comparação entre o último e primeiro ano disponível
            if df_anual_rec['% Reciclagem'].iloc[-1] > df_anual_rec['% Reciclagem'].iloc[0]:
                recomendacoes['tendencia_reciclagem'] = "crescente"
            elif df_anual_rec['% Reciclagem'].iloc[-1] < df_anual_rec['% Reciclagem'].iloc[0]:
                recomendacoes['tendencia_reciclagem'] = "decrescente"
            else:
                recomendacoes['tendencia_reciclagem'] = "estável"
    
    # Tipo de resíduo com menor taxa de reciclagem
    if 'Tipo de Resíduo' in df.columns and '% Reciclagem' in df.columns:
        df_tipo_rec = df.groupby('Tipo de Resíduo')['% Reciclagem'].mean().reset_index()
        if not df_tipo_rec.empty and not df_tipo_rec['% Reciclagem'].dropna().empty:
            recomendacoes['tipo_menor_reciclagem'] = df_tipo_rec.loc[df_tipo_rec['% Reciclagem'].idxmin()]['Tipo de Resíduo']

    # Região com menor taxa de reciclagem
    if 'Região' in df.columns and '% Reciclagem' in df.columns:
        df_regiao_rec = df.groupby('Região')['% Reciclagem'].mean().reset_index()
        if not df_regiao_rec.empty and not df_regiao_rec['% Reciclagem'].dropna().empty:
            recomendacoes['regiao_menor_reciclagem'] = df_regiao_rec.loc[df_regiao_rec['% Reciclagem'].idxmin()]['Região']

    # Tendência de eficiência energética (ex: kWh por painel)
    if 'Ano' in df.columns and 'Energia Gerada (kWh)' in df.columns and 'Painéis Solares Instalados' in df.columns:
        df_anual_energia = df.groupby('Ano').agg({
            'Energia Gerada (kWh)': 'sum',
            'Painéis Solares Instalados': 'sum'
        }).reset_index()
        df_anual_energia['Eficiência Painéis'] = np.where(
            (df_anual_energia['Painéis Solares Instalados'].notna()) & (df_anual_energia['Painéis Solares Instalados'] != 0),
            df_anual_energia['Energia Gerada (kWh)'] / df_anual_energia['Painéis Solares Instalados'],
            np.nan
        )
        df_anual_energia.dropna(subset=['Eficiência Painéis'], inplace=True) # Remover anos sem dados de eficiência
        if len(df_anual_energia) > 1:
            if df_anual_energia['Eficiência Painéis'].iloc[-1] > df_anual_energia['Eficiência Painéis'].iloc[0]:
                recomendacoes['tendencia_eficiencia'] = "crescente"
            elif df_anual_energia['Eficiência Painéis'].iloc[-1] < df_anual_energia['Eficiência Painéis'].iloc[0]:
                recomendacoes['tendencia_eficiencia'] = "decrescente"
            else:
                recomendacoes['tendencia_eficiencia'] = "estável"

    return recomendacoes


def download_dataframe(df, nome_arquivo):
    """Prepara um dataframe para download"""
    if df is None:
        return None
    output = BytesIO()
    # Use openpyxl se xlsxwriter não estiver disponível ou causar problemas
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Dados')
    except ImportError: # Fallback para openpyxl se xlsxwriter não estiver instalado
         with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Dados')

    dados = output.getvalue()
    b64 = base64.b64encode(dados).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{nome_arquivo}">Baixar dados em Excel</a>'
    return href

def gerar_relatorio_pdf(df): #Renomeado para gerar_relatorio_markdown, pois PDF é mais complexo
    """Gera um relatório em Markdown com os principais insights e recomendações"""
    if df is None:
        return None
    
    # Gerar insights
    insights_reciclagem = gerar_insight_reciclagem(df)
    insights_energia = gerar_insight_energia(df)
    recomendacoes = gerar_recomendacoes(df)

    # Default values if insights are None or keys are missing
    ano_maior_rec = insights_reciclagem.get('ano_maior_reciclagem', 'N/A') if insights_reciclagem else 'N/A'
    maior_rec = insights_reciclagem.get('maior_reciclagem', 0) if insights_reciclagem else 0
    tipo_maior_rec = insights_reciclagem.get('tipo_maior_reciclagem', 'N/A') if insights_reciclagem else 'N/A'
    taxa_maior_rec = insights_reciclagem.get('taxa_maior_reciclagem', 0) if insights_reciclagem else 0
    potencial_rec = insights_reciclagem.get('potencial', 0) if insights_reciclagem else 0

    ano_maior_en = insights_energia.get('ano_maior_energia', 'N/A') if insights_energia else 'N/A'
    maior_en = insights_energia.get('maior_energia', 0) if insights_energia else 0
    regiao_maior_en = insights_energia.get('regiao_maior_energia', 'N/A') if insights_energia else 'N/A'
    regiao_maior_ef = insights_energia.get('regiao_maior_eficiencia', 'N/A') if insights_energia else 'N/A'
    
    tend_rec = recomendacoes.get('tendencia_reciclagem', 'Indeterminada') if recomendacoes else 'Indeterminada'
    tipo_menor_rec = recomendacoes.get('tipo_menor_reciclagem', 'N/A') if recomendacoes else 'N/A'
    regiao_menor_rec = recomendacoes.get('regiao_menor_reciclagem', 'N/A') if recomendacoes else 'N/A'
    tend_ef = recomendacoes.get('tendencia_eficiencia', 'Indeterminada') if recomendacoes else 'Indeterminada'


    relatorio_md = f"""# Relatório de Análise de Resíduos e Energia

## Resumo Executivo

Este relatório apresenta uma análise detalhada dos dados de resíduos e energia, 
abrangendo diferentes regiões e tipos de resíduos. O objetivo é fornecer insights sobre tendências de reciclagem, 
geração de energia renovável e impacto ambiental.

## Principais Insights sobre Reciclagem

- O ano com maior taxa média de reciclagem foi {ano_maior_rec}, atingindo {maior_rec:.2f}%.
- O tipo de resíduo com maior taxa de reciclagem é {tipo_maior_rec}, com média de {taxa_maior_rec:.2f}%.
- Existe um potencial não aproveitado de aproximadamente {potencial_rec:.2f} toneladas de resíduos que poderiam ser reciclados.
- A tendência geral de reciclagem é {tend_rec}.

## Principais Insights sobre Energia

- O ano com maior geração de energia foi {ano_maior_en}, produzindo {maior_en:.2f} kWh.
- A região que mais gera energia é {regiao_maior_en}.
- A região com maior eficiência de painéis solares é {regiao_maior_ef}.
- A tendência de eficiência energética (kWh/painel) é {tend_ef}.

## Recomendações

### Para Melhoria da Reciclagem

1. **Foco em {tipo_menor_rec}**: Este tipo de resíduo apresenta a menor taxa de reciclagem e deve ser priorizado em campanhas de conscientização e infraestrutura.
2. **Atenção à região de {regiao_menor_rec}**: Esta região necessita de maior investimento em programas de reciclagem e infraestrutura.
3. **Implementação de políticas de incentivo**: Desenvolver políticas que incentivem a separação e reciclagem de resíduos, como programas de recompensa ou taxas reduzidas.
4. **Educação ambiental**: Intensificar programas educativos sobre a importância da reciclagem e seu impacto positivo no meio ambiente.

### Para Eficiência Energética

1. **Expansão de painéis solares**: Aumentar a instalação de painéis solares, especialmente em regiões com alta eficiência comprovada.
2. **Modernização tecnológica**: Investir em tecnologias mais eficientes para geração de energia renovável.
3. **Integração de sistemas**: Desenvolver sistemas que integrem a gestão de resíduos com a geração de energia, maximizando o aproveitamento de recursos.
4. **Monitoramento contínuo**: Implementar sistemas de monitoramento para acompanhar a eficiência dos painéis solares e identificar oportunidades de melhoria.

## Conclusão

A análise dos dados revela um panorama {tend_rec} em termos de reciclagem e {tend_ef} em termos de eficiência energética. 
As recomendações apresentadas visam potencializar os resultados positivos e mitigar os desafios identificados, 
contribuindo para um futuro mais sustentável e energeticamente eficiente.
"""
    
    relatorio_filename = "relatorio_residuos_energia.md"
    # Salvar o relatório em um arquivo markdown no diretório local do projeto
    try:
        with open(relatorio_filename, "w", encoding='utf-8') as f: # Adicionado encoding
            f.write(relatorio_md)
        return relatorio_filename
    except Exception as e:
        st.error(f"Erro ao salvar o relatório: {e}")
        return None


# --- Interface Principal do Streamlit ---
st.markdown("<h1 class='main-header'>Análise de Resíduos e Energia</h1>", unsafe_allow_html=True)

# Carregamento de dados
# Tentativa de carregar o arquivo do mesmo diretório do script
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "base_residuos_energia_nome.xlsx")

if os.path.exists(file_path):
    df_original = carregar_dados(file_path)
else:
    st.error(f"Arquivo '{file_path}' não encontrado. Verifique o caminho e o nome do arquivo.")
    df_original = None

# Estado da sessão para armazenar df_limpo e alterações
if 'df_limpo' not in st.session_state:
    st.session_state.df_limpo = None
if 'alteracoes_limpeza' not in st.session_state:
    st.session_state.alteracoes_limpeza = []


if df_original is not None and st.session_state.df_limpo is None: # Processar apenas uma vez
    st.session_state.df_limpo, st.session_state.alteracoes_limpeza = limpar_dados(df_original)

# Verifica se os dados foram carregados e limpos corretamente
if st.session_state.df_limpo is not None:
    df_limpo_atual = st.session_state.df_limpo # Usar o df_limpo do estado da sessão

    # Sidebar para filtros e seleção de anos
    st.sidebar.markdown("## Filtros")
    anos_disponiveis = []
    if 'Ano' in df_limpo_atual.columns:
        anos_disponiveis = sorted(df_limpo_atual['Ano'].unique())
    
    ano_selecionado = st.sidebar.selectbox(
        "Selecione um ano (opcional)", 
        [None] + [int(ano) for ano in anos_disponiveis] #Garante que os anos sejam inteiros se forem numéricos
    )
    
    # Menu de navegação
    st.sidebar.markdown("## Navegação")
    pagina = st.sidebar.radio(
        "Escolha uma seção:", 
        [
            "📊 Visão Geral dos Dados", 
            "🧹 Limpeza e Qualidade dos Dados", 
            "📈 Análise Exploratória",
            "📉 Estatísticas Descritivas",
            "🔍 Visualizações Interativas",
            "📆 Análise Temporal",
            "🌎 Análise Regional",
            "♻️ Análise por Tipo de Resíduo",
            "💡 Insights e Recomendações"
        ]
    )
    
    # Páginas do aplicativo
    if pagina == "📊 Visão Geral dos Dados":
        st.markdown("<h2 class='sub-header'>Visão Geral dos Dados</h2>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-header'>Primeiras linhas do dataset</h3>", unsafe_allow_html=True)
        st.dataframe(df_limpo_atual.head())
        st.markdown("<h3 class='section-header'>Informações gerais</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", df_limpo_atual.shape[0])
        with col2:
            st.metric("Total de Colunas", df_limpo_atual.shape[1])
        with col3:
            if 'Ano' in df_limpo_atual.columns and not df_limpo_atual['Ano'].empty:
                st.metric("Anos Analisados", f"{int(df_limpo_atual['Ano'].min())} - {int(df_limpo_atual['Ano'].max())}")
            else:
                st.metric("Anos Analisados", "N/A")

        if 'Região' in df_limpo_atual.columns:
            st.markdown("<h3 class='section-header'>Distribuição por Região</h3>", unsafe_allow_html=True)
            fig_regiao = px.pie(df_limpo_atual, names='Região', title='Distribuição de Registros por Região')
            st.plotly_chart(fig_regiao, use_container_width=True)
        
        if 'Tipo de Resíduo' in df_limpo_atual.columns:
            st.markdown("<h3 class='section-header'>Distribuição por Tipo de Resíduo</h3>", unsafe_allow_html=True)
            fig_tipo = px.pie(df_limpo_atual, names='Tipo de Resíduo', title='Distribuição de Registros por Tipo de Resíduo')
            st.plotly_chart(fig_tipo, use_container_width=True)
        
        if 'Ano' in df_limpo_atual.columns:
            st.markdown("<h3 class='section-header'>Distribuição por Ano</h3>", unsafe_allow_html=True)
            fig_ano = px.histogram(df_limpo_atual, x='Ano', title='Distribuição de Registros por Ano').update_xaxes(type='category') # Tratar anos como categoria no histograma
            st.plotly_chart(fig_ano, use_container_width=True)
        
        st.markdown("<h3 class='section-header'>Download dos Dados</h3>", unsafe_allow_html=True)
        href_download = download_dataframe(df_limpo_atual, "dados_residuos_energia_limpos.xlsx")
        if href_download:
            st.markdown(href_download, unsafe_allow_html=True)
    
    elif pagina == "🧹 Limpeza e Qualidade dos Dados":
        st.markdown("<h2 class='sub-header'>Limpeza e Qualidade dos Dados</h2>", unsafe_allow_html=True)
        
        if df_original is not None:
            st.markdown("<h3 class='section-header'>Verificação de Integridade (Dados Originais)</h3>", unsafe_allow_html=True)
            valores_ausentes, duplicatas, tipos_dados = verificar_integridade(df_original)
            
            if valores_ausentes is not None and duplicatas is not None and tipos_dados is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valores Ausentes (Originais)", valores_ausentes.sum())
                with col2:
                    st.metric("Duplicatas (Originais)", duplicatas)
                
                st.markdown("<h3 class='section-header'>Tipos de Dados (Originais)</h3>", unsafe_allow_html=True)
                # --- CORREÇÃO APLICADA AQUI ---
                df_tipos_dados_para_exibir = pd.DataFrame({'Tipo': tipos_dados.astype(str)})
                st.dataframe(df_tipos_dados_para_exibir)
                # --- FIM DA CORREÇÃO ---
            else:
                st.warning("Não foi possível verificar a integridade dos dados originais.")

            st.markdown("<h3 class='section-header'>Alterações Realizadas na Limpeza</h3>", unsafe_allow_html=True)
            if st.session_state.alteracoes_limpeza:
                for alteracao in st.session_state.alteracoes_limpeza:
                    st.markdown(f"- {alteracao}")
            else:
                st.markdown("Nenhuma alteração foi documentada ou necessária durante a limpeza.")
            
            st.markdown("<h3 class='section-header'>Comparação Antes e Depois (Primeiras Linhas)</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dados Originais**")
                st.dataframe(df_original.head())
            with col2:
                st.markdown("**Dados Limpos**")
                st.dataframe(df_limpo_atual.head())
        else:
            st.warning("Dados originais não carregados. Não é possível mostrar a seção de limpeza.")

    elif pagina == "📈 Análise Exploratória":
        st.markdown("<h2 class='sub-header'>Análise Exploratória de Dados</h2>", unsafe_allow_html=True)
        
        df_filtrado_eda = df_limpo_atual[df_limpo_atual['Ano'] == ano_selecionado] if ano_selecionado is not None and 'Ano' in df_limpo_atual.columns else df_limpo_atual
        colunas_numericas_eda = df_filtrado_eda.select_dtypes(include=np.number).columns.tolist()

        if not colunas_numericas_eda:
            st.warning("Não há colunas numéricas nos dados filtrados para análise exploratória.")
        else:
            coluna_selecionada_eda = st.selectbox("Selecione uma coluna numérica para análise:", colunas_numericas_eda, key="eda_col_num")
            
            # Detecção de outliers
            st.markdown("<h3 class='section-header'>Detecção de Outliers</h3>", unsafe_allow_html=True)
            metodo_outlier = st.radio("Método de detecção:", ["IQR (Intervalo Interquartil)", "Z-Score"], key="eda_outlier_method")
            metodo_sel = 'iqr' if metodo_outlier == "IQR (Intervalo Interquartil)" else 'zscore'
            
            outliers, limite_inferior, limite_superior = detectar_outliers(df_filtrado_eda, coluna_selecionada_eda, metodo_sel)
            
            if outliers is not None:
                st.write(f"Foram encontrados {len(outliers)} outliers ({len(outliers)/len(df_filtrado_eda[coluna_selecionada_eda].dropna())*100:.2f}% dos dados válidos).")
                if metodo_sel == 'iqr' and limite_inferior is not None and limite_superior is not None:
                    st.write(f"Limite inferior (IQR): {limite_inferior:.2f}")
                    st.write(f"Limite superior (IQR): {limite_superior:.2f}")
                
                fig_outlier_box = px.box(df_filtrado_eda, y=coluna_selecionada_eda, title=f"Boxplot de {coluna_selecionada_eda} com Outliers")
                st.plotly_chart(fig_outlier_box, use_container_width=True)
                
                if not outliers.empty:
                    st.markdown("<div class='warning-box'>Primeiros outliers detectados:</div>", unsafe_allow_html=True)
                    st.dataframe(outliers.head(10))
            
            # Teste de normalidade
            st.markdown("<h3 class='section-header'>Teste de Normalidade</h3>", unsafe_allow_html=True)
            (stat_sw, p_sw), (stat_ks, p_ks) = testar_normalidade(df_filtrado_eda, coluna_selecionada_eda)
            
            col1_norm, col2_norm = st.columns(2)
            with col1_norm:
                if p_sw is not None:
                    st.metric("Shapiro-Wilk p-valor", f"{p_sw:.4f}")
                    st.write("Interpretação (SW): " + ("Distribuição NÃO normal" if p_sw < 0.05 else "Não se pode rejeitar normalidade"))
                else:
                    st.write("Teste Shapiro-Wilk não aplicável ou dados insuficientes.")
            with col2_norm:
                if p_ks is not None:
                    st.metric("Kolmogorov-Smirnov p-valor", f"{p_ks:.4f}")
                    st.write("Interpretação (KS): " + ("Distribuição NÃO normal" if p_ks < 0.05 else "Não se pode rejeitar normalidade"))
                else:
                    st.write("Teste Kolmogorov-Smirnov não aplicável ou dados insuficientes.")

            fig_hist_eda = px.histogram(df_filtrado_eda, x=coluna_selecionada_eda, title=f"Distribuição de {coluna_selecionada_eda}", marginal="box")
            st.plotly_chart(fig_hist_eda, use_container_width=True)
            
            # Testes não paramétricos
            st.markdown("<h3 class='section-header'>Análise Não Paramétrica (Comparação de Grupos)</h3>", unsafe_allow_html=True)
            colunas_categoricas_eda = df_filtrado_eda.select_dtypes(include='object').columns.tolist()
            if not colunas_categoricas_eda:
                st.info("Nenhuma coluna categórica disponível para agrupamento.")
            else:
                grupo_coluna_eda = st.selectbox("Selecione uma coluna categórica para agrupar:", colunas_categoricas_eda, key="eda_group_col")
                
                teste_nome, p_valor_teste = realizar_teste_nao_parametrico(df_filtrado_eda, coluna_selecionada_eda, grupo_coluna_eda)
                
                if teste_nome and p_valor_teste is not None:
                    st.write(f"Teste realizado: {teste_nome}")
                    st.metric(f"p-valor ({teste_nome})", f"{p_valor_teste:.4f}")
                    st.write("Interpretação: " + ("Há diferença significativa entre os grupos" if p_valor_teste < 0.05 else "Não há diferença significativa entre os grupos"))
                    
                    fig_box_grupo = px.box(df_filtrado_eda, x=grupo_coluna_eda, y=coluna_selecionada_eda, title=f"{coluna_selecionada_eda} por {grupo_coluna_eda}")
                    st.plotly_chart(fig_box_grupo, use_container_width=True)
            
            # Correlação
            st.markdown("<h3 class='section-header'>Mapa de Correlação entre Variáveis Numéricas</h3>", unsafe_allow_html=True)
            fig_corr_eda = criar_mapa_calor(df_filtrado_eda)
            if fig_corr_eda:
                st.plotly_chart(fig_corr_eda, use_container_width=True)

    elif pagina == "📉 Estatísticas Descritivas":
        st.markdown("<h2 class='sub-header'>Estatísticas Descritivas</h2>", unsafe_allow_html=True)
        
        estatisticas = calcular_estatisticas(df_limpo_atual, ano_selecionado)
        
        if estatisticas is not None and not estatisticas.empty:
            st.markdown(f"<h3 class='section-header'>Estatísticas para {ano_selecionado if ano_selecionado else 'Todos os Anos'}</h3>", unsafe_allow_html=True)
            st.dataframe(estatisticas)
            
            st.markdown("<h3 class='section-header'>Visualização Detalhada por Coluna</h3>", unsafe_allow_html=True)
            colunas_numericas_stats = estatisticas.index.tolist()
            coluna_sel_stats = st.selectbox("Selecione uma coluna para visualização detalhada:", colunas_numericas_stats, key="stats_col_detail")
            
            df_filtrado_stats = df_limpo_atual[df_limpo_atual['Ano'] == ano_selecionado] if ano_selecionado is not None and 'Ano' in df_limpo_atual.columns else df_limpo_atual
            
            if coluna_sel_stats in df_filtrado_stats.columns:
                fig_box_stats = px.box(df_filtrado_stats, y=coluna_sel_stats, title=f"Boxplot de {coluna_sel_stats}")
                st.plotly_chart(fig_box_stats, use_container_width=True)
                
                # Estatísticas detalhadas
                media = df_filtrado_stats[coluna_sel_stats].mean()
                mediana = df_filtrado_stats[coluna_sel_stats].median()
                desvio_padrao = df_filtrado_stats[coluna_sel_stats].std()
                variancia = df_filtrado_stats[coluna_sel_stats].var()
                min_val = df_filtrado_stats[coluna_sel_stats].min()
                max_val = df_filtrado_stats[coluna_sel_stats].max()
                q1 = df_filtrado_stats[coluna_sel_stats].quantile(0.25)
                q3 = df_filtrado_stats[coluna_sel_stats].quantile(0.75)
                iqr_val = q3 - q1

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Média", f"{media:.2f}" if not np.isnan(media) else "N/A")
                    st.metric("Mediana", f"{mediana:.2f}" if not np.isnan(mediana) else "N/A")
                with col_s2:
                    st.metric("Desvio Padrão", f"{desvio_padrao:.2f}" if not np.isnan(desvio_padrao) else "N/A")
                    st.metric("Variância", f"{variancia:.2f}" if not np.isnan(variancia) else "N/A")
                with col_s3:
                    st.metric("Mínimo", f"{min_val:.2f}" if not np.isnan(min_val) else "N/A")
                    st.metric("Máximo", f"{max_val:.2f}" if not np.isnan(max_val) else "N/A")
                
                st.markdown("<h4 class='section-header'>Quartis e IQR</h4>", unsafe_allow_html=True)
                col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                with col_q1: st.metric("Q1 (25%)", f"{q1:.2f}" if not np.isnan(q1) else "N/A")
                with col_q2: st.metric("Q2 (Mediana)", f"{mediana:.2f}" if not np.isnan(mediana) else "N/A") # Repetido, mas ok
                with col_q3: st.metric("Q3 (75%)", f"{q3:.2f}" if not np.isnan(q3) else "N/A")
                with col_q4: st.metric("IQR", f"{iqr_val:.2f}" if not np.isnan(iqr_val) else "N/A")
        else:
            st.info("Não há estatísticas para exibir para a seleção atual.")

    elif pagina == "🔍 Visualizações Interativas":
        st.markdown("<h2 class='sub-header'>Visualizações Interativas</h2>", unsafe_allow_html=True)
        df_vis = df_limpo_atual[df_limpo_atual['Ano'] == ano_selecionado] if ano_selecionado is not None and 'Ano' in df_limpo_atual.columns else df_limpo_atual

        # Boxplots
        st.markdown("<h3 class='section-header'>Boxplots por Tipo de Resíduo</h3>", unsafe_allow_html=True)
        bp_gerado, bp_reciclado, bp_perc = criar_boxplot(df_vis) # Passa df_vis já filtrado
        
        tabs_bp = st.tabs(["Resíduo Gerado", "Resíduo Reciclado", "% Reciclagem"])
        with tabs_bp[0]: 
            if bp_gerado: st.plotly_chart(bp_gerado, use_container_width=True)
            else: st.info("Dados insuficientes para Boxplot de Resíduo Gerado.")
        with tabs_bp[1]:
            if bp_reciclado: st.plotly_chart(bp_reciclado, use_container_width=True)
            else: st.info("Dados insuficientes para Boxplot de Resíduo Reciclado.")
        with tabs_bp[2]:
            if bp_perc: st.plotly_chart(bp_perc, use_container_width=True)
            else: st.info("Dados insuficientes para Boxplot de % Reciclagem.")

        # Histogramas
        st.markdown("<h3 class='section-header'>Histogramas de Distribuição</h3>", unsafe_allow_html=True)
        hist_gerado, hist_reciclado, hist_energia = criar_histogramas(df_vis) # Passa df_vis
        
        tabs_hist = st.tabs(["Resíduo Gerado", "Resíduo Reciclado", "Energia Gerada"])
        with tabs_hist[0]:
            if hist_gerado: st.plotly_chart(hist_gerado, use_container_width=True)
            else: st.info("Dados insuficientes para Histograma de Resíduo Gerado.")
        with tabs_hist[1]:
            if hist_reciclado: st.plotly_chart(hist_reciclado, use_container_width=True)
            else: st.info("Dados insuficientes para Histograma de Resíduo Reciclado.")
        with tabs_hist[2]:
            if hist_energia: st.plotly_chart(hist_energia, use_container_width=True)
            else: st.info("Dados insuficientes para Histograma de Energia Gerada.")

        # Gráficos de dispersão
        st.markdown("<h3 class='section-header'>Gráficos de Dispersão</h3>", unsafe_allow_html=True)
        scatter_energia_emissoes, scatter_gerado_reciclado = criar_scatter_plots(df_vis) # Passa df_vis
        
        tabs_scatter = st.tabs(["Energia vs Emissões", "Resíduo Gerado vs Reciclado"])
        with tabs_scatter[0]:
            if scatter_energia_emissoes: st.plotly_chart(scatter_energia_emissoes, use_container_width=True)
            else: st.info("Dados insuficientes para o gráfico de Energia vs Emissões.")
        with tabs_scatter[1]:
            if scatter_gerado_reciclado: st.plotly_chart(scatter_gerado_reciclado, use_container_width=True)
            else: st.info("Dados insuficientes para o gráfico de Resíduo Gerado vs Reciclado.")
            
    elif pagina == "📆 Análise Temporal":
        st.markdown("<h2 class='sub-header'>Análise Temporal</h2>", unsafe_allow_html=True)
        ts_res, ts_energia, ts_rec = criar_series_temporais(df_limpo_atual) # Usa o df completo

        if ts_res:
            st.markdown("<h3 class='section-header'>Evolução de Resíduos</h3>", unsafe_allow_html=True)
            st.plotly_chart(ts_res, use_container_width=True)
        if ts_energia:
            st.markdown("<h3 class='section-header'>Evolução de Energia e Emissões</h3>", unsafe_allow_html=True)
            st.plotly_chart(ts_energia, use_container_width=True)
        if ts_rec:
            st.markdown("<h3 class='section-header'>Evolução da % de Reciclagem</h3>", unsafe_allow_html=True)
            st.plotly_chart(ts_rec, use_container_width=True)
        if not any([ts_res, ts_energia, ts_rec]):
            st.info("Dados insuficientes para gerar análises temporais.")

    elif pagina == "🌎 Análise Regional":
        st.markdown("<h2 class='sub-header'>Análise Regional</h2>", unsafe_allow_html=True)
        df_reg = df_limpo_atual[df_limpo_atual['Ano'] == ano_selecionado] if ano_selecionado is not None and 'Ano' in df_limpo_atual.columns else df_limpo_atual
        reg_res, reg_rec, reg_energia = criar_analise_por_regiao(df_reg) # Passa df_reg filtrado

        if reg_res:
            st.markdown("<h3 class='section-header'>Resíduos por Região</h3>", unsafe_allow_html=True)
            st.plotly_chart(reg_res, use_container_width=True)
        if reg_rec:
            st.markdown("<h3 class='section-header'>% de Reciclagem por Região</h3>", unsafe_allow_html=True)
            st.plotly_chart(reg_rec, use_container_width=True)
        if reg_energia:
            st.markdown("<h3 class='section-header'>Energia Gerada por Região</h3>", unsafe_allow_html=True)
            st.plotly_chart(reg_energia, use_container_width=True)
        if not any([reg_res, reg_rec, reg_energia]):
            st.info("Dados insuficientes para gerar análises regionais.")

    elif pagina == "♻️ Análise por Tipo de Resíduo":
        st.markdown("<h2 class='sub-header'>Análise por Tipo de Resíduo</h2>", unsafe_allow_html=True)
        df_tipo_res = df_limpo_atual[df_limpo_atual['Ano'] == ano_selecionado] if ano_selecionado is not None and 'Ano' in df_limpo_atual.columns else df_limpo_atual
        tipo_gerado, tipo_rec, tipo_perc = criar_analise_por_tipo_residuo(df_tipo_res) # Passa df_tipo_res filtrado

        if tipo_gerado:
            st.markdown("<h3 class='section-header'>Resíduos Gerados por Tipo</h3>", unsafe_allow_html=True)
            st.plotly_chart(tipo_gerado, use_container_width=True)
        if tipo_rec:
            st.markdown("<h3 class='section-header'>Resíduos Reciclados por Tipo</h3>", unsafe_allow_html=True)
            st.plotly_chart(tipo_rec, use_container_width=True)
        if tipo_perc:
            st.markdown("<h3 class='section-header'>% de Reciclagem por Tipo</h3>", unsafe_allow_html=True)
            st.plotly_chart(tipo_perc, use_container_width=True)
        if not any([tipo_gerado, tipo_rec, tipo_perc]):
            st.info("Dados insuficientes para gerar análises por tipo de resíduo.")
            
    elif pagina == "💡 Insights e Recomendações":
        st.markdown("<h2 class='sub-header'>Insights e Recomendações</h2>", unsafe_allow_html=True)
        
        insights_rec = gerar_insight_reciclagem(df_limpo_atual)
        insights_en = gerar_insight_energia(df_limpo_atual)
        recom = gerar_recomendacoes(df_limpo_atual)

        st.markdown("<h3 class='section-header'>Insights sobre Reciclagem</h3>", unsafe_allow_html=True)
        if insights_rec:
            st.markdown(f"- Ano com maior taxa de reciclagem: **{insights_rec.get('ano_maior_reciclagem', 'N/A')}** ({insights_rec.get('maior_reciclagem', 0):.2f}%)")
            st.markdown(f"- Tipo de resíduo com maior taxa de reciclagem: **{insights_rec.get('tipo_maior_reciclagem', 'N/A')}** ({insights_rec.get('taxa_maior_reciclagem', 0):.2f}%)")
            st.markdown(f"- Potencial de reciclagem não aproveitado: **{insights_rec.get('potencial', 0):.2f} toneladas**")
            if 'df_anual' in insights_rec and '% Reciclagem' in insights_rec['df_anual'].columns:
                fig_evol_rec = px.line(insights_rec['df_anual'], x='Ano', y='% Reciclagem', markers=True, title='Evolução da % Média de Reciclagem')
                st.plotly_chart(fig_evol_rec, use_container_width=True)
        else:
            st.info("Não foi possível gerar insights sobre reciclagem.")

        st.markdown("<h3 class='section-header'>Insights sobre Energia</h3>", unsafe_allow_html=True)
        if insights_en:
            st.markdown(f"- Ano com maior geração de energia: **{insights_en.get('ano_maior_energia', 'N/A')}** ({insights_en.get('maior_energia', 0):.2f} kWh)")
            st.markdown(f"- Região que mais gera energia: **{insights_en.get('regiao_maior_energia', 'N/A')}**")
            st.markdown(f"- Região com maior eficiência de painéis (kWh/painel): **{insights_en.get('regiao_maior_eficiencia', 'N/A')}**")
            if 'df_anual' in insights_en and 'Energia Gerada (kWh)' in insights_en['df_anual'].columns:
                fig_evol_en = px.line(insights_en['df_anual'], x='Ano', y='Energia Gerada (kWh)', markers=True, title='Evolução da Energia Gerada')
                st.plotly_chart(fig_evol_en, use_container_width=True)
        else:
            st.info("Não foi possível gerar insights sobre energia.")

        st.markdown("<h3 class='section-header'>Recomendações Estratégicas</h3>", unsafe_allow_html=True)
        if recom:
            st.markdown("<h4>Para Melhoria da Reciclagem:</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='highlight'>- Foco no tipo de resíduo: <b>{recom.get('tipo_menor_reciclagem', 'N/A')}</b> (menor taxa de reciclagem).</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='highlight'>- Atenção à região: <b>{recom.get('regiao_menor_reciclagem', 'N/A')}</b> (menor taxa de reciclagem).</div>", unsafe_allow_html=True)
            st.markdown("<div class='highlight'>- Implementar políticas de incentivo e educação ambiental.</div>", unsafe_allow_html=True)
            
            st.markdown("<h4>Para Eficiência Energética:</h4>", unsafe_allow_html=True)
            st.markdown("<div class='highlight'>- Expandir painéis solares em regiões de alta eficiência.</div>", unsafe_allow_html=True)
            st.markdown("<div class='highlight'>- Investir em modernização tecnológica e integração de sistemas.</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='highlight'>- Tendência geral da eficiência energética (kWh/painel): <b>{recom.get('tendencia_eficiencia', 'Indeterminada')}</b>.</div>", unsafe_allow_html=True)
        else:
            st.info("Não foi possível gerar recomendações.")

        st.markdown("<h3 class='section-header'>Gerar Relatório</h3>", unsafe_allow_html=True)
        if st.button("Gerar Relatório em Markdown"):
            caminho_relatorio = gerar_relatorio_pdf(df_limpo_atual) # A função foi renomeada mentalmente para gerar_relatorio_markdown
            if caminho_relatorio and os.path.exists(caminho_relatorio):
                st.success(f"Relatório gerado com sucesso: {caminho_relatorio}")
                with open(caminho_relatorio, "r", encoding='utf-8') as f: # Adicionado encoding
                    st.download_button(
                        label="Baixar Relatório (.md)",
                        data=f.read(),
                        file_name=os.path.basename(caminho_relatorio), # usa o nome do arquivo gerado
                        mime="text/markdown",
                    )
            else:
                st.error("Falha ao gerar ou encontrar o relatório.")

else:
    if df_original is None: # Se o arquivo original não foi encontrado
        pass # A mensagem de erro já foi exibida no carregamento
    else: # Se df_limpo for None por outra razão após o carregamento
        st.error("Não foi possível processar os dados. Verifique o console para mais detalhes se houver erros de limpeza.")