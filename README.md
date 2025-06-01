# ♻️ Dashboard Interativo para Análise de Resíduos e Energia

## 📝 Descrição

Este projeto apresenta um dashboard interativo construído com Streamlit para a análise de dados sobre gestão de resíduos e geração de energia. O objetivo é demonstrar um ciclo completo de análise de dados, desde o carregamento e limpeza até a exploração, visualização interativa, geração de insights e recomendações estratégicas.

O dashboard permite explorar dados como quantidade de resíduos gerados e reciclados, percentual de reciclagem, energia gerada (kWh), emissões evitadas (kg CO₂), e outros indicadores, filtrados por ano, região e tipo de resíduo.

⚠️ **Observação:** O projeto foi desenvolvido para carregar um arquivo chamado `base_residuos_energia_nome.xlsx` localizado no mesmo diretório do script `app.py`.

## 🚀 Processo de Análise de Dados (Passo a Passo)

O dashboard está estruturado para seguir as etapas clássicas de um projeto de análise de dados:

1.  📊 **Visão Geral dos Dados**:
    * Carregamento dos dados de um arquivo Excel.
    * Exibição das primeiras linhas e informações gerais como total de registros, colunas e período analisado.
    * Distribuição inicial por região, tipo de resíduo e ano.
    * Opção de download dos dados limpos.

2.  🧹 **Limpeza e Qualidade dos Dados**:
    * Verificação de integridade dos dados originais (valores ausentes, duplicatas, tipos de dados).
    * Aplicação de processos de limpeza, como preenchimento de valores ausentes (média para numéricos, moda para categóricos) e remoção de duplicatas.
    * Documentação das alterações realizadas.
    * Comparação entre os dados originais e limpos.

3.  📈 **Análise Exploratória (EDA)**:
    * Detecção de outliers utilizando métodos como IQR e Z-Score para colunas numéricas selecionadas.
    * Testes de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov).
    * Análise não paramétrica para comparação de grupos (Mann-Whitney U, Kruskal-Wallis).
    * Criação de mapa de correlação entre variáveis numéricas.

4.  📉 **Estatísticas Descritivas**:
    * Cálculo e exibição de estatísticas descritivas (média, mediana, desvio padrão, quartis, IQR, assimetria, curtose) para as variáveis numéricas, com opção de filtro por ano.
    * Visualização detalhada (boxplots) e métricas específicas para colunas selecionadas.

5.  🔍 **Visualizações Interativas**:
    * **Boxplots**: Distribuição de resíduos gerados, reciclados e percentual de reciclagem por tipo de resíduo.
    * **Histogramas**: Distribuição de resíduos gerados, reciclados e energia gerada.
    * **Gráficos de Dispersão**: Relação entre energia gerada vs. emissões evitadas e resíduo gerado vs. reciclado.

6.  📆 **Análise Temporal**:
    * Gráficos de séries temporais mostrando a evolução de resíduos gerados/reciclados, energia gerada/emissões evitadas e percentual de reciclagem ao longo dos anos.

7.  🌎 **Análise Regional**:
    * Gráficos de barras comparando resíduos gerados/reciclados, percentual de reciclagem e energia gerada entre diferentes regiões, com filtro opcional por ano.

8.  ♻️ **Análise por Tipo de Resíduo**:
    * Gráficos de pizza e barras mostrando a distribuição de resíduos gerados, reciclados e o percentual de reciclagem para cada tipo de resíduo, com filtro opcional por ano.

9.  💡 **Insights e Recomendações**:
    * Geração automática de insights chave sobre reciclagem (ex: ano com maior taxa, tipo de resíduo mais reciclado, potencial não aproveitado).
    * Geração automática de insights chave sobre energia (ex: ano com maior geração, região mais produtora/eficiente).
    * Sugestão de recomendações estratégicas para melhoria da reciclagem e eficiência energética.
    * Opção de gerar e baixar um relatório resumido em formato Markdown.

## ✨ Principais Análises e Funcionalidades

* **Análise Descritiva e Exploratória Completa**: Compreensão profunda da distribuição e características dos dados.
* **Detecção de Outliers e Testes Estatísticos**: Avaliação da normalidade e comparação entre grupos.
* **Visualizações Multidimensionais**: Gráficos interativos para analisar os dados sob diversas perspectivas (temporal, regional, por tipo de resíduo).
* **Identificação de Correlações**: Mapa de calor para entender as relações entre variáveis.
* **Geração de Insights Automatizados**: Sumarização dos principais achados sobre reciclagem e energia.
* **Recomendações Estratégicas**: Sugestões baseadas nos dados para otimizar a gestão de resíduos e a produção de energia.
* **Relatórios Dinâmicos**: Capacidade de filtrar dados por ano e gerar um relatório resumido.

## 💡 Principais Insights Potenciais (Exemplos)

* Identificação do ano com pico de reciclagem ou geração de energia.
* Descoberta de qual tipo de resíduo possui a maior/menor taxa de reciclagem.
* Comparação da eficiência na gestão de resíduos e geração de energia entre diferentes regiões.
* Análise de tendências ao longo do tempo para prever futuras demandas ou desafios.
* Quantificação do potencial de reciclagem não aproveitado.
* Correlação entre a quantidade de painéis solares instalados e a energia gerada ou emissões evitadas.

## 🛠️ Tecnologias Utilizadas

* 🐍 **Python**: Linguagem principal para desenvolvimento.
* 🌐 **Streamlit**: Para a criação do dashboard interativo e da interface web.
* 🐼 **Pandas**: Para manipulação e análise de dados tabulares.
* 🔢 **NumPy**: Para operações numéricas eficientes.
* 📊 **Plotly (Express & Graph Objects)**: Para a criação de gráficos interativos e visualizações de dados.
* 🔬 **SciPy**: Para testes estatísticos (Shapiro-Wilk, Kolmogorov-Smirnov, Mann-Whitney U, Kruskal-Wallis).
* 🗓️ **Os, Base64, BytesIO**: Para manipulação de arquivos e funcionalidades de download.


## 🗺️ Navegando no Dashboard

* Utilize a **Barra Lateral** para aplicar filtros (como seleção de ano) e para navegar entre as diferentes seções de análise.
* Interaja com os gráficos: passe o mouse para ver detalhes, clique em legendas para filtrar séries, utilize as abas dentro das seções para diferentes visualizações.
* Na seção "Visão Geral dos Dados", você pode baixar uma cópia dos dados limpos.
* Na seção "Insights e Recomendações", você pode gerar e baixar um relatório em formato Markdown.

