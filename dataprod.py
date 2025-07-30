import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

import collections

import plotly.graph_objects as go

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import shap

def filtra_dados_uso_llm_empresa(dado):
    if pd.isna(dado):
        return "Não informado"
    elif ("IA Generativa e LLMs como principal frente do negócio" in dado):
        return "Principal frente"
    elif ("Não" in dado):
        return "Não sabe opinar"
    elif ("IA Generativa e LLMs não é prioridade" in dado):
        return "Não é prioridade"
    elif ("Direcionamento centralizado do uso de AI generativa" in dado):
        return "Uso centralizado"
    elif ("Desenvolvedores utilizando Copilots" in dado):
        return "Uso de copilots"
    elif("produtos internos" in dado):
        return "Produto interno"
    elif("produtos externos" in dado):
        return "Produto externo"
    else:
        return "Independente"


def filtra_dados_uso_llm_usuario(dado):
    if pd.isna(dado):
        return "Não informado"

    if "Não utilizo nenhum tipo de solução de IA Generativa para melhorar a produtividade no dia a dia." in dado:
        return "Não utiliza"
    if "Utilizo apenas soluções gratuitas" in dado:
        return "Usa soluções gratuitas"
    if "Utilizo soluções pagas de AI Generativa (como por exemplo ChatGPT plus, MidJourney etc) e pago do meu próprio bolso." in dado:
        return "Uso pago individual"
    if "Utilizo soluções pagas de AI Generativa (como por exemplo ChatGPT plus, MidJourney etc) e a empresa em que trabalho paga pela solução." in dado:
        return "Empresa paga"
    if "copilot" in dado:
        return "Uso de copilots"


def categoriza_linha_simplificada(row):
    for col in colunas_atuacao[102:110]:
        if (row.get(col) == 1):
            if ("IA Generativa e LLMs como principal frente do negócio" in col):
                return "Principal frente"
            elif ("Não" in col):
                return "Não sabe opinar"
            elif ("IA Generativa e LLMs não é prioridade" in col):
                return "Não é prioridade"
            elif ("Direcionamento centralizado do uso de AI generativa" in col):
                return "Uso centralizado"
            elif ("Desenvolvedores utilizando Copilots" in col):
                return "Uso de copilots"
            elif("produtos internos" in col):
                return "Produto interno"
            elif("produtos externos" in col):
                return "Produto externo"
            else:
                return "Independente"

def categoriza_prioridade(row):
    for col in colunas_atuacao[102:110]:
        if (row.get(col) == 1):
            if ("IA Generativa e LLMs como principal frente do negócio" in col):
                return "Utiliza"
            elif ("Não" in col):
                return "Não prioriza"
            elif ("IA Generativa e LLMs não é prioridade" in col):
                return "Não prioriza"
            elif ("Direcionamento centralizado do uso de AI generativa" in col):
                return "Utiliza"
            elif ("Desenvolvedores utilizando Copilots" in col):
                return "Utiliza"
            elif("produtos internos" in col):
                return "Utiliza"
            elif("produtos externos" in col):
                return "Utiliza"
            else:
                return "Utiliza"


def aproxima_salario(salario):
    if pd.isna(salario):
        return np.nan
    if "Menos de" in salario:
        return 900
    elif "Acima de" in salario:
        return 45000
    else:
        salario = salario.replace("de R$", "").replace(" a ", "").replace("/mês", " ").replace("R$", "")
        salario = salario.lstrip()
        salario = salario.replace(".", "").split()
        return np.random.randint(int(salario[0]), int(salario[1]))

st.set_page_config(page_title="The State of AI - 2025", layout="wide")

st.title("🤖 The State of AI - 2025")
st.markdown("Uma análise do impacto do uso de inteligência artificial nas carreiras brasileiras.")

@st.cache_data
def load_data():
    return pd.read_csv("resources/dataset-stateofdata-2024.csv")

data = load_data()

colunas_demograficas = [coluna for coluna in data.columns if coluna.startswith("1.")]

colunas_trabalho = [coluna for coluna in data.columns if coluna.startswith("2.")]

colunas_empresa = [coluna for coluna in data.columns if coluna.startswith("3.")]

colunas_atuacao = [coluna for coluna in data.columns if coluna.startswith("4.")]

colunas_objetivos = [coluna for coluna in data.columns if coluna.startswith("5.")]

colunas_rotina_de = [coluna for coluna in data.columns if coluna.startswith("6.")]

colunas_rotina_da = [coluna for coluna in data.columns if coluna.startswith("7.")]

colunas_rotina_ds = [coluna for coluna in data.columns if coluna.startswith("8.")]

colunasTot = colunas_demograficas + colunas_trabalho + colunas_empresa + colunas_atuacao + colunas_rotina_de + colunas_rotina_da + colunas_rotina_ds

dados_limpos = data.filter(colunasTot)

dados_limpos['categorias_ia'] = dados_limpos.apply(categoriza_linha_simplificada, axis=1)

dados_limpos['prioridade_ia'] = dados_limpos.apply(categoriza_prioridade, axis=1)

dados_limpos['categoria_uso_llm_individual'] = dados_limpos[
    '4.m_usa_chatgpt_ou_copilot_no_trabalho?'
].apply(filtra_dados_uso_llm_usuario)

colunas_finais_analise = ['prioridade_ia', 'categorias_ia', 'categoria_uso_llm_individual', '1.l_nivel_de_ensino', '2.f_cargo_atual', '2.h_faixa_salarial', "2.k_satisfeito_atualmente", "2.l_motivo_insatisfacao"] + colunas_trabalho[11:23]



aba1, aba2, aba3, aba4 = st.tabs([
    "💰 Impacto da IA nos Salários",
    "😊 Impacto da IA na Satisfação Profissional",
    "👨‍💻 Uso da IA por Cargo e Nível de Ensino",
    "🔮 Simulador de Satisfação com a Empresa com base no uso de IA."
])

with aba1:
    st.markdown("### Hipótese analisada: *profissionais que fazem uso de IA em seu cotidiano, apoiado por suas empresas " \
    "tem maiores salários que o restante dos usuários.*")

    st.markdown("Aqui estão algumas análises em relação a distribuição salarial pelo tipo de uso de IA.\n" \
    "O uso de IA foi analisado tanto individualmente quanto pelo uso em empresa para comparação e algumas métricas interessantes foram encontradas.")


    st.subheader("Distribuição Salarial por Tipo de Uso de IA")
    dados_analise_h1 = dados_limpos[colunas_finais_analise].copy()
    dados_analise_h1['salario_aproximado'] = dados_analise_h1['2.h_faixa_salarial'].apply(aproxima_salario)

    dados_finais_analise_h1 = dados_analise_h1[colunas_finais_analise + ['salario_aproximado']]
    dados_finais_analise_h1 = dados_finais_analise_h1[dados_finais_analise_h1['categorias_ia'] != "Nenhuma"]

    df_individual = dados_finais_analise_h1[[
    'categoria_uso_llm_individual', 'salario_aproximado'
    ]].rename(columns={'categoria_uso_llm_individual': 'Categoria'})

    df_individual['Fonte'] = 'Uso Individual'

    df_empresa = dados_finais_analise_h1[[
        'categorias_ia', 'salario_aproximado'
    ]].rename(columns={'categorias_ia': 'Categoria'})

    df_empresa['Fonte'] = 'Uso na Empresa'

    df_combinado = pd.concat([df_individual, df_empresa])

    df_individual = df_individual[df_individual['Categoria'] != 'Não informado']
    
    with st.expander("Distribuição Salarial por Tipo de Uso de IA", expanded=True):

        fig1 = px.violin(
            df_individual,
            x="Categoria",
            y="salario_aproximado",
            box=True,
            title="Uso de IA Individual x Salário",
            color_discrete_sequence=["#636EFA"]
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
        st.markdown("Aqui, podemos perceber que os indivíduos cujo a empresa paga pelas ferramentas possui " \
        "em média maiores salários, seguidos pelo que fazem uso pago individual. Por fim, nas últimas posições temos " \
        "os indivíduos que fazem uso de soluções gratuitas ou nem utilizam IA.")

        fig2 = px.violin(
            df_empresa,
            x="Categoria",
            y="salario_aproximado",
            box=True,
            title="Uso de IA na Empresa x Salário",
            color_discrete_sequence=["#EF553B"]
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("Aqui, podemos perceber que os indivíduos cujo a empresa faz uso centralizado de ferramentas de IA " \
        "em média maiores salários, seguidos pelo que fazem uso em principal frente e as que fazem uso de Copilots. Por fim, nas últimas posições temos " \
        "as empresas que não priorizam e as pessoas que não sabem informar sobre o uso de IA.")

    dados_media = dados_finais_analise_h1.groupby(['prioridade_ia', 'categorias_ia'], as_index=False).agg({
            'salario_aproximado': 'mean'
        }) 

    with st.expander("Distribuição Salarial por prioridade da Empresa no uso de IA", expanded=True ):

        fig3 = px.treemap(
            dados_media,
            path=['prioridade_ia', 'categorias_ia'],  # Hierarchy: root -> prioridade -> categoria
            values='salario_aproximado',
            color='prioridade_ia',  # Optional: color by prioridade
            title='Salário Aproximado pela Prioridade dada a IA'
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("Aqui, vemos que esse treemap valida o gráfico mostrado acima, em que os salários de empresas que priorizam a IA são maiores do que as que não priorizam."
                    "E a ordem demonstrada segue o demonstrativo do gráfico de prioridade de IA nas empresas x salário.")

with aba2:
    st.markdown("### Hipótese analisada: *profissionais que fazem uso de ferramentas de IA com incentivo de suas empresas "
    "apresentam maior satisfação em seus empregos.*")

    dados_h2 = dados_finais_analise_h1.dropna(subset=["2.k_satisfeito_atualmente"])

    proporcao_satisfeitos = (
    dados_h2.groupby('categorias_ia')["2.k_satisfeito_atualmente"]
    .mean()
    .sort_values(ascending=False)
    )

    proporcao_satisfeitos_individual = (
        dados_h2.groupby('categoria_uso_llm_individual')["2.k_satisfeito_atualmente"]
        .mean()
        .sort_values(ascending=False)
    )

    df_plot = proporcao_satisfeitos.reset_index()
    df_plot.columns = ['Categoria IA', 'Proporcao Satisfeitos']


    df_plot2 = proporcao_satisfeitos_individual.reset_index()
    df_plot2.columns = ['Categoria uso individual IA', 'Proporcao Satisfeitos']


    df_plot2 = df_plot2[df_plot2['Categoria uso individual IA'] != 'Não informado']

    with st.expander("Satisfação X Uso individual de IA", expanded=True):

        fig2 = px.bar(
            df_plot2,
            x='Categoria uso individual IA',
            y='Proporcao Satisfeitos',
            color='Categoria uso individual IA',
            title='Proporção de profissionais satisfeitos por categoria de uso individual de IA',
            labels={'Proporcao Satisfeitos': 'Proporção Satisfeitos'},
            text='Proporcao Satisfeitos'
        )

        fig2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig2.update_layout(yaxis_tickformat='.0%', uniformtext_minsize=8, uniformtext_mode='hide')

        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Satisfação X Uso de IA na empresa", expanded=True):
        fig = px.bar(
            df_plot,
            x='Categoria IA',
            y='Proporcao Satisfeitos',
            color='Categoria IA',
            title='Proporção de profissionais satisfeitos por categoria de uso de IA na empresa',
            labels={'Proporcao Satisfeitos': 'Proporção Satisfeitos'},
            text='Proporcao Satisfeitos'
        )

        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(yaxis_tickformat='.0%', uniformtext_minsize=8, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("Aqui, percebemos que as empresas que centralizam o uso de IA, junto de empresas que tem" \
        "IA em sua principal frente possuem os profissionais mais satisfeitos. As empresas que IA não é prioridade apresentam menos satisfação no geral, " \
        " e abaixo há uma análise dessa categoria e dos motivos dessa insatisfação.")

        dados_insatisfeitos = dados_h2.loc[dados_h2['categorias_ia'] == "Não é prioridade"]

        dados_uso_ai = dados_insatisfeitos.loc[
        ~dados_limpos['2.l_motivo_insatisfacao'].isna(),
        colunas_trabalho[11:22]
        ].sum().astype(int).sort_values()

        df_insatisfeitos = dados_uso_ai.reset_index()
        df_insatisfeitos.columns = ['Motivo', 'Contagem']

        df_insatisfeitos['Motivo'] = df_insatisfeitos['Motivo'].str.replace(r"^\d+\.[a-zA-Z0-9]+\.\d+_", "", regex=True)

        fig3 = px.bar(
            df_insatisfeitos,
            x='Contagem',
            y='Motivo',
            orientation='h',
            title='Motivos de insatisfação entre profissionais de dados',
            labels={'Motivo': 'Motivo de Insatisfação', 'Contagem': 'Número de Menções'},
            color='Contagem',
            color_continuous_scale='Reds'
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("Vemos aqui que a maioria dos profissionais cujo as empresas não priorizam ferramentas de IA, tem como principal motivo de " \
        " insatisfação menos oportunidades de crescimento e aprendizado. Além disso, salários baixos e maturidade tecnológica são outras categorias de reclamação.")
        
with aba3:
    st.subheader("Outras hipóteses sobre o uso de IA")

    dados_h3 = pd.crosstab(
    dados_finais_analise_h1['1.l_nivel_de_ensino'],
    dados_finais_analise_h1['categoria_uso_llm_individual'],
    normalize='index'
    )   

    dados_h3 = dados_h3.reset_index().melt(id_vars='1.l_nivel_de_ensino', var_name='Uso de IA', value_name='Proporção')
    
    with st.expander("**Hipótese 1**: Pessoas com menor nível de ensino fazem mais uso de ferramentas gratuitas de IA.", expanded=True):
        fig = px.bar(
            dados_h3,
            x='1.l_nivel_de_ensino',
            y='Proporção',
            color='Uso de IA',
            title='Tipo de uso de IA por nível de ensino',
            labels={'1.l_nivel_de_ensino': 'Nível de Ensino'},
            barmode='stack'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Vemos que a hipótese não foi validada, pois o uso de ferramentas gratuitas de IA generativa se encontra bem equilibrado entre"
        "as pessoas sem educação formal, graduandos e pós-graduandos. Vemos um aumento considerável no uso pago a partir de alunos de Mestrado e Doutorado.")

    tabela_uso_por_cargo = pd.crosstab(
    dados_limpos['2.f_cargo_atual'],
    dados_limpos['categoria_uso_llm_individual'],
    normalize='index' 
    )

    grafico_uso_ia_por_cargo = dados_finais_analise_h1.groupby(
        ['2.f_cargo_atual', 'categoria_uso_llm_individual']
    ).size().reset_index(name='Contagem')

    with st.expander("**Hipótese 2**: Entre as carreiras de dados, o uso de ferramentas de IA é extremamente equilibrado, sem carreiras que fazem mais uso ou menos.", expanded=True):
        fig2 = px.treemap(
            grafico_uso_ia_por_cargo,
            path=['2.f_cargo_atual', 'categoria_uso_llm_individual'],
            values='Contagem',
            color='2.f_cargo_atual',
            title='Tipo de uso de IA por cargo'
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("Vemos que, em proporção, o uso de ferramentas de IA gratuita é bem equilibrada, porém, algumas carreiras fazem mais uso de ferramentas gratuitas que outras. " \
        "Um exemplo são os Analistas de Dados, que usam mais ferramentas gratuitas em relação a outros cargos como Cientistas de Dados e Engenheiros de Dados." \
        "Porém, vemos que as soluções gratuitas imperam entre os demais usos.")

with aba4:
    st.title("🔮 Simulador de Satisfação com a Empresa")

    st.markdown("Com esse simulador, podemos ter um insight da satisfação de um profissional em relação a sua empresa. " \
    " Assim, gestores podem avaliar os insights apresentados tanto nos gráficos quanto neste simulador para ver a situação atual de seus empregados " \
    " e tomar melhores decisões em relação a seus negócios.")

    colunas_previsao_satisfacao = ["2.k_satisfeito_atualmente", '2.h_faixa_salarial', '2.f_cargo_atual', "categorias_ia", "prioridade_ia", "categoria_uso_llm_individual"]

    df_previsao_satisfacao = dados_limpos.filter(colunas_previsao_satisfacao)
    df_previsao_satisfacao = df_previsao_satisfacao.dropna()

    df_previsao_satisfacao["satisfeito"] = df_previsao_satisfacao["2.k_satisfeito_atualmente"].astype(int)

    X_cat = df_previsao_satisfacao[["categoria_uso_llm_individual", "categorias_ia", '2.h_faixa_salarial' ,'2.f_cargo_atual']]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X_cat)

    y = df_previsao_satisfacao["satisfeito"]

    feature_names = encoder.get_feature_names_out()

    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_encoded, y)

    st.header("📋 Perfil do Uso de AI do Profissional e da Empresa")

    cargo = st.selectbox("Cargo", sorted(df_previsao_satisfacao['2.f_cargo_atual'].unique()))
    uso_individual = st.selectbox("Tipo de uso individual da IA generativa", sorted(df_previsao_satisfacao["categoria_uso_llm_individual"].unique()))
    uso_empresa = st.selectbox("Tipo de uso de ferramentas de IA na empresa?", sorted(df_previsao_satisfacao["categorias_ia"].unique()))
    salario = st.selectbox("Qual o salário recebido pelo profissional?", sorted(df_previsao_satisfacao['2.h_faixa_salarial'].unique()))

    X_input_df = pd.DataFrame([[uso_individual, uso_empresa, salario, cargo]],
                          columns=["categoria_uso_llm_individual", "categorias_ia", '2.h_faixa_salarial' ,'2.f_cargo_atual'])
    X_input_encoded = encoder.transform(X_input_df)

    proba = clf.predict_proba(X_input_encoded)[0][1]

    st.subheader("🎯 Resultado")
    st.metric("Probabilidade de estar satisfeito", f"{proba*100:.1f}%")
    st.progress(int(proba*100), width="stretch")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_encoded)
    shap_user = explainer.shap_values(X_input_encoded)[0]
    idx = 0

    shap_values = explainer.shap_values(X_input_encoded)[idx]
    expected_value = explainer.expected_value

    features_sample = X_input_encoded[idx]

    better_names = {
        '2.h_faixa_salarial_Menos de R$ 1.000/mês': 'Salário < R$1.000',
        '2.h_faixa_salarial_de R$ 1.001/mês a R$ 2.000/mês': 'R$1.001 - R$2.000',
        '2.h_faixa_salarial_de R$ 2.001/mês a R$ 3.000/mês': 'R$2.001 - R$3.000',
        '2.h_faixa_salarial_de R$ 3.001/mês a R$ 4.000/mês': 'R$3.001 - R$4.000',
        '2.h_faixa_salarial_de R$ 4.001/mês a R$ 6.000/mês': 'R$4.001 - R$6.000',
        '2.h_faixa_salarial_de R$ 6.001/mês a R$ 8.000/mês': 'R$6.001 - R$8.000',
        '2.h_faixa_salarial_de R$ 8.001/mês a R$ 12.000/mês': 'R$8.001 - R$12.000',
        '2.h_faixa_salarial_de R$ 12.001/mês a R$ 16.000/mês': 'R$12.001 - R$16.000',
        '2.h_faixa_salarial_de R$ 16.001/mês a R$ 20.000/mês': 'R$16.001 - R$20.000',
        '2.h_faixa_salarial_de R$ 20.001/mês a R$ 25.000/mês': 'R$20.001 - R$25.000',
        '2.h_faixa_salarial_de R$ 25.001/mês a R$ 30.000/mês': 'R$25.001 - R$30.000',
        '2.h_faixa_salarial_de R$ 30.001/mês a R$ 40.000/mês': 'R$30.001 - R$40.000',
        '2.h_faixa_salarial_Acima de R$ 40.001/mês': 'Salário > R$40.000',
        '2.f_cargo_atual_Analista de Dados/Data Analyst': 'Cargo: Analista de Dados',
        '2.f_cargo_atual_Analista de BI/BI Analyst': 'Cargo: Analista de BI',
        '2.f_cargo_atual_Analista de Negócios/Business Analyst': 'Cargo: Analista de Negócios',
        '2.f_cargo_atual_Analista de Suporte/Analista Técnico': 'Cargo: Suporte Técnico',
        '2.f_cargo_atual_Desenvolvedor/ Engenheiro de Software/ Analista de Sistemas': 'Cargo: Eng. de Software',
        '2.f_cargo_atual_Engenheiro de Dados/Data Engineer/Data Architect': 'Cargo: Eng. de Dados',
        '2.f_cargo_atual_Engenheiro de Machine Learning/ML Engineer/AI Engineer': 'Cargo: Eng. de ML/IA',
        '2.f_cargo_atual_Arquiteto de Dados/Data Architect': 'Cargo: Arquiteto de Dados',
        '2.f_cargo_atual_Analytics Engineer': 'Cargo: Eng. de Analytics',
        '2.f_cargo_atual_Cientista de Dados/Data Scientist': 'Cargo: Cientista de Dados',
        '2.f_cargo_atual_Professor/Pesquisador': 'Cargo: Professor/Pesquisador',
        '2.f_cargo_atual_Data Product Manager/ Product Manager (PM/APM/DPM/GPM/PO)': 'Cargo: Product Manager',
        '2.f_cargo_atual_Outras Engenharias (não inclui dev)': 'Cargo: Outra Engenharia',
        '2.f_cargo_atual_Estatístico': 'Cargo: Estatístico',
        '2.f_cargo_atual_Outra Opção': 'Cargo: Outro',
        "categoria_uso_llm_individual_Usa soluções gratuitas": "LLM individual: Gratuito",
        "categoria_uso_llm_individual_Empresa paga": "LLM individual: Empresa paga",
        "categoria_uso_llm_individual_Uso pago individual": "LLM individual: Uso pago individual",
        "categoria_uso_llm_individual_Não utiliza": "LLM individual: Não utiliza",
        'categorias_ia_Independente': 'IA: Uso independente',
        'categorias_ia_Não sabe opinar': 'IA: Não sabe opinar',
        'categorias_ia_Não é prioridade': 'IA: Não prioritária',
        'categorias_ia_Uso centralizado': 'IA: Uso centralizado',
        'categorias_ia_Uso de copilots': 'IA: Uso de copilots',
        'categorias_ia_Produto interno': 'IA: Produto interno',
        'categorias_ia_Produto externo': 'IA: Produto externo',
        'categorias_ia_Principal frente': 'IA: Principal frente',
        'categorias_ia_None': 'IA: Não informado'
    }
    
    st.header("📊 Importância Geral das Variáveis (SHAP)")

    with st.spinner("Calculando SHAP para amostra..."):
        sample_size = min(300, len(X_encoded))
        shap_sample = X_encoded[:sample_size]
        shap_values_sample = explainer.shap_values(shap_sample)
        if isinstance(shap_values_sample, list):  # binary classifier
            shap_values_sample = shap_values_sample[1]

        shap_df = pd.DataFrame(shap_values_sample, columns=feature_names)
        shap_df.rename(columns=better_names, inplace=True)

        mean_shap = shap_df.mean()
        mean_abs_shap = shap_df.abs().mean()

        shap_dict = dict(zip(mean_abs_shap.index, zip(mean_abs_shap, mean_shap)))
        
        shap_dict = collections.OrderedDict(sorted(shap_dict.items(), key=lambda x: x[1][0], reverse=True))

        features = list(shap_dict.keys())[:5]
        abs_values = [shap_dict[f][0] for f in features]
        mean_values = [shap_dict[f][1] for f in features]
        
        colors = ['red' if val < 0 else 'green' for val in mean_values]

        fig = px.bar(
            x=abs_values,
            y=features,
            orientation='h',
            labels={'x': 'Impacto médio absoluto (SHAP)', 'y': 'Variável'},
            title='🔍 Importância das Variáveis segundo o modelo (SHAP)',
        )

        fig.update_traces(
            marker_color=colors,
            textposition='auto'
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_tickformat=".3f",
            xaxis=dict(showgrid=True),
            plot_bgcolor='rgba(0,0,0,0)',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### LEGENDA: ")
        st.markdown("   🔴 Impacto negativo na satisfação")
        st.markdown("   🟢 Impacto positivo na satisfação")

        st.markdown("Fazendo uso do método SHAP, utilizado para medir a importância das variáveis em um modelo de classificação"
                    " podemos perceber que certos atributos como a não priorização de ferramentas de IA em uma empresa tem impacto negativo na satisfação" \
                    " de seus funcionários até maior que os baixos salários. ")