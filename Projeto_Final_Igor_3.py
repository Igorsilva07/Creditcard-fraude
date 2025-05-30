import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import shap # Importa a biblioteca SHAP
import matplotlib.pyplot as plt # Para SHAP plots que usam matplotlib (waterfall, summary)

# --- 1. CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="Detector e Explicador de Fraudes",
    page_icon="🔎",
    layout="wide", # Usar layout wide para mais espaço para os gráficos SHAP
    initial_sidebar_state="auto"
)

# --- INICIALIZAÇÃO DO JAVASCRIPT DO SHAP (MOVIDO PARA O TOPO) ---
shap.initjs()

# --- 2. CARREGAMENTO DE RECURSOS (MODELO E SCALER) COM CACHE ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('best_rf_model.pkl')
        scaler_obj = joblib.load('scaler_time_amount.pkl')
        return model, scaler_obj
    except FileNotFoundError:
        st.error("Ops! Não consegui encontrar os arquivos do modelo ('best_rf_model.pkl') ou do 'scaler' ('scaler_time_amount.pkl').")
        st.info("Por favor, garanta que esses arquivos estão na mesma pasta onde você está rodando o Streamlit.")
        st.stop()
    except Exception as e:
        st.error(f"Aconteceu um erro inesperado ao carregar os arquivos: {e}. Verifique se eles não estão corrompidos.")
        st.stop()

# Carrega os recursos (apenas uma vez)
best_rf_model, scaler = load_resources()

# Obter nomes das features do modelo
if hasattr(best_rf_model, 'feature_names_in_'):
    model_features = list(best_rf_model.feature_names_in_)
else:
    # Fallback caso feature_names_in_ não esteja disponível no modelo salvo
    # Assume a ordem Time, V1-V28, Amount
    model_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    st.warning("Atenção: Não foi possível obter o nome das informações (features) diretamente do modelo. Estou usando uma ordem padrão, o que pode não ser ideal.")

# --- TÍTULO E DESCRIÇÃO PRINCIPAL ---
st.title("🔎 Desvende as Fraudes: Analisador Inteligente de Transações")
st.markdown("""
    Bem-vindo! Este aplicativo te ajuda a **entender por que** uma transação financeira é classificada como **fraude** ou **legítima** pelo nosso sistema de inteligência artificial.

    Você pode **criar sua própria transação** ou **usar um exemplo** para ver a explicação em tempo real.
""")

# --- BARRA LATERAL PARA INFORMAÇÕES E DIAGNÓSTICO ---
st.sidebar.title("Sobre este Analisador")
st.sidebar.info(
    "Aqui você pode explorar as decisões de um **modelo de Inteligência Artificial** "
    "(um tipo de Random Forest) que foi treinado para **detectar fraudes financeiras**. "
    "A ideia é que você veja não só o resultado (fraude ou não fraude), mas também **o que levou a essa decisão**, "
    "usando uma técnica chamada **SHAP** para 'abrir a caixa preta' do modelo."
)

with st.sidebar.expander("Verificar arquivos internos"):
    current_dir_app = os.getcwd()
    st.write(f"Pasta onde o programa está rodando: `{current_dir_app}`")
    files_in_dir = os.listdir(current_dir_app)
    st.write("Arquivos encontrados nesta pasta:")
    st.dataframe(pd.DataFrame(files_in_dir, columns=["Nome do Arquivo"]))
    model_exists_diag = 'best_rf_model.pkl' in files_in_dir
    scaler_exists_diag = 'scaler_time_amount.pkl' in files_in_dir
    if model_exists_diag and scaler_exists_diag:
        st.success("✅ Arquivos importantes do modelo encontrados e prontos!")
    else:
        st.error("❌ Atenção: Arquivos importantes do modelo NÃO encontrados! O aplicativo pode não funcionar corretamente.")

# --- DEFINIÇÃO DE EXEMPLOS DE TRANSAÇÕES ---
fraud_example_1 = {
    'Time': 406.0, 'Amount': 0.0,
    'V1': -2.312226542, 'V2': 1.951992011, 'V3': -1.609850732, 'V4': 3.997905595,
    'V5': -0.522187865, 'V6': -1.426545319, 'V7': -2.537387306, 'V8': 1.391657371,
    'V9': -2.770089277, 'V10': -2.772272145, 'V11': 3.094970669, 'V12': -5.900423984,
    'V13': -0.427306899, 'V14': -6.137725328, 'V15': -0.313490212, 'V16': -3.111693797,
    'V17': -6.446485044, 'V18': -0.561394747, 'V19': -1.542456488, 'V20': 0.0,
    'V21': 0.517232371, 'V22': -0.035049369, 'V23': -0.465211075, 'V24': 0.320198199,
    'V25': 0.044519522, 'V26': 0.177839798, 'V27': 0.261145003, 'V28': -0.143275767
}
legit_example_1 = {
    'Time': 123456.0, 'Amount': 25.50,
    'V1': 0.00, 'V2': 0.00, 'V3': 0.00, 'V4': 0.00, 'V5': 0.00, 'V6': 0.00, 'V7': 0.00,
    'V8': 0.00, 'V9': 0.00, 'V10': 0.00, 'V11': 0.00, 'V12': 0.00, 'V13': 0.00, 'V14': 0.00,
    'V15': 0.00, 'V16': 0.00, 'V17': 0.00, 'V18': 0.00, 'V19': 0.00, 'V20': 0.00, 'V21': 0.00,
    'V22': 0.00, 'V23': 0.00, 'V24': 0.00, 'V25': 0.00, 'V26': 0.00, 'V27': 0.00, 'V28': 0.00
}
example_transactions = {
    "Exemplo de Fraude (Real)": fraud_example_1,
    "Exemplo Legítimo (V's ≈ 0)": legit_example_1,
}

# --- SELEÇÃO DO TIPO DE ENTRADA ---
st.header("1. Como Vamos Analisar a Transação?")
input_method = st.radio(
    "Escolha uma opção:",
    ("Digitar os dados da transação", "Carregar um exemplo pronto")
)

current_transaction_values = {}

if input_method == "Digitar os dados da transação":
    st.info("Insira os valores da transação. Entenda o que são as 'Features V' abaixo:")
    st.markdown("""
        ### **O que são as 'Features V' (V1, V2, ..., V28)?**

        Imagine que os dados de uma transação (como o número do cartão, a conta, etc.) são muito sensíveis e não podem ser mostrados diretamente. Para proteger sua privacidade, esses dados foram **transformados em códigos numéricos complexos** que chamamos de **'Features V'** (V1 a V28).

        Pense nelas como **"digitais" ou "impressões digitais" codificadas de uma transação**. Você não consegue saber exatamente qual é a conta ou o valor original, mas o modelo consegue identificar **padrões** nesses códigos.

        **Por que são importantes?**
        * **Privacidade:** Protegem suas informações reais.
        * **Detecção de Padrões Ocultos:** É nesses códigos que o modelo de fraude encontra os **sinais mais fortes e "estranhos"** que geralmente indicam uma fraude. Valores muito altos ou muito baixos, ou combinações específicas nessas "Features V", são como **luzes de alerta** para o nosso sistema.

        **Se você digitar zeros para as 'Features V'** (V1 a V28), o modelo não terá esses "sinais de alerta" codificados. Por isso, ele provavelmente classificará a transação como **legítima**, pois não verá nada fora do padrão nas "digitais" da transação.
    """)

    col_man1, col_man2 = st.columns(2)
    with col_man1:
        current_transaction_values['Time'] = st.number_input(
            "⏱️ Tempo (segundos) desde a 1ª Transação:",
            min_value=0.0, value=1000.0, help="Quanto tempo, em segundos, passou desde a primeira transação registrada no sistema."
        )
    with col_man2:
        current_transaction_values['Amount'] = st.number_input(
            "💰 Valor da Transação:",
            min_value=0.0, value=50.0, format="%.2f", help="O valor em dinheiro da transação (ex: R$ 50,00)."
        )

    for i in range(1, 29):
        current_transaction_values[f'V{i}'] = 0.0 # Define um valor padrão para as features V

    st.markdown("---")
    st.subheader("Ajuste as 'Features V' Chave (Opcional - Valores mais comuns ficam entre -5.0 e 5.0)")
    st.markdown("Se quiser simular uma fraude, tente ajustar esses valores. Valores muito 'estranhos' aqui podem indicar uma transação fraudulenta para o modelo:")

    key_v_features = ['V10', 'V12', 'V14', 'V16', 'V17', 'V4', 'V11']
    key_v_features = [f for f in key_v_features if f in model_features]

    cols_v_sliders = st.columns(3)
    for i, feature in enumerate(key_v_features):
        with cols_v_sliders[i % 3]:
            current_transaction_values[feature] = st.slider(
                f"{feature}:",
                min_value=-10.0, max_value=10.0,
                value=float(current_transaction_values.get(feature, 0.0)),
                step=0.01
            )

elif input_method == "Carregar um exemplo pronto":
    selected_example_name = st.selectbox(
        "Selecione um exemplo de transação para analisar:",
        list(example_transactions.keys())
    )
    current_transaction_values = example_transactions[selected_example_name].copy()
    st.info(f"O exemplo '{selected_example_name}' foi carregado. Clique em 'Analisar Transação' para ver o que o modelo diz!")

# --- PREPARAÇÃO PARA PREDIÇÃO ---
input_df_raw = pd.DataFrame([current_transaction_values])
if model_features:
    input_df_raw = input_df_raw.reindex(columns=model_features, fill_value=0.0)

with st.expander("Ver todos os dados da transação (versão técnica)"):
    st.dataframe(input_df_raw)

# --- BOTÃO PARA ANALISAR ---
st.markdown("---")
if st.button("🚀 Analisar Esta Transação", type="primary"):
    if not input_df_raw.empty:
        with st.spinner("Analisando a transação e pensando na explicação..."):
            time.sleep(1)

            time_amount_to_scale = input_df_raw[['Time', 'Amount']]
            scaled_time_amount = scaler.transform(time_amount_to_scale)

            input_df_processed = input_df_raw.copy()
            input_df_processed[['Time', 'Amount']] = scaled_time_amount
            if model_features:
                input_df_processed = input_df_processed[model_features]

            prediction = best_rf_model.predict(input_df_processed)
            prediction_proba = best_rf_model.predict_proba(input_df_processed)[:, 1] # Probabilidade da classe 1 (fraude)

            st.markdown("---")
            st.header("2. O Veredito do Modelo")

            col_pred_res, col_prob_res = st.columns(2)
            with col_pred_res:
                if prediction[0] == 1:
                    st.error("### 🚨 Previsão: FRAUDE DETECTADA!")
                    st.write("Nosso sistema de IA classificou esta transação como **fraudulenta**. Isso indica um alto risco!")
                else:
                    st.success("### ✅ Previsão: TRANSAÇÃO LEGÍTIMA")
                    st.write("Nosso sistema de IA classificou esta transação como **legítima**. Parece tudo certo aqui.")

            with col_prob_res:
                st.metric(label="Chance de ser Fraude", value=f"{prediction_proba[0]*100:.2f}%")
                st.progress(float(prediction_proba[0]))
                st.write(f"Uma chance de {prediction_proba[0]*100:.2f}% significa que o modelo tem essa confiança na sua classificação.")

            st.markdown("---")
            st.header("3. Entenda POR QUE o Modelo Decidiu Isso")
            st.markdown("""
                ### A "Caixa Preta" da Inteligência Artificial? Não Aqui!

                Modelos de Inteligência Artificial podem parecer complexos, como uma "caixa preta" que toma decisões sem explicar o motivo. Mas quando a decisão é importante, como detectar uma fraude, é fundamental entender **por que** o modelo chegou àquela conclusão.

                É aí que entra o **SHAP**!

                **O que o SHAP faz?**
                O SHAP é uma ferramenta poderosa que funciona como um **investigador** da decisão do modelo. Ele examina cada informação da sua transação (o `Tempo`, o `Valor` e as `Features V`) e revela **o quanto cada uma delas contribuiu** para a previsão final do modelo. Ele mostra o "peso" de cada dado na balança da decisão.

                **Como funciona a explicação no gráfico?**
                Imagine que cada informação da transação é como um "empurrão" na direção da decisão final:
                * **Barras VERMELHAS (para a direita):** Significam que aquela informação específica **aumentou a chance** da transação ser considerada uma **FRAUDE**. Quanto mais longa a barra vermelha, maior foi a contribuição para a suspeita de fraude.
                * **Barras AZUIS (para a esquerda):** Significam que aquela informação **diminuiu a chance** de ser uma **FRAUDE** (ou seja, ela aponta para uma transação **legítima**). Quanto mais longa a barra azul, mais forte foi a indicação de que a transação é segura.

                No gráfico, você verá:
                * A **linha base** (`E[f(x)]`) é a chance *média* de fraude em todas as transações que o modelo já analisou. Pense nela como o ponto de partida ou o "normal".
                * A **seta final** (`f(x)`) é a chance de fraude **desta transação específica**, após considerar todos os "empurrões" (influências) das informações. É o resultado final que o SHAP nos ajuda a desvendar!
            """)

            explainer = shap.TreeExplainer(best_rf_model)
            raw_shap_output = explainer.shap_values(input_df_processed)
            raw_expected_output = explainer.expected_value

            final_shap_values_for_plot = None
            final_base_value_for_plot = None

            # --- Lógica de extração de SHAP Values e Expected Value ---
            if isinstance(raw_shap_output, list) and len(raw_shap_output) == 2:
                final_shap_values_for_plot = raw_shap_output[1][0]
                final_base_value_for_plot = raw_expected_output[1]
            elif isinstance(raw_shap_output, np.ndarray) and raw_shap_output.ndim == 3 and raw_shap_output.shape[2] == 2:
                final_shap_values_for_plot = raw_shap_output[0, :, 1]
                final_base_value_for_plot = raw_expected_output[1]
            elif isinstance(raw_shap_output, np.ndarray) and raw_shap_output.ndim == 2 and raw_shap_output.shape[0] == 1:
                final_shap_values_for_plot = raw_shap_output[0]
                final_base_value_for_plot = raw_expected_output
            else:
                st.error(f"Erro: Não consegui entender o formato dos dados de explicação (SHAP). Por favor, avise quem desenvolveu o app. Detalhes técnicos: {type(raw_shap_output)} - {raw_shap_output.shape if hasattr(raw_shap_output, 'shape') else 'Sem shape'}.")
                st.stop()

            if isinstance(final_base_value_for_plot, np.ndarray) and np.size(final_base_value_for_plot) == 1:
                final_base_value_for_plot = final_base_value_for_plot.item()
            elif isinstance(final_base_value_for_plot, list) and len(final_base_value_for_plot) == 1:
                final_base_value_for_plot = final_base_value_for_plot[0]

            if not isinstance(final_base_value_for_plot, (int, float)):
                st.error(f"Erro: O valor base para a explicação não é um número. Detalhes técnicos: {final_base_value_for_plot} (tipo {type(final_base_value_for_plot)}).")
                st.stop()

            if final_shap_values_for_plot is None:
                st.error("Erro: Os valores de explicação (SHAP) não foram gerados corretamente. Por favor, avise quem desenvolveu o app.")
                st.stop()
            # --- Fim da lógica de extração de SHAP Values e Expected Value ---


            # =====================================================================
            # INÍCIO DO BLOCO SHAP PARA WATERFALL PLOT (Matplotlib-based)
            # =====================================================================
            st.subheader("Visualização dos Principais Fatores que Influenciam a Decisão")

            try:
                # Criar o objeto Explanation para o waterfall plot
                explanation_for_waterfall = shap.Explanation(
                    values=final_shap_values_for_plot,
                    base_values=final_base_value_for_plot,
                    data=input_df_processed.iloc[0],
                    feature_names=model_features
                )

                # Criar a figura e eixos para o Matplotlib com tamanho ajustado
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(7, 4)) # Tamanho ajustado aqui!

                # Passar o objeto Explanation para a função waterfall
                shap.plots.waterfall(explanation_for_waterfall, show=False)

                plt.tight_layout()
                st.pyplot(fig_waterfall) # Exibe o plot no Streamlit
                plt.close(fig_waterfall) # Libera a memória da figura para evitar avisos do Streamlit
            except Exception as e_waterfall_plot:
                st.error(f"Não foi possível gerar o gráfico de explicação. Erro: {e_waterfall_plot}")
                st.write("Se vir este erro, por favor, tire um print e mostre ao desenvolvedor do aplicativo.")
                # Detalhes técnicos para depuração:
                # st.write(f"  - Base Value (final): {final_base_value_for_plot} (Tipo: {type(final_base_value_for_plot)})")
                # st.write(f"  - SHAP Values (final) shape: {final_shap_values_for_plot.shape if final_shap_values_for_plot is not None else 'None'}")
                # st.write(f"  - Feature Values (primeira linha processada) shape: {input_df_processed.iloc[0].shape}")
            # =====================================================================
            # FIM DO BLOCO SHAP PARA WATERFALL PLOT
            # =====================================================================

            st.info("Brinque com os valores de Tempo, Valor e, principalmente, as **'Features V'** na seção acima e veja como o gráfico muda! Assim você entende melhor o que mais pesa para o modelo.")

    else:
        st.warning("Por favor, selecione ou digite os dados da transação antes de clicar em 'Analisar'.")

st.markdown("---")
st.caption("Desenvolvido por Igor França como parte do projeto de Cientista de Dados EBAC. ✨")