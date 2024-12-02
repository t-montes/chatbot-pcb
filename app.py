import streamlit as st
import pandas as pd

def answer(q):
    if q == "Hola":
        return "¡Hola!", 0
    else:
        return "No puedo responder a eso.", 1

st.set_page_config(layout="wide", page_title="ProCi | Chatbot-Financiero Demo", page_icon="./assets/bot-logo.png")

with open("assets/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
st.image('./assets/header.png')

def display_message(role, content, avatar, dataframe=None, sql_query=None, add_to_history=True):
    message = {"role": role, "content": content}
    if dataframe is not None: message["dataframe"] = dataframe
    if sql_query is not None: message["sql_query"] = sql_query
    if add_to_history: st.session_state.messages.append(message)
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if role == "bot":
            if dataframe is not None:
                with st.expander("Ver la respuesta de la consulta SQL relacionada"):
                    st.dataframe(dataframe, use_container_width=True, hide_index=True)
            if sql_query is not None:
                with st.expander("Ver la consulta SQL relacionada"):
                    st.code(sql_query, language="sql", line_numbers=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    display_message(
        message["role"],
        message["content"],
        './assets/user-logo.png' if message["role"] == "human" else './assets/bot-logo.png',
        message.get("dataframe"),
        message.get("sql_query"),
        add_to_history=False
    )

if prompt := st.chat_input("¡Déjame mostrar mi magia! ¿Cuál es tu pregunta?"):
    display_message("human", prompt, './assets/user-logo.png')

    response, status = answer(prompt)
    if status == 0:
        df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        code = "SELECT * FROM table_name\nWHERE column1 = 1"
        display_message("bot", response, './assets/bot-logo.png', dataframe=df, sql_query=code)
    else:
        display_message("bot", response, './assets/bot-logo.png')
