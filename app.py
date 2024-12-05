# ************************* Chatbot Financiero GenAI *************************
import streamlit as st
from agent import Agent

# ************************ Init ************************
if "history" not in st.session_state:
    st.session_state.history = []

agent = Agent()

# ************************ Front ************************
st.set_page_config(layout="wide", page_title="ProCi | Chatbot-Financiero Demo", page_icon="./assets/bot-logo.png")

with open("assets/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
st.image('./assets/header_costco.png')

def display_message(role, content, status, dataframe=None, sql_query=None, add_to_history=True):
    message = {"role": role, "content": content, "status": "success" if status else "error"}
    if dataframe is not None: message["dataframe"] = dataframe
    if sql_query is not None: message["sql_query"] = sql_query
    if add_to_history: st.session_state.history.append(message)
    if role == "human":
        with st.chat_message(role, avatar='./assets/user-logo.png'):
            st.markdown(content)
    else:
        with st.chat_message(role, avatar='./assets/bot-logo.png'):
            st.markdown(content)
            if dataframe is not None:
                with st.expander("Ver la tabla de datos"):
                    st.dataframe(dataframe, use_container_width=True, hide_index=True)
            if sql_query is not None:
                with st.expander("Ver la consulta SQL"):
                    st.code(sql_query, language="sql", line_numbers=True)

for message in st.session_state.history:
    display_message(
        message["role"],
        message["content"],
        message["status"] == "success",
        message.get("dataframe"),
        message.get("sql_query"),
        add_to_history=False
    )

if prompt := st.chat_input("¡Déjame mostrar mi magia! ¿Cuál es tu pregunta?"):
    display_message("human", prompt, True)
    with st.spinner("Procesando..."):
        response, status, dataframe, sql_query = agent(prompt, st.session_state.history)
    display_message("bot", response, status, dataframe, sql_query)
