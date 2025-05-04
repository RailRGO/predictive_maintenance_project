import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="Предиктивное обслуживание оборудования",
    page_icon="🔧",
    layout="wide"
)

# Создание боковой панели навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ["Анализ и модель", "Презентация"]
)

# Отображение выбранной страницы
if page == "Анализ и модель":
    st.title("Анализ данных и модель предиктивного обслуживания")
    from analysis_and_model import analysis_and_model_page
    analysis_and_model_page()
    
elif page == "Презентация":
    st.title("Презентация проекта")
    from presentation import presentation_page
    presentation_page()