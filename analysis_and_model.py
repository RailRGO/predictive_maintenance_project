import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from ucimlrepo import fetch_ucirepo
import io

def analysis_and_model_page():
    # Создаем вкладки для разных этапов анализа
    tabs = st.tabs(["Загрузка данных", "Исследовательский анализ", "Обучение модели", "Предсказания"])
    
    # Глобальные переменные для хранения данных и моделей
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    
    # Вкладка 1: Загрузка данных
    with tabs[0]:
        st.header("Загрузка данных")
        
        data_source = st.radio(
            "Выберите источник данных:",
            ["Загрузить из UCI репозитория", "Загрузить CSV файл"]
        )
        
        if data_source == "Загрузить из UCI репозитория":
            if st.button("Загрузить данные из UCI"):
                with st.spinner("Загрузка данных..."):
                    # Загрузка датасета из UCI репозитория
                    dataset = fetch_ucirepo(id=601)
                    features = dataset.data.features
                    targets = dataset.data.targets
                    data = pd.concat([features, targets], axis=1)
                    st.session_state.data = data
                    st.success("Данные успешно загружены!")
                    
                    # Показать информацию о датасете
                    st.subheader("Информация о датасете")
                    st.write(f"Количество записей: {data.shape[0]}")
                    st.write(f"Количество признаков: {data.shape[1]}")
                    st.write("Первые 5 строк данных:")
                    st.dataframe(data.head())
        
        else:
            uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success("Данные успешно загружены!")
                
                # Показать информацию о датасете
                st.subheader("Информация о датасете")
                st.write(f"Количество записей: {data.shape[0]}")
                st.write(f"Количество признаков: {data.shape[1]}")
                st.write("Первые 5 строк данных:")
                st.dataframe(data.head())
        
        if st.session_state.data is not None:
            st.subheader("Проверка на пропущенные значения")
            missing_values = st.session_state.data.isnull().sum()
            st.write(missing_values)
            
            if missing_values.sum() == 0:
                st.success("В данных нет пропущенных значений!")
            else:
                st.warning(f"В данных обнаружено {missing_values.sum()} пропущенных значений.")
    
    # Вкладка 2: Исследовательский анализ
    with tabs[1]:
        st.header("Исследовательский анализ данных")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Предобработка данных
            st.subheader("Предобработка данных")
            
            # Удаление ненужных столбцов
            columns_to_drop = st.multiselect(
                "Выберите столбцы для удаления:",
                data.columns,
                default=["UDI", "Product ID"] if "UDI" in data.columns and "Product ID" in data.columns else []
            )
            
            if columns_to_drop:
                data = data.drop(columns=columns_to_drop)
                st.write("Данные после удаления столбцов:")
                st.dataframe(data.head())
            
            # Преобразование категориальных переменных
            if 'Type' in data.columns:
                st.subheader("Преобразование категориальных переменных")
                st.write("Преобразование столбца 'Type' в числовой формат")
                
                label_encoder = LabelEncoder()
                data['Type'] = label_encoder.fit_transform(data['Type'])
                st.session_state.label_encoder = label_encoder
                
                st.write("Данные после преобразования:")
                st.dataframe(data.head())
            
            # Статистический анализ
            st.subheader("Статистический анализ")
            st.write("Описательная статистика:")
            st.dataframe(data.describe())
            
            # Визуализация распределения целевой переменной
            st.subheader("Распределение целевой переменной")
            
            target_column = st.selectbox(
                "Выберите целевую переменную:",
                ["Machine failure", "Target"] if "Machine failure" in data.columns else ["Target"]
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            target_counts = data[target_column].value_counts()
            sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax)
            ax.set_title(f"Распределение {target_column}")
            ax.set_xlabel("Класс")
            ax.set_ylabel("Количество")
            
            # Добавляем проценты над столбцами
            total = len(data)
            for i, count in enumerate(target_counts.values):
                percentage = count / total * 100
                ax.text(i, count + 50, f"{percentage:.1f}%", ha='center')
            
            st.pyplot(fig)
            
            # Корреляционная матрица
            st.subheader("Корреляционная матрица")
            
            numeric_cols = data.select_dtypes(include=['number']).columns
            corr_matrix = data[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
            
            # Сохраняем предобработанные данные
            st.session_state.data = data
            
            # Визуализация распределения признаков
            st.subheader("Распределение числовых признаков")
            
            selected_features = st.multiselect(
                "Выберите признаки для визуализации:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_features:
                fig, axes = plt.subplots(len(selected_features), 2, figsize=(15, 4*len(selected_features)))
                
                for i, feature in enumerate(selected_features):
                    # Гистограмма
                    sns.histplot(data=data, x=feature, hue=target_column, kde=True, ax=axes[i, 0])
                    axes[i, 0].set_title(f"Распределение {feature}")
                    
                    # Ящик с усами
                    sns.boxplot(data=data, x=target_column, y=feature, ax=axes[i, 1])
                    axes[i, 1].set_title(f"Boxplot для {feature}")
                
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            st.warning("Сначала загрузите данные на вкладке 'Загрузка данных'")
    
    # Вкладка 3: Обучение модели
    with tabs[2]:
        st.header("Обучение модели")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Выбор целевой переменной
            target_column = st.selectbox(
                "Выберите целевую переменную:",
                ["Machine failure", "Target"] if "Machine failure" in data.columns else ["Target"],
                key="target_select"
            )
            
            # Выбор признаков
            feature_columns = [col for col in data.columns if col != target_column]
            selected_features = st.multiselect(
                "Выберите признаки для обучения модели:",
                feature_columns,
                default=feature_columns
            )
            
            if selected_features:
                # Разделение данных
                X = data[selected_features]
                y = data[target_column]
                
                test_size = st.slider("Размер тестовой выборки (%)", 10, 40, 20) / 100
                random_state = st.number_input("Random state", 0, 100, 42)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Масштабирование данных
                if st.checkbox("Масштабировать данные", value=True):
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    st.session_state.scaler = scaler
                
                # Сохраняем тестовые данные для оценки
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Выбор модели
                model_type = st.selectbox(
                    "Выберите модель:",
                    ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]
                )
                
                # Настройка параметров модели
                if model_type == "Logistic Regression":
                    C = st.slider("Параметр регуляризации C", 0.01, 10.0, 1.0)
                    max_iter = st.slider("Максимальное количество итераций", 100, 1000, 100)
                    
                    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
                
                elif model_type == "Random Forest":
                    n_estimators = st.slider("Количество деревьев", 10, 200, 100)
                    max_depth = st.slider("Максимальная глубина", 2, 20, 10)
                    
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                
                elif model_type == "XGBoost":
                    n_estimators = st.slider("Количество деревьев", 10, 200, 100, key="xgb_n_est")
                    learning_rate = st.slider("Скорость обучения", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Максимальная глубина", 2, 10, 6, key="xgb_max_depth")
                    
                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                
                elif model_type == "SVM":
                    C = st.slider("Параметр регуляризации C", 0.1, 10.0, 1.0, key="svm_c")
                    kernel = st.selectbox("Ядро", ["linear", "rbf", "poly"])
                    
                    model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
                
                # Обучение модели
                if st.button("Обучить модель"):
                    with st.spinner(f"Обучение модели {model_type}..."):
                        model.fit(X_train, y_train)
                        st.session_state.model = model
                        
                        # Оценка модели
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        conf_matrix = confusion_matrix(y_test, y_pred)
                        class_report = classification_report(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                        
                        st.success(f"Модель успешно обучена! Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
                        
                        # Визуализация результатов
                        st.subheader("Результаты обучения модели")
                        
                        # Метрики
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.4f}")
                        with col2:
                            st.metric("ROC-AUC", f"{roc_auc:.4f}")
                        
                        # Матрица ошибок
                        st.subheader("Матрица ошибок (Confusion Matrix)")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Предсказанные значения')
                        ax.set_ylabel('Истинные значения')
                        ax.set_title('Матрица ошибок')
                        st.pyplot(fig)
                        
                        # Отчет о классификации
                        st.subheader("Отчет о классификации")
                        st.text(class_report)
                        
                        # ROC-кривая
                        st.subheader("ROC-кривая")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        ax.plot(fpr, tpr, label=f"{model_type} (AUC = {roc_auc:.4f})")
                        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайное угадывание')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('ROC-кривая')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Важность признаков (для моделей, которые поддерживают это)
                        if model_type in ["Random Forest", "XGBoost"]:
                            st.subheader("Важность признаков")
                            
                            if model_type == "Random Forest":
                                importances = model.feature_importances_
                                feature_names = selected_features
                            elif model_type == "XGBoost":
                                importances = model.feature_importances_
                                feature_names = selected_features
                            
                            feature_importance = pd.DataFrame({
                                'Признак': feature_names,
                                'Важность': importances
                            }).sort_values(by='Важность', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Важность', y='Признак', data=feature_importance, ax=ax)
                            ax.set_title('Важность признаков')
                            st.pyplot(fig)
                
                # Если модель уже обучена, показываем результаты
                if st.session_state.model is not None and st.session_state.X_test is not None and st.session_state.y_test is not None:
                    st.subheader("Результаты последней обученной модели")
                    
                    model = st.session_state.model
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("ROC-AUC", f"{roc_auc:.4f}")
            
            else:
                st.warning("Выберите хотя бы один признак для обучения модели")
        
        else:
            st.warning("Сначала загрузите данные на вкладке 'Загрузка данных'")
    
    # Вкладка 4: Предсказания
    with tabs[3]:
        st.header("Предсказания на новых данных")
        
        if st.session_state.model is not None and st.session_state.data is not None:
            data = st.session_state.data
            model = st.session_state.model
            
            st.subheader("Введите значения признаков для предсказания")
            
            with st.form("prediction_form"):
                # Создаем поля ввода для каждого признака
                input_data = {}
                
                # Если есть признак Type, создаем для него выпадающий список
                if 'Type' in data.columns:
                    type_options = ['L', 'M', 'H']
                    selected_type = st.selectbox("Type", type_options)
                    
                    # Преобразуем выбранный тип в числовой формат
                    # Используем простое отображение вместо LabelEncoder
                    type_mapping = {'L': 0, 'M': 1, 'H': 2}
                    input_data['Type'] = type_mapping[selected_type]
                
                # Создаем поля ввода для числовых признаков
                numeric_features = [col for col in data.columns if col not in ['Type', 'Machine failure', 'Target']]
                
                for feature in numeric_features:
                    # Получаем минимальное и максимальное значения из данных
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    
                    # Создаем поле ввода с подсказками о диапазоне значений
                    input_data[feature] = st.number_input(
                        f"{feature} (диапазон: {min_val:.2f} - {max_val:.2f})",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=0.01 if feature in ['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]'] else 1.0
                    )
                
                submit_button = st.form_submit_button("Предсказать")
            
            if submit_button:
                # Создаем DataFrame из введенных данных
                input_df = pd.DataFrame([input_data])
                
                # Масштабируем данные, если использовалось масштабирование
                if st.session_state.scaler is not None:
                    input_scaled = st.session_state.scaler.transform(input_df)
                else:
                    input_scaled = input_df
                
                # Делаем предсказание
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)[:, 1]
                
                # Выводим результат
                st.subheader("Результат предсказания")
                
                if prediction[0] == 1:
                    st.error(f"Предсказание: Отказ оборудования (класс 1)")
                else:
                    st.success(f"Предсказание: Нормальная работа оборудования (класс 0)")
                
                st.write(f"Вероятность отказа: {prediction_proba[0]:.4f}")
                
                # Визуализация вероятности
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(['Вероятность отказа'], [prediction_proba[0]], color='red' if prediction[0] == 1 else 'green')
                ax.set_xlim(0, 1)
                ax.axvline(x=0.5, color='gray', linestyle='--')
                for i, v in enumerate([prediction_proba[0]]):
                    ax.text(v + 0.01, i, f"{v:.4f}", va='center')
                st.pyplot(fig)
                
                # Пояснение к предсказанию
                st.subheader("Интерпретация результата")
                if prediction[0] == 1:
                    st.write("""
                    Модель предсказывает, что при заданных параметрах произойдет отказ оборудования. 
                    Рекомендуется провести профилактическое обслуживание или проверить параметры работы оборудования.
                    """)
                else:
                    st.write("""
                    Модель предсказывает, что при заданных параметрах оборудование будет работать нормально. 
                    Тем не менее, рекомендуется регулярно проводить плановое обслуживание.
                    """)
        
        else:
            st.warning("Сначала загрузите данные и обучите модель на вкладках 'Загрузка данных' и 'Обучение модели'")

if __name__ == "__main__":
    analysis_and_model_page()