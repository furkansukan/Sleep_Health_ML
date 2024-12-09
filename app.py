import warnings
import numpy as np
import shap
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import plotly.express as px
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Sidebar'da sayfa seçimi
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Data Analysis", "ML", "Contact"])

# Sayfa 1: Data Analysis
if page == "Data Analysis":
    st.title("Data Analysis Page")


    # CSV dosyasını yükleme
    #@st.cache
    def load_data():
        return pd.read_csv('sample_data.csv')  # CSV dosyanızın ismini buraya ekleyin.


    sleep_data = load_data()

    # CSV verilerini ekrana yazdırma
    st.write("Data Set")
    st.dataframe(sleep_data)
    # Veriyi ekranda göster
    st.title("Sleep Data Information")
    st.write(sleep_data.describe().style.background_gradient())

    st.title("Statistical info including string values")
    st.write(sleep_data.describe(include='O'))
    st.title("Exploratory Data Analysis(EDA)")

    st.title("for number of values of columns")
    number_of_values = sleep_data.nunique()
    st.write(number_of_values)

    st.title("Data Visualization")
    # Plotlama işlemi için seaborn ve matplotlib kullan
    sns.pairplot(data=sleep_data.drop('Person ID', axis=1), hue='Sleep Disorder', palette='mako')
    st.write("Pairplot of Data for Sleep Disorder")

    # Görselleştirmeyi Streamlit'e yerleştir
    st.pyplot(plt, use_container_width=True)

    st.title("Distribution of Persons with Sleep Disorder")
    fig = px.histogram(sleep_data, x='Sleep Disorder',
                       barmode="group", color='Sleep Disorder',
                       color_discrete_sequence=['white', '#4A235A', '#C39BD3'],
                       text_auto=True)

    fig.update_layout(title='<b>Distribution of Persons with Sleep Disorder</b>',
                      title_font={'size': 25},
                      showlegend=True)

    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig)
    st.title("Sleep Disorder and Gender Distribution")

    # Sleep Disorder ve Gender'a göre sayım işlemi
    gender_counts = sleep_data.groupby('Sleep Disorder')['Gender'].value_counts()

    # Streamlit üzerinde göster
    st.write("Number of Persons by Sleep Disorder and Gender")
    st.write(gender_counts)

    st.title("Average Sleep Duration by Quality and Sleep Disorder")

    # Pivot table oluştur ve stilize et
    pivot_data = sleep_data.pivot_table(
        index='Quality of Sleep',
        columns='Sleep Disorder',
        values='Sleep Duration',
        aggfunc='mean'
    )

    # Streamlit'e background gradient ile göster
    st.dataframe(pivot_data.style.background_gradient(cmap='viridis'))

    st.title("The Effect of Physical Activity on Sleep - Violin Plot")

    # Violin grafiğini oluştur ve streamlit'e yerleştir
    fig = px.violin(
        sleep_data,
        x="Sleep Disorder",
        y='Physical Activity Level',
        color='Sleep Disorder',
        color_discrete_sequence=['white', '#4A235A', '#C39BD3'],
        violinmode='overlay'
    )

    fig.update_layout(
        title='<b>The Effect of Activities on Sleep</b>',
        title_font={'size': 25}
    )

    fig.update_yaxes(showgrid=False)

    # Streamlit'e gömme
    st.plotly_chart(fig)

    st.title("Most Affected Ages in Each Type of Sleep Disorder - Bar Plot")

    # Bar grafiğini oluştur ve Streamlit'e yerleştir
    sleep_data.pivot_table(
        index='Gender',
        columns='Sleep Disorder',
        values='Age',
        aggfunc='median'
    ).plot(
        kind='bar',
        title='Most affected ages in each type of Sleep Disorder',
        alpha=0.7
    )

    plt.xlabel('Gender')
    plt.ylabel('Median Age')
    plt.legend(title='Sleep Disorder')

    # Streamlit'e yerleştir
    st.pyplot(plt)

    sleep_data['Blood Pressure'] = sleep_data['Blood Pressure'].apply(
        lambda x: 0 if x in ['120/80', '126/83', '125/80', '128/84', '129/84', '117/76', '118/76', '115/75', '125/82',
                             '122/80'] else 1)
    # 0 = normal blood pressure
    # 1 = abnormal blood pressure
    sleep_data["Age"] = pd.cut(sleep_data["Age"], 2, labels=[0, 1])
    sleep_data["Heart Rate"] = pd.cut(sleep_data["Heart Rate"], 4, labels=[0, 1, 2, 3])



    LE = LabelEncoder()

    categories = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Physical Activity Level', 'BMI Category',
                  'Heart Rate', 'Daily Steps', 'Sleep Disorder']
    for label in categories:
        sleep_data[label] = LE.fit_transform(sleep_data[label])
    sleep_data["Daily Steps"] = pd.cut(sleep_data["Daily Steps"], 4, labels=[0, 1, 2, 3])
    sleep_data["Sleep Duration"] = pd.cut(sleep_data["Sleep Duration"], 3, labels=[0, 1, 2])
    sleep_data["Physical Activity Level"] = pd.cut(sleep_data["Physical Activity Level"], 4, labels=[0, 1, 2, 3])

    sleep_data.drop(['Person ID'], axis=1, inplace=True)
    st.write(sleep_data)
    st.session_state.sleep_data = sleep_data






# Sayfa 2: ML
elif page == "ML":
    st.title("Machine Learning")
    sleep_data = st.session_state.sleep_data

    x = sleep_data.iloc[:, :-1]
    y = sleep_data.iloc[:, -1]

    x_shape = colored(x.shape, "magenta", None, attrs=["blink"])
    y_shape = colored(y.shape, "magenta", None, attrs=["blink"])
    print('The dimensions of x is : ', x_shape)
    print('The dimensions of y is : ', y_shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=32, shuffle=True)
    x_train_shape = colored(x_train.shape, "magenta", None, attrs=["blink"])
    x_test_shape = colored(x_test.shape, "magenta", None, attrs=["blink"])
    y_train_shape = colored(y_train.shape, "magenta", None, attrs=["blink"])
    y_test_shape = colored(y_test.shape, "magenta", None, attrs=["blink"])

    print("x train dimensions :", x_train_shape)
    print("x test dimensions: ", x_test_shape)
    print("y train dimensions :", y_train_shape)
    print("y test dimensions :", y_test_shape)

    st.title("Logistic Regression")
    LR = LogisticRegression().fit(x_train, y_train)
    # Calculate scores and apply colored output
    LR_training_score = round(LR.score(x_train, y_train) * 100, 2)
    LR_testing_score = round(LR.score(x_test, y_test) * 100, 2)

    # Display scores in Streamlit
    st.write(f"LR Training Score: {LR_training_score}")
    st.write(f"LR Testing Score: {LR_testing_score}")
    LR_y_pred = LR.predict(x_test)
    st.title("XGBClassifier Model")


    xgb = XGBClassifier(enable_categorical=True).fit(x_train, y_train)
    xgb_training_score = round(xgb.score(x_train, y_train) * 100, 2)
    xgb_testing_score = round(xgb.score(x_test, y_test) * 100, 2)
    st.write(f"XGB training score: {xgb_training_score}")
    st.write(f"XGB Testing Score: {xgb_testing_score}")
    xgb_y_pred = xgb.predict(x_test)

    st.title("GradientBoostingClassifier Model")
    GBC = GradientBoostingClassifier().fit(x_train,y_train)
    GBC_training_score = round(GBC.score(x_train, y_train) * 100, 2)
    GBC_testing_score = round(GBC.score(x_test, y_test) * 100, 2)
    st.write(f"GBC training score: {GBC_training_score}")
    st.write(f"GBC Testing Score: {GBC_testing_score}")
    GBC_y_pred = GBC.predict(x_test)


    st.title("SVC Model")
    svc = SVC().fit(x_train,y_train)
    svc_training_score = round(svc.score(x_train, y_train) * 100, 2)
    svc_testing_score = round(svc.score(x_test, y_test) * 100, 2)
    st.write(f"SVC training score: {svc_training_score}")
    st.write(f"SVC Testing Score: {svc_testing_score}")
    svc_y_pred = svc.predict(x_test)

    st.title("Models evaluation")
    # Model isimleri
    models_predictions = [LR_y_pred, xgb_y_pred, GBC_y_pred, svc_y_pred]
    model_names = ["Logistic Regression", "XGBoost", "Gradient Boosting Classifier", "Support Vector Classifier"]

    # Grafik oluştur ve Streamlit'te göster
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, (ax, y_pred, model_name) in enumerate(zip(axes.flat, models_predictions, model_names), 1):
        cm = confusion_matrix(y_test, y_pred)  # Confusion Matrix hesaplama
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='BuPu',
            xticklabels=['None', 'Sleep Apnea', 'Insomnia'],
            yticklabels=['None', 'Sleep Apnea', 'Insomnia'],
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {model_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # Alt grafiklerin aralığını ayarla
    plt.tight_layout()

    # Streamlit'te grafik göster
    st.pyplot(fig)

    shap_values = shap.TreeExplainer(xgb).shap_values(x_test)

    st.title("SHAP Summary Plot")
    st.subheader("Feature Importance Across Classes")

    # SHAP özet grafiğini oluştur ve Streamlit'te göster
    plt.figure()  # Yeni bir figür oluştur
    shap.summary_plot(shap_values, x_test, class_names=['None', 'Sleep_Apnea', 'Insomnia'], show=False)
    st.pyplot(plt.gcf())  # Mevcut figürü alıp Streamlit'te göster

# Sayfa 3: Contact
elif page == "Contact":
    st.title("Contact Page")

    st.title("Daha Fazla Soru ve İletişim İçin")
    st.write("""
    Bu projeyle ilgili herhangi bir sorunuz veya geri bildiriminiz olursa benimle iletişime geçmekten çekinmeyin! Aşağıdaki platformlar üzerinden ulaşabilirsiniz:
    """)

    st.markdown("""
    - 📧 **E-posta**: [furkansukan10@gmail.com](mailto:furkansukan10@gmail.com)  
    - 🪪 **LinkedIn**: [furkansukan](https://www.linkedin.com/in/furkansukan)  
    - 🔗 **Kaggle**: [Profilim](https://www.kaggle.com/furkansukan)  
    - 🐙 **GitHub**: [furkansukan](https://github.com/furkansukan)  
    
    """)
    
    st.write("Görüş ve önerilerinizi duymaktan mutluluk duyarım!")
    # EKLEME YERİ: Contact sayfasına ait kodlarınızı buraya ekleyin.
    # ----------------------------------------
    # Örnek: st.write("İletişim bilgileri burada görünecek")
    # ----------------------------------------
