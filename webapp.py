import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# โหลดโมเดล
rf_model = joblib.load("rf_model.pkl")
gb_model = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")
nn_model_boston = joblib.load("nn_model_boston.pkl")
scaler_boston = joblib.load("scaler_boston.pkl")

# ตั้งค่า Sidebar Menu
st.sidebar.title("📌 Menu")
page = st.sidebar.radio(
    "🔍 Select menu",
    [
        "📖 Machine Learning",
        "📊 Ticket Price Prediction",
        "🤖 Neural Network",
        "📈 House Price Prediction"
    ]
)

st.title("Machine Learning & Neural Network Web Application 📊")

# ------------------- หน้า 1: Machine Learning Overview -------------------
if page == "📖 Machine Learning":
    st.header("📖 Machine Learning Overview")
    st.write("""
    ## 🔍 1️⃣ การเตรียมข้อมูล (Data Preparation)
    - ข้อมูลที่ใช้: `titanic_dataset_with_missing.csv` ซึ่งมีค่าที่ขาดหายไป
    - เทคนิคที่ใช้:
        - เติมค่าที่หายไปด้วยค่าเฉลี่ย (`mean()`) ซึ่งเป็นวิธีที่เหมาะสมสำหรับข้อมูลเชิงตัวเลข (García et al., 2010)
        - แยกข้อมูลเป็น **Feature** และ **Target (Fare)** สำหรับการพยากรณ์
        - ใช้ `StandardScaler()` เพื่อทำให้ค่ามีมาตราส่วนที่เหมาะสมกับอัลกอริทึม (Han et al., 2011)

    ## 🧠 2️⃣ ทฤษฎีของอัลกอริทึมที่ใช้ (Machine Learning Algorithms)
    - **Random Forest (RF)**: อัลกอริทึมแบบ Ensemble Learning ที่รวมหลาย Decision Trees เข้าไว้ด้วยกัน เพื่อลด Overfitting และเพิ่มความแม่นยำ (Breiman, 2001)
    - **Gradient Boosting (GB)**: อัลกอริทึมที่ใช้หลักการ Boosting โดยให้โมเดลแต่ละตัวเรียนรู้จากข้อผิดพลาดของตัวก่อนหน้า ทำให้มีความสามารถในการเรียนรู้สูงขึ้น (Friedman, 2001)

    ## 🔧 3️⃣ ขั้นตอนการพัฒนาโมเดล
    - แบ่งข้อมูลเป็น **80% Training** และ **20% Testing** ด้วย `train_test_split()`
    - ใช้ `StandardScaler()` เพื่อทำการ Normalize ข้อมูล
    - เทรนโมเดล `RandomForestRegressor` และ `GradientBoostingRegressor`
    - ใช้ Mean Absolute Error (MAE) และ R² Score ในการวัดผลลัพธ์ของโมเดล (Chicco et al., 2021)

    ### 📚 อ้างอิง:
    - Breiman, L. (2001). Random forests. *Machine learning*, 45(1), 5-32.
    - Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of statistics*, 1189-1232.
    - García, S., Luengo, J., & Herrera, F. (2010). Data preprocessing in data mining. *Springer*.
    - Han, J., Kamber, M., & Pei, J. (2011). Data mining: concepts and techniques. *Elsevier*.
    - Chicco, D., Warrens, M. J., & Jurman, G. (2021). The coefficient of determination R². *Computers in Biology and Medicine*, 138, 104857.
    """)

# ------------------- หน้า 2: Ticket Price Prediction -------------------
if page == "📊 Ticket Price Prediction":
    st.subheader("🔍 Predict Ticket Price")
    
    # โหลดข้อมูล (โหลดครั้งเดียว)
    if "df_ticket" not in st.session_state:
        df_ticket = pd.read_csv("titanic_dataset_with_missing.csv")
        st.session_state["df_ticket"] = df_ticket
    else:
        df_ticket = st.session_state["df_ticket"]

    features = df_ticket.drop(columns=['fare'], errors='ignore').columns.tolist()
    
    input_data = [st.number_input(f"{feature}", value=df_ticket[feature].mean()) for feature in features]
    
    if st.button("Predict Price"):
        input_array = np.array([input_data]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred_rf = rf_model.predict(input_scaled)[0]
        pred_gb = gb_model.predict(input_scaled)[0]
        weighted_avg = (0.6 * pred_rf) + (0.4 * pred_gb)
        st.success(f"🎟️ Predicted Ticket Price: **${weighted_avg:.2f}**")

# ------------------- หน้า 3: Neural Network Overview -------------------
if page == "🤖 Neural Network":
    st.header("🤖 Neural Network Overview")
    st.write("""
    ## 🔍 1️⃣ การเตรียมข้อมูล (Data Preparation)
    - ข้อมูลที่ใช้: `boston_housing_simulated_with_missing.csv`
    - เทคนิคที่ใช้:
        - เติมค่าที่หายไปด้วยค่ามัธยฐาน (`median()`) เพื่อลดผลกระทบจากค่า outlier (Little & Rubin, 2019)
        - ใช้ `StandardScaler()` เพื่อทำให้ค่าต่างๆ อยู่ในช่วงที่เหมาะสมสำหรับการเรียนรู้ของ Neural Network (Lecun et al., 1998)

    ## 🧠 2️⃣ ทฤษฎีของอัลกอริทึมที่ใช้ (Neural Network)
    - ใช้ **Multi-layer Perceptron (MLP Regressor)** ซึ่งเป็นโครงข่ายประสาทเทียมที่ใช้การส่งต่อข้อมูลแบบ Feedforward
    - โครงสร้างโมเดล:
        - **Hidden Layers**: 4 ชั้น (`512, 256, 128, 64`) พร้อม Activation Function `ReLU`
        - **Optimization Algorithm**: ใช้ `Adam` optimizer ซึ่งสามารถปรับค่า learning rate ได้แบบ Adaptive (Kingma & Ba, 2014)
        - **Early Stopping**: หยุดการเทรนเมื่อค่าความผิดพลาดไม่ลดลง เพื่อลด Overfitting (Prechelt, 1998)

    ## 🔧 3️⃣ ขั้นตอนการพัฒนาโมเดล
    - แบ่งข้อมูลเป็น **80% Training** และ **20% Testing**
    - ทำการ Scaling ด้วย `StandardScaler()`
    - ใช้ `MLPRegressor` เทรนโมเดล พร้อมพารามิเตอร์ที่ถูกปรับแต่งแล้ว
    - ใช้ **Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)** และ **R² Score** ในการประเมินผลลัพธ์ (Chicco et al., 2021)

    ### 📚 อ้างอิง:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
    - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
    - Prechelt, L. (1998). Early stopping-but when? *Neural Networks: Tricks of the trade*, 55-69.
    - Little, R. J., & Rubin, D. B. (2019). Statistical analysis with missing data. *John Wiley & Sons*.
    - Chicco, D., Warrens, M. J., & Jurman, G. (2021). The coefficient of determination R². *Computers in Biology and Medicine*, 138, 104857.
    """)

# ------------------- หน้า 4: House Price Prediction -------------------
if page == "📈 House Price Prediction":
    st.subheader("🏡 House Price Prediction (Neural Network)")

    # 🔹 ลบข้อมูลเก่าจาก session_state ถ้ามี
    if "df_housing" in st.session_state:
        del st.session_state["df_housing"]

    # 🔹 โหลดข้อมูลใหม่
    df_housing = pd.read_csv("boston_housing_simulated_with_missing.csv")
    df_housing.fillna(df_housing.median(), inplace=True)

    # 🔹 แปลงชื่อคอลัมน์ให้เป็น lowercase
    df_housing.columns = df_housing.columns.str.lower()

    # 🔹 บันทึก dataset ที่ถูกต้องใน session_state
    st.session_state["df_housing"] = df_housing

    # 🔹 ใช้ Feature Name ที่ Model เคย Train
    features = list(scaler_boston.feature_names_in_)

    # 🔹 ตรวจสอบว่าฟีเจอร์ที่ต้องการมีอยู่ใน dataset หรือไม่
    missing_features = [f for f in features if f not in df_housing.columns]

    # 🔹 เติมค่า 0 ให้ฟีเจอร์ที่ขาดหายไป
    for feature in missing_features:
        df_housing[feature] = 0  

    # 🔹 เรียงลำดับคอลัมน์ให้ตรงกับที่โมเดลใช้
    df_housing = df_housing[features]

    # 🔹 รับค่า input จากผู้ใช้ (ใช้ text_input เพื่อให้กรอกได้)
    input_data = []
    for feature in features:
        value = st.text_input(f"{feature}", "0.0")  # ✅ เปลี่ยนเป็น text_input
        try:
            value = float(value)  # ✅ แปลงเป็นตัวเลข
        except ValueError:
            value = 0.0  # ✅ ถ้ากรอกผิด ให้ใช้ค่า 0.0
        input_data.append(value)

    # 🔹 คำนวณราคาบ้านเมื่อกดปุ่ม
    if st.button("Predict House Price"):
        input_array = pd.DataFrame([input_data], columns=features)
        input_scaled = scaler_boston.transform(input_array)
        predicted_price = nn_model_boston.predict(input_scaled)[0]
        st.success(f"🏡 Predicted House Price: **${predicted_price:.2f}**")

    # 🔹 ตรวจสอบว่ามี 'medv' หรือไม่
    if "medv" in df_housing.columns:
        y_actual = df_housing["medv"]

        # 🔹 กราฟ Actual vs Predicted
        st.subheader("📈 Actual vs Predicted Prices")

        X_test = df_housing[features]
        X_test_scaled = scaler_boston.transform(X_test)
        y_predicted = nn_model_boston.predict(X_test_scaled)

        fig, ax = plt.subplots()
        ax.scatter(y_actual, y_predicted, color="blue", label="Predicted")
        ax.plot(
            [y_actual.min(), y_actual.max()],
            [y_actual.min(), y_actual.max()],
            color="red",
            linestyle="--",
            label="Ideal Fit",
        )
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Actual vs Predicted Prices (Neural Network)")
        ax.legend()
        st.pyplot(fig)
