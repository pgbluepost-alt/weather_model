
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# タイトルと説明
st.set_page_config(page_title="AI天気予報アプリ", page_icon="☀️")
st.title("☀️ AI天気予報システム")
st.write("今日の気温を入力すると、AIが明日の気温を予測します。")

# モデルの読み込み
model = joblib.load("weather_model.joblib")

# 入力フォーム
st.sidebar.header("設定")
today_temp = st.sidebar.number_input("今日の平均気温 (℃)", value=20.0, step=0.1)

# 予測実行
if st.button("予測する"):
    # AIへの入力形式を整える
    input_data = pd.DataFrame([[today_temp]], columns=['前日の平均気温'])
    prediction = model.predict(input_data)[0]
    
    # 結果の表示
    st.subheader(f"明日の予測気温")
    st.metric(label="予測値", value=f"{prediction:.1f} ℃", delta=f"{prediction - today_temp:.1f} ℃ (今日との差)")
    
    st.success("AIによる推論が完了しました。")

st.info("このアプリは過去30年の気象データから学習した線形回帰モデルを使用しています。")
