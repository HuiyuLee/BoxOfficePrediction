import streamlit as st
import numpy as np
import pandas as pd

st.title("Hello, Streamlit!")
st.write("這是一個簡單的範例，歡迎來到 Streamlit 的世界！")
st.title("互動範例：按鈕與輸入框")
# 文字輸入框
name = st.text_input("請輸入你的名字：", value="你的名字")
st.write(f"你好，{name}！")
# 按鈕
if st.button("點擊我"):
    st.write("你剛剛點擊了按鈕！")

st.sidebar.title("控制面板")
option = st.sidebar.selectbox("選擇一個選項：", ["選項1", "選項2", "選項3"])
st.write(f"你選擇了：{option}")


st.title("檔案上傳範例")
uploaded_file = st.file_uploader("選擇一個 CSV 檔案", type="csv")
if uploaded_file is not None:
    # 讀取 CSV 檔案
    df = pd.read_csv(uploaded_file)
    st.write("上傳的資料：")
    st.dataframe(df)
    
    st.write("資料摘要：")
    st.write(df.describe())

st.title("數據分析展示範例")
# 隨機生成數據
data = np.random.randn(100, 2)
df = pd.DataFrame(data, columns=["變數A", "變數B"])
st.write("隨機數據：")
st.dataframe(df)
st.write("線圖展示：")
st.line_chart(df)
st.write("直方圖展示：")
st.bar_chart(df)

import time
import streamlit as st

st.title("模型訓練進度")

progress_bar = st.progress(0)
status_text = st.empty()

epochs = 50
for i in range(epochs):
    # 模擬訓練過程
    time.sleep(0.1)
    
    # 更新進度條與文字
    progress_bar.progress(int((i+1)/epochs * 100))
    status_text.text(f"正在訓練... 第 {i+1}/{epochs} epoch")

st.success("訓練完成 ✅")
