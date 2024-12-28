# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 09:49:39 2024

@author: 86185
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载数据集
df = pd.read_csv('Dataset.csv')

# 划分特征和目标变量
X = df.drop(['Negative Emotions'], axis=1)
y = df['Negative Emotions']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Negative Emotions']
)

# 显示数据集的前几行
df.head()

from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import GridSearchCV

# CatBoost模型参数
params_cat = {
    'iterations': 1000,               # 迭代次数，相当于树的数量
    'learning_rate': 0.02,            # 学习率，控制每一步的步长，用于防止过拟合
    'depth': 6,                       # 树的深度，控制模型复杂度
    'eval_metric': 'Logloss',         # 评价指标，这里使用对数损失（logloss）
    'random_seed': 42,                # 随机种子，用于重现模型的结果
    'verbose': 0,                     # 控制 CatBoost 输出信息的详细程度，0表示无输出
    'od_type': 'Iter',                # 过拟合检测类型，Iter 表示在每次迭代后进行检测
    'od_wait': 50,                    # 在过拟合检测后，需要等待的迭代次数
    'colsample_bylevel': 0.6,         # 每棵树随机选择的特征比例
    'subsample': 0.7,                 # 每次迭代时随机选择的样本比例
}

# 初始化CatBoost分类模型
model_cat = CatBoostClassifier(**params_cat)

# 定义参数网格，用于网格搜索
param_grid = {
    'iterations': [100, 200, 300, 400, 500],  # 迭代次数
    'depth': [3, 4, 5, 6, 7],                 # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1], # 学习率
}

# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search = GridSearchCV(
    estimator=model_cat,
    param_grid=param_grid,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
    n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)

# 使用最优参数训练模型
best_model = grid_search.best_estimator_

from sklearn.metrics import classification_report

# 使用最佳模型预测测试集
y_pred = best_model.predict(X_test)

# 输出模型报告，查看评价指标
print(classification_report(y_test, y_pred))

import joblib

# 保存模型到文件
joblib.dump(best_model, 'CatBoost.pkl')


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('CatBoost.pkl')

# Define feature options
Age_options = {
    1: '≤35 (1)',
    2: '＞35 (2)',
}
Occupation_options = {
    1: '有稳定工作 (1)',
    2: '无稳定工作 (2)'
}
Method_of_delivery_options = {
    1: '顺产 (1)',
    2: '剖宫产 (2)',
}
Marital_status_options = {
    1: '已婚 (1)',
    2: '未婚 (2)',
}
Educational_degree_options = {
    1: '专科及以下 (1)',
    2: '本科及以上 (2)',
}
average_monthly_household_income_options = {
    1: '≤5000 (1)',
    2: '＞5000 (2)',
}
medical_insurance_options = {
    1: 'No (1)',
    2: 'Yes (2)',
}
mode_of_conception_options = {
    1: '自然受孕 (1)',
    2: '人工受孕 (2)',
}
Pregnancy_complications_options = {
    1: 'Yes',
    2: 'No'
}
Breastfeeding_options = {
    1: 'Yes',
    2: 'No'
}
rooming_in_options = {
    1: 'Yes',
    2: 'No'
}
Planned_pregnancy_options = {
    1: 'Yes',
    2: 'No'
}

# Define feature names
feature_names = [
    "Intrapartum pain", "Postpartum pain", "Resilience", "Family support", "Psychological birth trauma","Age","Occupation","Method of delivery","Marital status","Educational degree","Average monthly household income","Medical insurance","Mode of conception","Pregnancy complications","Breastfeeding","Rooming-in","Planned pregnancy",
]

# Streamlit user interface
st.title("Negative Emotions Predictor")

# Intrapartum pain: numerical input
Intrapartum_pain = st.number_input("Intrapartum pain:", min_value=0, max_value=10, value=5)

# Postpartum pain: numerical input
Postpartum_pain = st.number_input("Postpartum pain:", min_value=0, max_value=10, value=5)

# Resilience: numerical input
Resilience = st.number_input("Resilience:", min_value=6, max_value=30, value=18)

# Family support: numerical input
Family_support = st.number_input("Family support:", min_value=0, max_value=10, value=5)

# Psychological birth trauma: numerical input
Psychological_birth_trauma = st.number_input("Psychological birth trauma:", min_value=0, max_value=42, value=14)

# Age: categorical selection
Age = st.selectbox("Age (1=≤35, 2=＞35):", options=[1, 2], format_func=lambda x: '≤35 (1)' if x == 1 else '＞35 (2)')

# Occupation: categorical selection
Occupation = st.selectbox("Occupation (1=有稳定工作, 2=无稳定工作):", options=[1, 2], format_func=lambda x: '有稳定工作 (1)' if x == 1 else '无稳定工作 (2)')

# Method of delivery: categorical selection
Method_of_delivery = st.selectbox("Method of delivery (1=顺产, 2=剖宫产):", options=[1, 2], format_func=lambda x: '顺产 (1)' if x == 1 else '剖宫产 (2)')

# Marital status: categorical selection
Marital_status = st.selectbox("Marital status (1=已婚, 2=未婚):", options=[1, 2], format_func=lambda x: '已婚 (1)' if x == 1 else '未婚 (2)')

# Educational degree: categorical selection
Educational_degree = st.selectbox("Educational degree (1=专科及以下, 2=本科及以上):", options=[1, 2], format_func=lambda x: '专科及以下 (1)' if x == 1 else '本科及以上 (2)')

# Average monthly household income: categorical selection
Average_monthly_household_income = st.selectbox("Average monthly household income (1=≤5000, 2=＞5000):", options=[1, 2], format_func=lambda x: '≤5000 (1)' if x == 1 else '＞5000 (2)')

# Medical insurance: categorical selection
Medical_insurance = st.selectbox("Medical insurance (1=No, 2=Yes):", options=[1, 2], format_func=lambda x: 'No (1)' if x == 1 else 'Yes (2)')

# Mode of conception: categorical selection
Mode_of_conception = st.selectbox("Mode of conception (1=自然受孕, 2=人工受孕):", options=[1, 2], format_func=lambda x: '自然受孕 (1)' if x == 1 else '人工受孕 (2)')

# Pregnancy complications: categorical selection
Pregnancy_complications = st.selectbox("Pregnancy complications (1=Yes, 2=No):", options=[1, 2], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (2)')

# Breastfeeding: categorical selection
Breastfeeding = st.selectbox("Breastfeeding (1=Yes, 2=No):", options=[1, 2], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (2)')

# Rooming-in: categorical selection
Rooming_in = st.selectbox("Rooming-in (1=Yes, 2=No):", options=[1, 2], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (2)')

# Process inputs and make predictions
feature_values = [Intrapartum_pain, Postpartum_pain, Resilience, Family_support, Psychological_birth_trauma, Age, Occupation, Method_of_delivery, Marital_status, Educational_degree, Average_monthly_household_income, Medical_insurance, Mode_of_conception, Pregnancy_complications, Breastfeeding, Rooming_in]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )
    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
    
