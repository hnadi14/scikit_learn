
import plotly.graph_objects as go
from ipywidgets import interact, IntSlider
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import os
import joblib


# بارگذاری دادهها
df = pd.read_csv('mall_customers.csv')

# بررسی دادههای گمشده
print("دادههای گمشده:\n", df.isnull().sum())

# استخراج ویژگیهای مورد استفاده
X = df[['Annual_Income_(k$)', 'Spending_Score_(1-100)']]

# پیدا کردن بهترین تعداد خوشه
wcss = []
silhouette_scores = []
max_clusters = 10

for k in range(2, max_clusters + 1):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    pipeline.fit(X)
    labels = pipeline.named_steps['kmeans'].labels_
    wcss.append(pipeline.named_steps['kmeans'].inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

best_k = np.argmax(silhouette_scores) + 2  # +2 چون از k=2 شروع کردیم
print(f"بهترین تعداد خوشه (Silhouette): {best_k} (امتیاز: {max(silhouette_scores):.2f})")

# آموزش مدل نهایی
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=best_k, random_state=42, n_init=10))
])
final_pipeline.fit(X)
df['label'] = final_pipeline.named_steps['kmeans'].labels_


# بارگذاری دادهها
df = pd.read_csv('mall_customers.csv')

# استخراج ویژگیها
X = df[['Annual_Income_(k$)', 'Spending_Score_(1-100)']]

# پیدا کردن بهترین تعداد خوشه
wcss = []
silhouette_scores = []
max_clusters = 10

for k in range(2, max_clusters + 1):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    pipeline.fit(X)
    labels = pipeline.named_steps['kmeans'].labels_
    wcss.append(pipeline.named_steps['kmeans'].inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

best_k = np.argmax(silhouette_scores) + 2
print(f"بهترین تعداد خوشه: {best_k} (Silhouette Score: {max(silhouette_scores):.2f})")

# آموزش مدل نهایی
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=best_k, random_state=42, n_init=10))
])
final_pipeline.fit(X)
df['label'] = final_pipeline.named_steps['kmeans'].labels_

# *********************************************
# ************** ذخیره خوشهها *****************
# *********************************************

# ایجاد پوشه برای ذخیره فایلها
output_dir = 'clusters'
os.makedirs(output_dir, exist_ok=True)

# ذخیره هر خوشه در یک فایل CSV جداگانه
for cluster in range(best_k):
    cluster_data = df[df['label'] == cluster]
    file_path = os.path.join(output_dir, f'cluster_{cluster}.csv')
    cluster_data.to_csv(file_path, index=False)
    print(f"خوشه {cluster} با {len(cluster_data)} نمونه در '{file_path}' ذخیره شد.")



# ذخیره مدل
joblib.dump(final_pipeline, 'kmeans_customer_segmentation.pkl')

# *********************************************
# ************** نمودارهای تعاملی **************
# *********************************************

# 1. نمودار Elbow تعاملی
fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(
    x=list(range(2, max_clusters + 1)),
    y=wcss,
    mode='lines+markers',
    name='WCSS'
))
fig_elbow.update_layout(
    title='Elbow Method (تعیین بهترین تعداد خوشه)',
    xaxis_title='تعداد خوشهها',
    yaxis_title='WCSS',
    template='plotly_white'
)
fig_elbow.show()

# 2. نمودار Silhouette تعاملی
fig_silhouette = go.Figure()
fig_silhouette.add_trace(go.Scatter(
    x=list(range(2, max_clusters + 1)),
    y=silhouette_scores,
    mode='lines+markers',
    name='Silhouette Score'
))
fig_silhouette.update_layout(
    title='Silhouette Score (ارزیابی کیفیت خوشهها)',
    xaxis_title='تعداد خوشهها',
    yaxis_title='امتیاز',
    template='plotly_white'
)
fig_silhouette.show()

# 3. نمودار خوشهبندی تعاملی
fig_cluster = px.scatter(
    df,
    x='Annual_Income_(k$)',
    y='Spending_Score_(1-100)',
    color='label',
    color_continuous_scale='viridis',
    title='Customer Segmentation (تعیین خوشهها)',
    labels={'Annual_Income_(k$)': 'درآمد سالانه (هزار دلار)', 'Spending_Score_(1-100)': 'امتیاز خرید'},
    hover_data=['CustomerID', 'Age', 'Gender']
)
fig_cluster.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig_cluster.show()

# 4. نمودار 3 بعدی تعاملی (با Age)
fig_3d = px.scatter_3d(
    df,
    x='Annual_Income_(k$)',
    y='Spending_Score_(1-100)',
    z='Age',
    color='label',
    color_continuous_scale='viridis',
    title='تجزیه و تحلیل 3 بعدی مشتریان',
    labels={'Annual_Income_(k$)': 'درآمد سالانه', 'Spending_Score_(1-100)': 'امتیاز خرید'}
)
fig_3d.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
fig_3d.show()


# 5. نمودار پویا برای تغییر تعداد خوشهها
@interact(k=IntSlider(min=2, max=10, step=1, value=best_k, description='تعداد خوشهها:'))
def plot_dynamic_clusters(k):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    pipeline.fit(X)
    labels = pipeline.named_steps['kmeans'].labels_

    fig = px.scatter(
        df,
        x='Annual_Income_(k$)',
        y='Spending_Score_(1-100)',
        color=labels,
        color_continuous_scale='viridis',
        title=f'خوشهبندی با {k} خوشه',
        labels={'Annual_Income_(k$)': 'درآمد سالانه', 'Spending_Score_(1-100)': 'امتیاز خرید'}
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    return fig

import pandas as pd
import joblib

# بارگذاری مدل
pipeline = joblib.load('kmeans_customer_segmentation.pkl')

# دادههای مشتری جدید (مثال)
new_customer = {
    'Annual_Income_(k$)': [160],  # درآمد سالانه به هزار دلار
    'Spending_Score_(1-100)': [65]  # امتیاز خرید بین 1 تا 100
}

# تبدیل به DataFrame
new_customer_df = pd.DataFrame(new_customer)

# پیشبینی خوشه
predicted_cluster = pipeline.predict(new_customer_df)[0]
print(f"مشتری جدید به خوشه {predicted_cluster} تعلق دارد.")

# *********************************************
# ************** نمایش جزئیات خوشه *************
# *********************************************

# بارگذاری دادههای اصلی برای تحلیل بیشتر
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_(k$)', 'Spending_Score_(1-100)']]

# مراکز خوشهها
centers = pipeline.named_steps['kmeans'].cluster_centers_

# فاصله تا مرکز خوشه
distance_to_center = ((new_customer_df.iloc[0] - centers[predicted_cluster])**2).sum()**0.5
print(f"فاصله تا مرکز خوشه: {distance_to_center:.2f}")

# نمایش مشتریان خوشه موردنظر
cluster_customers = df[pipeline.named_steps['kmeans'].labels_ == predicted_cluster]
print("\nمشتریان این خوشه:")
print(cluster_customers.head())