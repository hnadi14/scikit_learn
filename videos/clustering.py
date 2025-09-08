
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ۱. بارگذاری داده
df = pd.read_csv('pbVideos.csv')
# ۲. پاکسازی داده
# حذف ردیف‌های با مقادیر منفی یا صفر
df = df[(df['Calculated_Views'] > 0) & (df['Duration_in_Seconds'] > 0)]

# ۳. استانداردسازی داده‌ها
features = ['Calculated_Views', 'Duration_in_Seconds']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ۴. تعیین تعداد خوشه بهینه
# روش Elbow
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(12, 5))
sns.lineplot(x=K_range, y=inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# روش Silhouette برای K=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score for K=3: {silhouette_avg:.2f}')

# ۵. آموزش مدل نهایی
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ۶. تفسیر خوشه‌ها
cluster_profile = df.groupby('Cluster')[features].mean().reset_index()
print("مشخصات خوشه‌ها:")
print(cluster_profile)

# ۷. افزودن برچسب‌های خوانا
df['ClusterLabel'] = df['Cluster'].map({
    0: 'Medium Engagement',
    1: 'High Engagement',
    2: 'Low Engagement'
})

# ۸. ذخیره نتایج
df.to_csv('Clustered_Videos.csv', index=False)

# ۹. نمودار نهایی خوشه‌ها
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x='Calculated_Views',
    y='Duration_in_Seconds',
    hue='ClusterLabel',
    data=df,
    palette='viridis'
)
plt.title('Clustering of Videos by Views and Duration')
plt.xlabel('Calculated Views')
plt.ylabel('Duration (Seconds)')
plt.grid(True)
plt.show()