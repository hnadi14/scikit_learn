import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN  # اضافه کردن DBSCAN
from kneed import KneeLocator  # برای شناسایی خودکار نقطه آرنج

# خواندن داده‌ها
data = pd.read_csv('mall_customers.csv')
X = data[['Annual_Income_(k$)', 'Spending_Score_(1-100)']]

# استانداردسازی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# محاسبه ماتریس لینکیج
linked = linkage(X_scaled, method='ward')

# تعیین خودکار تعداد خوشه
last = linked[-10:, 2]
last_rev = last[::-1]
acceleration = np.diff(last_rev)
knee = np.argmax(acceleration) + 2 if len(acceleration) > 0 else 2

# محاسبه آستانه
threshold = linked[-knee, 2] if knee > 0 else 0

# رسم دندوگرام
plt.figure(figsize=(12, 6))
dendrogram(linked, orientation='top', distance_sort='descending')
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold:.1f}')
plt.legend()
plt.show()

# خوشه‌بندی سلسله مراتبی
agg_cluster = AgglomerativeClustering(n_clusters=knee, linkage='ward')
data['Agglomerative_Cluster'] = agg_cluster.fit_predict(X_scaled)

# خوشه‌بندی K-Means
kmeans = KMeans(n_clusters=knee, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)



# تابع تعیین خودکار eps
def find_optimal_eps(X, min_samples=5):
    # محاسبه فاصله k-NN
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances[:, -1], axis=0)

    # شناسایی نقطه آرنج با KneeLocator
    kneedle = KneeLocator(
        range(len(distances)),
        distances,
        S=1.0,  # حساسیت (مقدار بیشتر → حساسیت بیشتر)
        curve="convex",
        direction="increasing"
    )

    optimal_eps = distances[kneedle.elbow] if kneedle.elbow else 0.5
    return optimal_eps, distances


min_samples = 5  # معمولاً برای داده‌های 2 بعدی، min_samples=2*dim یا 5
optimal_eps, distances = find_optimal_eps(X_scaled, min_samples=min_samples)
print(optimal_eps)
# خوشه‌بندی DBSCAN با پارامترهای خودکار
dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
data['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# رسم نمودار KNN Distance برای انتخاب eps
plt.figure(figsize=(12, 6))
plt.plot(np.sort(distances))
plt.axvline(x=np.argmax(np.diff(distances)) + 1, color='r', linestyle='--')
plt.title('KNN Distance Plot for Optimal eps')
plt.xlabel('Points sorted by distance')
plt.ylabel('k-distance')
plt.show()

# تبدیل داده‌ها به آرایه نامپای برای نمایش
X_np = np.array(X)

# رسم نمودار پراکندگی برای هر سه روش
plt.figure(figsize=(18, 6))

# نمودار خوشه‌بندی سلسله مراتبی
plt.subplot(1, 3, 1)
for i in range(knee):
    mask = (data['Agglomerative_Cluster'] == i).values
    plt.scatter(X_np[mask, 0], X_np[mask, 1], label=f'Cluster {i+1}')
plt.title('Agglomerative Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

# نمودار K-Means
plt.subplot(1, 3, 2)
for i in range(knee):
    mask = (data['KMeans_Cluster'] == i).values
    plt.scatter(X_np[mask, 0], X_np[mask, 1], label=f'Cluster {i+1}')
plt.title('K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

# نمودار DBSCAN
plt.subplot(1, 3, 3)
unique_clusters = np.unique(data['DBSCAN_Cluster'])
for cluster in unique_clusters:
    if cluster == -1:
        # نویزها (Noise Points) را با رنگ خاکستری نمایش می‌دهیم
        mask = (data['DBSCAN_Cluster'] == cluster).values
        plt.scatter(X_np[mask, 0], X_np[mask, 1], color='gray', label='Noise')
    else:
        mask = (data['DBSCAN_Cluster'] == cluster).values
        plt.scatter(X_np[mask, 0], X_np[mask, 1], label=f'Cluster {cluster+1}')
plt.title('DBSCAN Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.tight_layout()
plt.show()