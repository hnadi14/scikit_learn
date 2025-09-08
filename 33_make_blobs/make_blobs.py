import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors  # اضافه کردن این خط

# تولید داده‌های آزمایشی
centers = [[1, 1], [-1, -1], [1, -1],[2,2]]
X, y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
X_scaled = StandardScaler().fit_transform(X)


# تابع تعیین خودکار پارامترها
def find_optimal_eps(X, min_samples):
    neighbors = min_samples - 1
    nn = NearestNeighbors(n_neighbors=neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors()
    distances = np.sort(distances[:, -1], axis=0)

    # اضافه کردن خط زیر برای برگرداندن kneedle
    kneedle = KneeLocator(range(len(distances)), distances,
                          S=1.0, curve="convex", direction="increasing")
    optimal_eps = distances[kneedle.elbow] if kneedle.elbow else 0.3
    return optimal_eps, distances, kneedle  # اضافه کردن kneedle به خروجی


# اعمال تابع و دریافت kneedle
min_samples = 4
optimal_eps, distances, kneedle = find_optimal_eps(X_scaled, min_samples)  # تغییر این خط

# رسم نمودار k-distance
plt.figure(figsize=(12, 6))
plt.plot(distances)
plt.axvline(x=kneedle.elbow, color='r', linestyle='--',
            label=f'Optimal eps={optimal_eps:.2f}')
plt.title('k-distance Graph')
plt.xlabel('Points sorted by distance')
plt.ylabel('k-th nearest neighbor distance')
plt.legend()
plt.show()

# بقیه کد (DBSCAN و نمودارها) بدون تغییر
db = DBSCAN(eps=optimal_eps, min_samples=min_samples).fit(X_scaled)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"تعداد خوشه‌ها: {n_clusters}")
print(f"تعداد نویزها: {n_noise}")
print(f"Silhouette Score: {metrics.silhouette_score(X_scaled, labels):.2f}")