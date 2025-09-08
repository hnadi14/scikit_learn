import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.metrics.pairwise import euclidean_distances

# *********************************************
# ************** محاسبه فاصله اقلیدسی *************
# *********************************************
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]

# محاسبه ماتریس فاصله بین a و b
euclidean_dist = euclidean_distances(a, b)
print("فاصله اقلیدسی بین a و b:")
print(euclidean_dist)

# *********************************************
# ************** خوشهبندی سلسلهمراتبی ************
# *********************************************
x = np.array([
    [5, 3],
    [6, 4],
    [15, 12],
    [5, 3]
])

# محاسبه ماتریس لینکیج با روش Average Linkage
linked = shc.linkage(x, method='ward')

print("\nماتریس لینکیج:")
print(linked)

# *********************************************
# ************** رسم دندروگرام *****************
# *********************************************
plt.figure(figsize=(10, 5))
plt.title("دندروگرام خوشهبندی سلسلهمراتبی")
plt.xlabel("شماره نمونه")
plt.ylabel("فاصله")

# رسم دندروگرام با برچسبهای دادهها
dend = shc.dendrogram(
    linked,
    # labels=['a', 'b', 'c', 'd'],
    leaf_rotation=90,
    leaf_font_size=12,
    color_threshold=0.7*max(linked[:,2])
)

plt.axhline(y=6, color='r', linestyle='--', label='آستانه خوشهبندی')
plt.legend()
plt.tight_layout()
plt.show()