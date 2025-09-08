import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# خواندن و پیشپردازش داده‌ها
movies = pd.read_csv('movies.csv')
movies.drop_duplicates(inplace=True)
movies.dropna(inplace=True)
movies.reset_index(drop=True, inplace=True)

# تفکیک و رمزنگاری ژانرها
genres = movies['genres'].str.split('|').apply(lambda x: list(set(x)))
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(genres)
feature_names = mlb.classes_

# محاسبه TF-IDF
tfidf_transformer = TfidfTransformer()
genres_tfidf = tfidf_transformer.fit_transform(genres_encoded)

# کاهش ابعاد با PCA و TruncatedSVD
pca = PCA(n_components=2, random_state=42)
svd = TruncatedSVD(n_components=2, random_state=42)

# برای PCA نیاز به داده‌های نرمال‌شده و چگال
scaler = StandardScaler(with_mean=False)
genres_tfidf_scaled = scaler.fit_transform(genres_tfidf)
genres_pca = pca.fit_transform(genres_tfidf_scaled.toarray())

# برای TruncatedSVD مستقیم روی داده‌های خام TF-IDF
genres_svd = svd.fit_transform(genres_tfidf)


# تابع رسم نمودارهای مقایسه‌ای
# تابع رسم نمودارهای مقایسه‌ای (اصلاح شده)
def plot_dimension_reduction(embedding, title, clusters=None, highlight_index=None, indices=None):
    plt.figure(figsize=(14, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=clusters, palette='viridis', s=100, legend='brief')

    if highlight_index is not None and indices is not None:
        plt.scatter(embedding[highlight_index, 0], embedding[highlight_index, 1], s=200,
                    facecolors='none', edgecolors='r', linewidth=2, label='Test Movie')
        neighbors = indices[0][1:]
        plt.scatter(embedding[neighbors, 0], embedding[neighbors, 1], s=150,
                    facecolors='none', edgecolors='magenta', linewidth=2, label='Recommendations')

    plt.title(f'{title} - Movie Clusters')
    plt.xlabel(f'{title} Component 1')
    plt.ylabel(f'{title} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for i, txt in enumerate(movies['title']):
        plt.annotate(txt[:20], (embedding[i, 0], embedding[i, 1]), fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.show()





# انتخاب خودکار تعداد خوشه‌ها برای K-Means
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(genres_tfidf)
    score = silhouette_score(genres_tfidf, clusters)
    silhouette_scores.append(score)
optimal_k = K_range[np.argmax(silhouette_scores)]

# مدل‌سازی K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters_kmeans = kmeans.fit_predict(genres_tfidf)


# تابع رسم نمودار Silhouette
def plot_silhouette(X, cluster_labels):
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(optimal_k):
        ith_cluster_sil_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_sil_values.sort()
        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / optimal_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_sil_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    plt.show()


# تابع تست فیلم (اصلاح شده)
def test_movie_recommendations(movie_index):
    print(f"\n----- Testing Movie #{movie_index} -----")
    print("Title:", movies.iloc[movie_index]['title'])
    print("Genres:", movies.iloc[movie_index]['genres'])

    # محاسبه فیلم‌های مشابه
    recommender = NearestNeighbors(n_neighbors=6, metric='cosine')
    recommender.fit(genres_tfidf)
    distances, indices = recommender.kneighbors(genres_tfidf[movie_index].reshape(1, -1))

    print("\nTop 5 Recommendations:")
    recommendations = movies.iloc[indices[0][1:]][['title', 'genres']]
    print(recommendations.to_string())

    # رسم نمودارهای توصیه‌گر (اضافه کردن indices)
    plot_dimension_reduction(genres_pca, 'PCA with Recommendations',
                             clusters_kmeans,
                             highlight_index=movie_index,
                             indices=indices)  # انتقال indices

    plot_dimension_reduction(genres_svd, 'TruncatedSVD with Recommendations',
                             clusters_kmeans,
                             highlight_index=movie_index,
                             indices=indices)  # انتقال indices

# تابع ارزیابی کامل
def evaluate_model():
    # گزارش خوشه‌بندی
    print(f"Optimal Number of Clusters: {optimal_k}")
    print("Cluster Distribution:")
    print(pd.Series(clusters_kmeans).value_counts().sort_index())

    # نمودار Silhouette
    plot_silhouette(genres_tfidf, clusters_kmeans)

    # تست فیلم شماره 5 (اگر وجود داشته باشد)
    if len(movies) >= 5:
        test_movie_recommendations(0)  # ایندکس 4 برای فیلم پنجم
    else:
        print("Not enough movies to test index 5")


# اجرای ارزیابی
evaluate_model()