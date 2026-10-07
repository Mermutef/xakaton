# clustering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings

warnings.filterwarnings('ignore')
from time import time


def load_master(filepath='data/master_features.csv'):
    df = pd.read_csv(filepath)
    # Гарантируем, что все данные числовые
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in non_numeric:
        if col == 'ЛС':
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    ids = df['ЛС']
    feats = df.drop(columns=['ЛС'])
    return ids, feats


def perform_clustering(feats, n_clusters_range=range(5, 10), random_state=42, batch_size=10000):
    # Масштабирование
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)

    # PCA для ускорения (сохраним 95% дисперсии)
    pca = PCA(n_components=0.95, random_state=random_state)
    reduced = pca.fit_transform(scaled)
    print(f"PCA reduced dimensions from {scaled.shape[1]} to {reduced.shape[1]}")

    results = {}
    for n in n_clusters_range:
        t0 = time()
        kmeans = MiniBatchKMeans(n_clusters=n, random_state=random_state,
                                 batch_size=batch_size, n_init=3, max_iter=100)
        labels = kmeans.fit_predict(reduced)
        t1 = time()
        sil = silhouette_score(reduced, labels, sample_size=50000)  # оценка на подвыборке
        results[n] = {'model': kmeans, 'labels': labels, 'silhouette': sil}
        print(f"n_clusters={n}, silhouette={sil:.4f} (time {t1 - t0:.1f}s)")

    best_n = max(results, key=lambda k: results[k]['silhouette'])
    best_model = results[best_n]['model']
    best_labels = results[best_n]['labels']
    print(f"Selected n_clusters={best_n} (silhouette={results[best_n]['silhouette']:.4f})")
    return best_model, best_labels, scaler, pca, best_n


if __name__ == '__main__':
    ids, feats = load_master()
    print(f"Features shape: {feats.shape}")
    model, labels, scaler, pca, best_n = perform_clustering(feats)
    # Сохраняем
    joblib.dump(model, 'models/kmeans.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca.pkl')
    cluster_df = pd.DataFrame({'ЛС': ids, 'cluster': labels})
    cluster_df.to_csv('data/clusters.csv', index=False)
    # Центры кластеров в исходном пространстве
    # Восстанавливаем центры из MiniBatchKMeans (находятся в reduced space)
    centers_reduced = model.cluster_centers_
    centers_orig = scaler.inverse_transform(pca.inverse_transform(centers_reduced))
    centers_df = pd.DataFrame(centers_orig, columns=feats.columns)
    centers_df.to_csv('data/cluster_centers.csv', index=False)
    print("Clustering done.")
