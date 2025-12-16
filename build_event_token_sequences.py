import numpy as np
from sklearn.cluster import MiniBatchKMeans

X_train = np.load("data/processed/X_train.npy")  # (N,L,D)
X_test  = np.load("data/processed/X_test.npy")

N, L, D = X_train.shape
K = 512  # vocab size (try 256 or 512 for speed)

kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=42)
kmeans.fit(X_train.reshape(-1, D))

X_train_tok = kmeans.predict(X_train.reshape(-1, D)).reshape(N, L)
X_test_tok  = kmeans.predict(X_test.reshape(-1, D)).reshape(X_test.shape[0], L)

np.save("data/processed/X_train_tok.npy", X_train_tok)
np.save("data/processed/X_test_tok.npy", X_test_tok)
