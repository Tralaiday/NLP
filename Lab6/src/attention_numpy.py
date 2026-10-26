import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray
):
    d_k = K.shape[1]
    S = np.dot(Q, K.T) / np.sqrt(d_k)
    A = __softmax(S)
    return np.dot(A, V)


def __softmax(S: np.ndarray):
    results = [(np.exp(s) / np.sum(np.exp(S))) for s in S]
    return np.array(results)

if __name__ == '__main__':
    Q = np.array([[1, 0]])
    K = np.array([[1, 0], [0, 1], [1, 1]])
    V = np.array([[1, 0], [0, 1], [2, 2]])
    print(Q.shape, K.shape, V.shape)
    print(scaled_dot_product_attention(Q, K, V))