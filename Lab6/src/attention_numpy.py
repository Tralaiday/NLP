import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray
):
    d_k = K.shape[1]
    S = np.dot(Q, K.T) / np.sqrt(d_k)
    A = __softmax(S)
    C = np.dot(A, V)
    return S, A, C


def __softmax(S: np.ndarray):
    shifted = S - np.max(S, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

if __name__ == '__main__':
    Q = np.array([[1, 0]])
    K = np.array([[1, 0], [0, 1], [1, 1]])
    V = np.array([[1, 0], [0, 1], [2, 2]])
    print(Q.shape, K.shape, V.shape)
    scores, attention, context = scaled_dot_product_attention(Q, K, V)
    print("Scores:", scores)
    print("Attention:", attention)
    print("Context:", context)