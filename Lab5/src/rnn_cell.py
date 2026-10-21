import numpy as np

class SimpleRNNCell:

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

    def __tanh(self, x):
        return np.tanh(x)

    def forward(self, x_t, h_prev):
        linear_combination = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
        h_t = self.__tanh(linear_combination)
        return h_t

if __name__ == '__main__':
    batch_size = 2
    hidden_size = 5
    input_size = 10

    x_t = np.random.randn(input_size, batch_size)
    h_prev = np.zeros((hidden_size, batch_size))

    RNNCell = SimpleRNNCell(input_size, hidden_size)

    h_t = RNNCell.forward(x_t, h_prev)

    print(f"Kích thước của x_t: {x_t.shape}")
    print(f"Kích thước của h_prev: {h_prev.shape}")
    print(f"Kích thước đầu ra h_t: {h_t.shape}")