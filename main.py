import numpy as np
import matplotlib.pyplot as plt

def main():
    
    x_train = get_x_train()
    y_train = get_y_train(x_train)
    
    w = np.dot(np.linalg.pinv(x_train), y_train)

    print(w)

def get_x_train():
    x1 = np.random.uniform(-2, 10, 12)
    bias = np.ones_like(x1)
    return np.column_stack((bias, x1))


def get_y_train(x):
    return np.square(x) + 10


if __name__ == '__main__':
    main()
 