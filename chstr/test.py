import numpy as np





if __name__ == '__main__':

    dt = 0.001
    t = np.arange(-50, 50, dt)

    a = -0.9

    print(a**t)