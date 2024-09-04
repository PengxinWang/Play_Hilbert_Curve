import numpy as np
import matplotlib.pyplot as plt

def draw(order=1):
    plt.style.use('ggplot')
    plt.figure()
    points = hilbert(order).T
    plt.plot(points[0]+.5, points[1]+.5)
    # plt.xlim(0, order+1)
    # plt.ylim(0, order+1)
    # plt.xticks([])
    # plt.yticks([])
    plt.xticks(np.arange(order+2))
    plt.yticks(np.arange(order+2))
    plt.title(f'Order {order} Peusdo Hilbert Curve')
    plt.grid(True)
    plt.show()

def hilbert(order=1):
    if order == 1:
        points = np.array([[0,0], [0,1], [1,1], [1,0]])
    else:
        N = 2**(2*order)
        points = np.zeros((N, 2))
        for i in range(N):
            for _ in range(order):
                i = i >> 2
                index = i & 3
                if index == 0:
                    continue
                elif index == 1:
                    points[i][0] += (order-1)**2
                elif index == 2:
                    points[i][0] += (order-1)**2
                    points[i][1] += (order-1)**2
                else:
                    points[i][1] += (order-1)**2           
    return points

def main():
    draw(int(input()))

if __name__ == '__main__':
    main()