import numpy as np
import matplotlib.pyplot as plt

def draw(order=1):
    plt.style.use('ggplot')
    plt.figure()
    points = hilbert(order).T
    plt.plot(points[0]+.5, points[1]+.5)
    plt.xlim(0, 2**order)
    plt.ylim(0, 2**order)
    plt.xticks([])
    plt.yticks([])
    # plt.xticks(np.arange(2**order+1))
    # plt.yticks(np.arange(2**order+1))
    plt.title(f'Order {order} Peusdo Hilbert Curve')
    plt.grid(True)
    plt.show()

def hilbert(order=1):
    """iteratively construct hilbert coodbook"""
    N = 2**(2*order)
    base_points = np.array([[0,0], [0,1], [1,1], [1,0]])
    points = np.tile(base_points, (int(N/4),1))
    indexes = np.arange(N)
    if order == 1:
        return points
    for i in range(2, order+1):
        for j, index in enumerate(indexes):
            index = index >> 2 # right shift 2 bits 
            indexes[j] = index
            last_2_bits = index & 3 # bit mask last 2 bits
            len = 2**(i-1) 
            if last_2_bits == 0:
                points[j, 0], points[j, 1] = points[j, 1], points[j, 0] # rotate clockwise if last_2_bits==00 
            elif last_2_bits == 1:
                points[j, 1] += len
            elif last_2_bits == 2:
                points[j, 0] += len
                points[j, 1] += len
            elif last_2_bits == 3:
                points[j, 0], points[j, 1] = len-1-points[j, 1], len-1-points[j, 0] # rotate counterclockwise if last_2_bits==11
                points[j, 0] += len 
    return points

def main():
    draw(int(input()))

if __name__ == '__main__':
    main()