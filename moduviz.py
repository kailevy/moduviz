import math
import numpy as np
import matplotlib.pyplot as plt

def fm_test(data_fn, step, wc, kf, A, time):
    """
    data_fn: function for data signal
    step: discrete time sample rate
    wc: carrier frequency
    kf: hertz/volt variation from carrier frequency
    A: amplitude of output signal
    time: length of signal
    """
    fm = []
    carrier = []
    integral = [0]

    times = np.arange(0,time,step)
    carrier = np.sin(np.multiply(times,2*np.pi*wc))

    for t in times:
        integral.append(integral[-1] + step*data_fn(t))

    fm = np.multiply(A,np.cos(np.multiply(np.add(np.multiply(times,wc),np.multiply(integral[1:],kf)),2*np.pi)))

    plt.plot(times,data_fn(times))
    plt.figure()
    plt.plot(times,carrier)
    plt.figure()
    plt.plot(times,fm)
    plt.show()

def am_test(data_fn, time, step, wc):
    """
    data_fn: function for data signal (must take array)
    time: length of signal
    step: discrete time sample rate
    wc: carrier frequency
    """
    times = np.arange(0,time,step)
    carrier = np.cos(np.multiply(times,2*np.pi*wc))
    am = np.multiply(data_fn(times),carrier)

    plt.plot(times,data_fn(times))
    plt.figure()
    plt.plot(times,carrier)
    plt.figure()
    plt.plot(times,am)
    plt.show()

if __name__ == '__main__':
    fm_test(np.sin, 0.001, 3, 2, 1, 10)
