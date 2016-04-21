import math
import numpy as np
import matplotlib.pyplot as plt

def square(t):
    freq = 2
    return 2*(int(freq*t)%2)-1

def plot_mods(times, data,carrier,modulated):
    plt.subplot(3,1,1)
    plt.plot(times, data)
    plt.title('Data Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.subplot(3,1,2)
    plt.plot(times, carrier)
    plt.title('Carrier Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.subplot(3,1,3)
    plt.plot(times, modulated)
    plt.title('Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def fm_test(data_fn, time, step, wc, kf, A):
    """
    data_fn: function for data signal
    time: length of signal
    step: discrete time sample rate
    wc: carrier frequency
    kf: hertz/volt variation from carrier frequency
    A: amplitude of output signal
    """
    fn = np.vectorize(data_fn)
    times = np.arange(0,time,step)
    carrier = np.sin(np.multiply(times,2*np.pi*wc))
    integral = [0]

    for t in times:
        integral.append(integral[-1] + step*fn(t))
    fm = np.multiply(A,np.cos(np.multiply(np.add(np.multiply(times,wc),np.multiply(integral[1:],kf)),2*np.pi)))

    return times, fn(times), carrier, fm

def am_test(data_fn, time, step, wc):
    """
    data_fn: function for data signal (must take array)
    time: length of signal
    step: discrete time sample rate
    wc: carrier frequency
    """
    fn = np.vectorize(data_fn)
    times = np.arange(0,time,step)
    carrier = np.cos(np.multiply(times,2*np.pi*wc))
    am = np.multiply(fn(times),carrier)

    return times, fn(times), carrier, am

if __name__ == '__main__':
    plot_mods(*am_test(np.sin, 10, 0.001, 3))
    plot_mods(*fm_test(np.sin, 10, 0.001, 3, 2, 1))
