import math
import numpy as np
import matplotlib.pyplot as plt

import kivy
from kivy.garden.graph import Graph, MeshLinePlot, MeshStemPlot, SmoothLinePlot
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class Grapher(BoxLayout):
    def __init__(self, n=3, xmin=0, xmax=10, ymin=-1, ymax=1, xlabel='Time'):
        super(Grapher, self).__init__(orientation='vertical')
        self.graph = []
        self.plots = []
        self.plots.append(MeshStemPlot(color=[1,0,0,1]))
        self.plots.append(MeshLinePlot(color=[0,1,0,1]))
        self.plots.append(MeshLinePlot(color=[0,0,1,1]))
        self.reset_plots()
        for i in range(n):
            self.graph.append(Graph(xlabel=xlabel, ylabel='Amp', x_ticks_minor=5,
            x_ticks_major=25, y_ticks_major=1,
            y_grid_label=True, x_grid_label=True, padding=5,
            x_grid=True, y_grid=True, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
            self.graph[i].add_plot(self.plots[i])
            self.add_widget(self.graph[i])

    def reset_plots(self):
        for plot in self.plots:
            plot.points = [(0,0)]

    def graph_points(self,p,signal):
        self.plots[p].points = signal

    def get_graphs(self):
        return self.graph

class Main(App):

    def build(self):
        Ts = 0.01
        time = Grapher()
        fourier = Grapher(3,-0.5/Ts,0.5/Ts, 0, 1, 'Freq')
        graphs = [time, fourier]
        signal = np.array(fm_test(np.sin, 10, Ts, 3, 2, 1))
        # sel = [0,3]
        # signal = list(signal[sel])
        for i in range(3):
            select = [(signal[0][x], signal[i+1][x]) for x in range(len(signal[0]))]
            time.graph_points(i,select)
            transform = view_freq(signal[i+1], Ts)
            fourier.graph_points(i, [(transform[0][x], transform[1][x]) for x in range(len(transform[0]))])

        return graphs[1]


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
    step: discrete time sample interval
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
    step: discrete time sample interval
    wc: carrier frequency
    """
    fn = np.vectorize(data_fn)
    times = np.arange(0,time,step)
    carrier = np.cos(np.multiply(times,2*np.pi*wc))
    am = np.multiply(fn(times),carrier)

    return times, fn(times), carrier, am

def view_freq(signal, step):
    """
    signal: signal vector to be plotted
    step: discrete time sample interval
    """
    n = len(signal) # length of the signal
    frq = np.fft.fftfreq(signal.size, step)

    Y = np.fft.fft(signal)/n # fft computing and normalization

    # plt.stem(frq,np.abs(Y),'r') # plotting the spectrum
    # plt.xlabel('Freq (Hz)')
    # plt.ylabel('|Y(freq)|')
    # plt.show()
    return frq, np.abs(Y)


if __name__ == '__main__':
    Main().run()
    # view_freq(am_test(np.sin, 10, 0.01, 3)[2], 0.01)
    # plot_mods(*am_test(np.sin, 10, 0.001, 3))
    # plot_mods(*fm_test(np.sin, 10, 0.001, 3, 2, 1))
