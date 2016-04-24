import math
import numpy as np
import matplotlib.pyplot as plt

import kivy
from kivy.garden.graph import Graph, MeshLinePlot, MeshStemPlot, SmoothLinePlot
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

import gui

class BiggerBox(BoxLayout):
    def __init__(self,w1,w2,orientation='horizontal'):
        super(BiggerBox, self).__init__(orientation=orientation)
        self.add_widget(w1)
        self.add_widget(w2)

class Grapher(BoxLayout):
    def __init__(self, n=3, xmin=0, xmax=10, ymin=-1, ymax=1, xlabel='Time'):
        """
        n: number of graphs
        xmin: minimum x
        xmax: maximum x
        ymin: minimum y
        ymax: maximum y
        xlabel: label for x axis (time of freq)
        """
        super(Grapher, self).__init__(orientation='vertical')
        self.graph = []
        self.plots = []
        if xlabel == 'Time':
            self.plots.append(MeshLinePlot(color=[1,0,0,1]))
            self.plots.append(MeshLinePlot(color=[0,1,0,1]))
            self.plots.append(MeshLinePlot(color=[0,0,1,1]))
        else:
            self.plots.append(MeshStemPlot(color=[1,0,0,1]))
            self.plots.append(MeshStemPlot(color=[0,1,0,1]))
            self.plots.append(MeshStemPlot(color=[0,0,1,1]))
        self.reset_plots()
        for i in range(n):
            self.graph.append(Graph(xlabel=xlabel, ylabel='Amp', x_ticks_minor=5,
            x_ticks_major=25, y_ticks_major=1,
            y_grid_label=True, x_grid_label=True, padding=5,
            x_grid=True, y_grid=True, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
            self.graph[i].add_plot(self.plots[i])
            self.add_widget(self.graph[i])

    def reset_plots(self):
        """
        resets plots
        """
        for plot in self.plots:
            plot.points = [(0,0)]

    def graph_points(self,p,signal):
        """
        adds points from signal to index p
        """
        self.plots[p].points = signal

class Main(App):
    def __init__(self, Ts):
        """
        Ts: sample time step
        """
        super(Main, self).__init__()
        self.interactive = gui.MainWidget(callback=self.visualize, size=[1200,200])
        self.Ts = Ts
        self.time = Grapher()
        self.fourier = Grapher(3,-0.5/self.Ts,0.5/self.Ts, 0, 1, 'Freq')

    def build(self):
        big_box = BiggerBox(self.time,self.fourier,'horizontal')
        biggest_box = BiggerBox(big_box, self.interactive, 'vertical')
        return biggest_box

    def modulate(self, signal, mod):
        if mod:
            return fm_test(10, self.Ts, 3, 2, 1, signal=signal)
        else:
            return am_test(10, self.Ts, 3, signal=signal)

    def visualize(self, signal, mod):
        """
        signal: data signal
        mod: am/fm (fm=1, am=0)
        """
        signal = self.modulate(signal, mod)
        for i in range(3):
            select = [(signal[0][x], signal[i+1][x]) for x in range(len(signal[0]))]
            self.time.graph_points(i,select)
            transform = view_freq(signal[i+1], self.Ts)
            self.fourier.graph_points(i, [(transform[0][x], transform[1][x]) for x in range(len(transform[0]))])


def plot_mods(times, data,carrier,modulated):
    """
    uses matplotlib to plot the modulating signals
    """
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

def fm_test(time, Ts, wc, kf, A, signal=None, data_fn=None):
    """
    time: length of signal
    Ts: discrete time sample interval
    wc: carrier frequency
    kf: hertz/volt variation from carrier frequency
    A: amplitude of output signal
    signal: data signal
    data_fn: function for data signal
    """
    times = np.arange(0,time,Ts)
    carrier = np.sin(np.multiply(times,2*np.pi*wc))
    integral = [0]
    if data_fn:
        fn = np.vectorize(data_fn)
        signal = fn(times)
        # for t in times:
        #     integral.append(integral[-1] + Ts*fn(t))
    for i,t in enumerate(times):
        integral.append(integral[-1] + Ts*signal[i])

    fm = np.multiply(A,np.cos(np.multiply(np.add(np.multiply(times,wc),np.multiply(integral[1:],kf)),2*np.pi)))

    return times, signal, carrier, fm

def am_test(time, Ts, wc, signal=None, data_fn=None):
    """
    time: length of signal
    Ts: discrete time sample interval
    wc: carrier frequency
    signal: data signal
    data_fn: function for data signal (must take array)
    """
    times = np.arange(0,time,Ts)
    carrier = np.sin(np.multiply(times,2*np.pi*wc))
    if data_fn:
        fn = np.vectorize(data_fn)
        signal = fn(times)

    am = np.multiply(signal, carrier)

    return times, signal, carrier, am

def view_freq(signal, Ts):
    """
    signal: signal vector to be plotted
    Ts: discrete time sample interval
    """
    signal = np.array(signal)
    n = len(signal) # length of the signal
    frq = np.fft.fftfreq(signal.size, Ts)

    Y = np.fft.fft(signal)/n # fft computing and normalization
    return frq, np.abs(Y)


if __name__ == '__main__':
    main = Main(0.01)
    main.run()
    # Main(0.01, fm_test(10, 0.01, 3, 2, 1, signal=signal)).run()
