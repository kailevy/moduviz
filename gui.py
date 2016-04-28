from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.slider import Slider
from kivy.graphics import Color, Bezier, Line
from kivy.core.window import WindowBase

import math

class MainWidget(Widget):

    def __init__(self, callback=None, *args, **kwargs):
        super(MainWidget, self).__init__(*args, **kwargs)

        self.callback = callback
        self.is_fm = True

        x = y = 300
        points = []
        for i in range(50,self.size[0]-150):
            points.extend([i, y])
        self.s = SignalDraw(points=points, size=[self.size[0]-100,self.size[1]])

        self.add_widget(self.s)

        self.b = Button(text='Submit', size=[100,100], pos=[self.size[0]-100,0])
        self.b.bind(on_press=self.submit)
        self.add_widget(self.b)

        self.dropdown = DropDown()

        btn = Button(text='Sine', size_hint_y=None, height=44)
        btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
        self.dropdown.add_widget(btn)

        btn = Button(text='Cosine', size_hint_y=None, height=44)
        btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
        self.dropdown.add_widget(btn)

        btn = Button(text='Square', size_hint_y=None, height=44)
        btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
        self.dropdown.add_widget(btn)

        self.am_button = Button(text='Frequency Modulation (FM)', size=[100,100], pos=[self.size[0]-100,100])
        self.am_button.bind(on_release=lambda btn: self.set_am_fm(self.am_button.text))
        self.add_widget(self.am_button)

        self.mainbutton = Button(text='Choose signal', size_hint=(None, None), size=[100,100], pos=[self.size[0]-100,200])
        self.mainbutton.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=self.dropdown_select)

        self.add_widget(self.mainbutton)

        self.slider = Slider(min=1, max=30, value=3*math.pi, pos=[self.size[0]-100,300])
        self.add_widget(self.slider)

    def set_am_fm(self, text):
        if text[0]=='F':
            setattr(self.am_button, 'text', 'Amplitude Modulation (AM)')
            self.is_fm = False
        else:
            setattr(self.am_button, 'text', 'Frequency Modulation (FM)')
            self.is_fm = True

    def dropdown_select(self, instance, x):
        setattr(self.mainbutton, 'text', x)


        if (x == "Sine"):
            data = []
            for i in range(1000):
                data.append(math.sin(2*math.pi*i/250))

        if (x == "Cosine"):
            data = []
            for i in range(1000):
                data.append(math.cos(2*math.pi*i/250))

        if (x == "Square"):
            data = []
            for i in range(1000):
                data.append(2*((i/200)%2)-1)

        self.s.set_data(data)

    def submit(self, instance):
        self.callback(self.s.data, self.is_fm)

class SignalDraw(Widget):

    def __init__(self, points=[], loop=False, *args, **kwargs):
        super(SignalDraw, self).__init__(*args, **kwargs)
        self.d = 10  # pixel tolerance when clicking on a point
        self.points = points
        self.loop = loop
        self.current_point = None  # index of point being dragged
        self.previous_touch = None
        self.previous_point = None
        self.previous_value = None
        self.data = []
        for i in range(1000):
            self.data.append(0)

        with self.canvas:
            Color(1.0, 0.0, 1.0)
            self.line = Line(
                    points=self.points,
                    close=False,
                    dash_offset=0,
                    dash_length=0)

        self.set_data(self.data)

    def set_data(self, data):
        self.data = data
        ps = []
        for i in range(len(self.data)):
            ps.extend([50+i/float(len(self.data))*(self.size[0]-200), self.size[1]/2*(1+0.9*self.data[i])])
        self.points = ps
        self.line.points = ps

    def compute_data(self):
        data = []
        i = 1
        while i<len(self.points):
            data.append((2*self.points[i]/self.size[1]-1)/0.9)
            i+=2
        mx = max(max(data), math.fabs(min(data)))
        if mx>1:
            for i in range(len(data)):
                data[i] = data[i]/mx
        self.data = data

    def on_touch_down(self, touch):
        self.previous_touch = touch
        if self.collide_point(touch.pos[0], touch.pos[1]):
            for i, p in enumerate(list(zip(self.points[::2],
                                           self.points[1::2]))):
                if (abs(touch.pos[0] - self.pos[0] - p[0]) < self.d):
                    self.current_point = i + 1
                    return True
            return super(SignalDraw, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.collide_point(touch.pos[0], touch.pos[1]):
            if self.current_point:
                self.current_point = None
                self.previous_point = None
                return True
            return super(SignalDraw, self).on_touch_up(touch)

    def on_touch_move(self, touch):
        if self.collide_point(touch.pos[0], touch.pos[1]):
            for i, p in enumerate(list(zip(self.points[::2],
                                           self.points[1::2]))):
                if (abs(touch.pos[0] - self.pos[0] - p[0]) < self.d):
                    self.current_point = i + 1
            c = self.current_point
            p = self.previous_point
            if c:
                current_value = touch.pos[1] - self.pos[1]
                previous_value = self.previous_value
                if p and previous_value:
                    mn = min(p,c)
                    mx = max(p,c)
                    for point in range(mn,mx+1):
                        if (mx == mn):
                            self.points[(c - 1) * 2 + 1] = current_value
                        else:
                            self.points[(point - 1) * 2 + 1] = (current_value - previous_value)/(c-p)*(point-p)+previous_value
                self.line.points = self.points
                self.previous_point = self.current_point
                self.previous_value = current_value
                self.compute_data()
                return True
            return super(SignalDraw, self).on_touch_move(touch)


class Main(App):

    def build(self):
        size = [800,300]
        return MainWidget(callback=self.thing, size=size, pos=[0,0])

    def thing(self, data, is_fm):
        print(len(data))
        print(is_fm)

if __name__ == '__main__':
    Main().run()
