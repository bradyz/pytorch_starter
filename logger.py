import visdom
import numpy as np


class Logger(object):
    def __init__(self, use_visdom):
        self.vis = visdom.Visdom() if use_visdom else None
        self.use_visdom = use_visdom

        self.epoch = 0

        self.metrics = {
                'loss_train', 'loss_test', 'accuracy_train', 'accuracy_test'}
        self.plots = {
                name: ScalarPlot(self.vis, name, use_visdom)
                for name in self.metrics}

    def update(self, **kwargs):
        for key, val in kwargs.items():
            self.plots[key].draw(val)

    def draw(self):
        if not self.use_visdom:
            return

        for _, val in self.plots.items():
            val.draw()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def should_save(self):
        return True

    def load_state_dict(self, state):
        self.epoch = state['epoch']

        for name, plot in self.plots.items():
            plot.values = state[name]

    def state_dict(self):
        state = dict()

        state['epoch'] = self.epoch

        for name, plot in self.plots.items():
            state[name] = plot.values

        return state


class ScalarPlot(object):
    def __init__(self, vis, title, use_visdom):
        self.vis = vis
        self.use_visdom = use_visdom

        self.title = title
        self.values = list()

    def draw(self, y=None):
        if y is not None:
            self.values.append(y)

        if not self.values:
            return
        elif not self.use_visdom:
            return

        self.vis.line(
                X=np.float32(list(range(len(self.values)))),
                Y=np.float32(self.values),
                win=self.title,
                opts=dict(title=self.title))
