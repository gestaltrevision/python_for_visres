import os, glob
import psychopy
from psychopy import event, logging, monitors, visual
import numpy as np

from psychopy_ext import exp

class Experiment(exp.Experiment):

    def setup(self):
        """
        Initializes the experiment.

        A random seed is set for `random` and `numpy.random`. The seed
        is set using the 'set:time' option.
        Also, runtime information is fully recorded, log file is set
        and a window is created.
        """
        self.runtime_info = psychopy.info.RunTimeInfo(author='JonasKubilius',
                version='0.1', verbose=True, win=False, randomSeed='set:time')
        key, value = exp.get_version()
        self.runtime_info[key] = value  # updates with psychopy_ext version

        self._set_keys_flat()
        self.seed = int(self.runtime_info['experimentRandomSeed.string'])
        np.random.seed(self.seed)

        self.set_logging(self.paths['logs'] + self.info['subjid'])
        self.create_win(debug=self.rp['debug'])
        self.mouse = event.Mouse(win=self.win)
        self._initialized = True

    def create_win(self, debug=False, color='DimGray', units='deg', **kwargs):
        """Generates a :class:`psychopy.visual.Window` for presenting stimuli.

        :Kwargs:
            - debug (bool, default: False)
                - If True, then the window is half the screen size.
                - If False, then the windon is full screen.
            - color (str, str with a hexadecimal value, or a tuple of 3 values, default: "DimGray')
                Window background color. Default is dark gray. (`See accepted
                color names <http://www.w3schools.com/html/html_colornames.asp>`_
        """
        current_level = logging.getLevel(logging.console.level)
        logging.console.setLevel(logging.ERROR)
        monitor = monitors.Monitor(self.computer.name,
            distance=self.computer.distance,
            width=self.computer.width)
        logging.console.setLevel(current_level)
        res = exp.get_mon_sizes(self.computer.screen)
        monitor.setSizePix(res)

        if 'size' not in kwargs:
            try:
                kwargs['size'] = self.computer.win_size
            except:
                if not debug:
                    kwargs['size'] = tuple(res)
                else:
                    kwargs['size'] = (res[0]/2, res[1]/2)

        self.win = visual.Window(
            size=kwargs['size'],
            monitor = monitor,
            units = units,
            fullscr = not debug,
            allowGUI = debug, # mouse will not be seen unless debugging
            color = color,
            winType = 'pyglet',
            screen = self.computer.screen,
            viewScale = self.computer.view_scale
        )
