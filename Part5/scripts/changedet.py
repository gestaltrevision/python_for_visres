import os

import numpy.random as rnd          # for random number generators

from psychopy import visual, core, event
from psychopy_ext import exp

from collections import OrderedDict

import fix
import computer
PATHS = exp.set_paths('change_detection', computer)
PATHS['images'] = '../Part2/images/'


class ChangeDet(fix.Experiment):
    """
    Change Detection Experiment
    ===========================

    In this experiment you will see photographs flickering with a tiny detail in them changing.
    Your task is to detect where the change is occuring.
    To make it harder, there are bubbles randomly covering the part of the photos.

    Hit **spacebar to begin**. When you detect a change, hit **spacebar** again.
    """
    def __init__(self,
                 name='exp',
                 info=OrderedDict([('exp_name', 'Change Detection'),
                                   ('subjid', 'cd_'),
                                   ('gender', ('male', 'female')),
                                   ('age', 18),
                                   ('left-handed', False)
                                  ]),
                 rp=None,
                 actions='run',
                 order='sequential'
                 ):
        super(ChangeDet, self).__init__(name=name, info=info,
                rp=rp, actions=actions,
                paths=PATHS, computer=computer)

        # user-defined parameters
        self.imlist = ['1','2','3','4','5','6']  # image names without the suffixes
        self.asfx = 'a.jpg'  # suffix for the first image
        self.bsfx = 'b.jpg'  # suffix for the second image
        self.scrsize = (900, 600)  # screen size in px
        self.stimsize = (9, 6)  # stimulus size in degrees visual angle
        self.timelimit = 30  # sec
        self.n_bubbles = 40
        self.changetime = .500  #sec

        self.computer.valid_responses = {'space': 1}

        self.trial_instr = ('Press spacebar to start the trial.\n\n'
                            'Hit spacebar again when you detect a change.')

    def create_win(self, *args, **kwargs):
        super(ChangeDet, self).create_win(size=self.scrsize, units='deg',
                                          *args, **kwargs)

    def create_stimuli(self):
        """Define your stimuli here, store them in self.s
        """
        self.s = {}
        self.s['bitmap1'] = visual.ImageStim(self.win, size=self.stimsize)
        self.s['bitmap2'] = visual.ImageStim(self.win, size=self.stimsize)
        self.s['bubble'] = visual.Circle(self.win, fillColor='black', lineColor='black')

    def create_trial(self):
        """Define trial composition
        """
        self.trial = [exp.Event(self,
                                dur=self.timelimit,  # in seconds
                                display=[self.s['bitmap1'], self.s['bitmap2']],
                                func=self.show_stim)
                     ]

    def create_exp_plan(self):
        """Put together trials
        """
        # Check if all images exist
        for im in self.imlist:
            if (not os.path.exists(os.path.join(self.paths['images'], im+self.asfx)) or
                not os.path.exists(os.path.join(self.paths['images'], im+self.bsfx))):
                raise Exception('Image files not found in image folder: ' + str(im))

        # Randomize the image order
        rnd.shuffle(self.imlist)

        # Create the orientations list: half upright, half inverted
        orilist = [0,180]*(len(self.imlist)/2)

        # Randomize the orientation order
        rnd.shuffle(orilist)

        exp_plan = []
        for trialno, (im, ori) in enumerate(zip(self.imlist, orilist)):
            exp_plan.append(OrderedDict([
                        ('im', im),
                        ('ori', ori),
                        ('onset', ''),  # empty ones will be filled up
                        ('dur', ''),    # during runtime
                        ('corr_resp', 1),
                        ('subj_resp', ''),
                        ('accuracy', ''),
                        ('rt', ''),
                        ]))
        self.exp_plan = exp_plan

    def before_trial(self):
        """Set up stimuli prior to a trial
        """
        im_fname = os.path.join(self.paths['images'], self.this_trial['im'])
        self.s['bitmap1'].setImage(im_fname + self.asfx)
        self.s['bitmap1'].setOri(self.this_trial['ori'])
        self.s['bitmap2'].setImage(im_fname + self.bsfx)
        self.s['bitmap2'].setOri(self.this_trial['ori'])
        self.bitmap = self.s['bitmap1']

        if self.thisTrialN > 0:  # no need for instructions for the first trial
            self.show_text(text=self.trial_instr, wait=0)

    def show_stim(self, *args, **kwargs):
        """Control stimuli during the trial
        """
        # Empty the keypresses list
        event.clearEvents()
        keys = []
        change_clock = core.Clock()

        # Start the trial
        # Stop trial if spacebar or escape has been pressed, or if 30s have passed

        while len(keys) == 0 and self.trial_clock.getTime() < self.this_event.dur:
             # Switch the image
            if self.bitmap == self.s['bitmap1']:
                self.bitmap = self.s['bitmap2']
            else:
                self.bitmap = self.s['bitmap1']

            self.bitmap.draw()

            # Draw bubbles of increasing radius at random positions
            for radius in range(self.n_bubbles):
                self.s['bubble'].setRadius(radius/100.)
                self.s['bubble'].setPos(((rnd.random()-.5) * self.stimsize[0],
                               (rnd.random()-.5) * self.stimsize[1] ))
                self.s['bubble'].draw()

            # Show the new screen we've drawn
            self.win.flip()

            # For the duration of 'changetime',
            # Listen for a spacebar or escape press

            change_clock.reset()
            while change_clock.getTime() <= self.changetime:
                keys = self.last_keypress(keyList=self.computer.valid_responses.keys(),
                                     timeStamped=self.trial_clock)

                if len(keys) > 0:
                    break
        return keys

if __name__ == "__main__":
    ChangeDet(rp={'no_output':True, 'debug':True}).run()
