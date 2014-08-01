#===============
# Import modules
#===============

import os                           # for file/folder operations
import numpy.random as rnd          # for random number generators
from psychopy import visual, event, core, gui, data


#==============================================
# Settings that we might want to tweak later on
#==============================================

datapath = 'data'                   # directory to save data in
impath = 'images'                   # directory where images can be found
imlist = ['1','2','3','4','5','6']  # image names without the suffixes
asfx = 'a.jpg'                      # suffix for the first image
bsfx = 'b.jpg'                      # suffix for the second image
scrsize = (1200,800)                # screen size in pixels
timelimit = 30                      # image freezing time in seconds
changetime = .5                     # image changing time in seconds
n_bubbles = 40                      # number of bubbles overlayed on the image


#========================================
# Store info about the experiment session
#========================================

exp_name = 'Change Detection'
exp_info = {}

# Get subject name, gender, age, handedness through a dialog box
# If 'Cancel' is pressed, quit
# Get date and time
# Store this information as general session info

# Create a unique filename for the experiment data
if not os.path.isdir(datapath):
    os.makedirs(datapath)
data_fname = exp_info['participant'] + '_' + exp_info['date']
data_fname = os.path.join(datapath, data_fname)


#=========================
# Prepare conditions lists
#=========================

# Check if all images exist
for im in imlist:
    if (not os.path.exists(os.path.join(impath, im+asfx)) or
        not os.path.exists(os.path.join(impath, im+bsfx))):
        raise Exception('Image files not found in image folder: ' + str(im)) 
        
# Randomize the image order
rnd.shuffle(imlist)

# Create the orientations list: half upright, half inverted
orilist = [0,1]*(len(imlist)/2)

# Randomize the orientation order
rnd.shuffle(orilist)


#===============================
# Creation of window and stimuli
#===============================

# Open a window

# Define trial start text
text = "Press spacebar to start the trial"

# Define the bitmap stimuli (contents can still change)
# Define a bubble (position and size can still change)


#==========================
# Define the trial sequence
#==========================

# Define a list of trials with their properties:
#   - Which image (without the suffix)
#   - Which orientation
stim_order = []
for im, ori in zip(imlist, orilist):
    stim_order.append({'im': im, 'ori': ori})

trials = data.TrialHandler(stim_order, nReps=1, extraInfo=exp_info,
                           method='sequential', originPath=datapath)


#=====================
# Start the experiment
#=====================

for trial in trials:
    
    # Display trial start text
    
    # Wait for a spacebar press to start the trial, or escape to quit
    
    # Set the images, set the orientation
    im_fname = os.path.join(impath, trial['im'])
    trial['ori']
    
    # Empty the keypresses list
    keys = []  
    
    # Start the trial
    # Stop trial if spacebar or escape has been pressed, or if 30s have passed
    while not response and time < timelimit:
        
        # Switch the image
        
        # Draw bubbles of increasing radius at random positions
        for radius in range(n_bubbles):  
            radius/2.
            pos = ((rnd.random()-.5) * scrsize[0],
                   (rnd.random()-.5) * scrsize[1] )

        # For the duration of 'changetime',
        # Listen for a spacebar or escape press
        while time < changetime:
            if response:
                break

    # Analyze the keypress
    if response:
        if escape_pressed:
            # Escape press = quit the experiment
            break
        elif spacebar_pressed:
            # Spacebar press = correct change detection; register response time
            acc = 1
            rt
    else:
        # No press = failed change detection; maximal response time
        acc = 0
        rt = timelimit
    
    # Add the current trial's data to the TrialHandler
    trials.addData('rt', rt)
    trials.addData('acc', acc)

    # Advance to the next trial


#======================
# End of the experiment
#======================

# Save all data to a file
trials.saveAsWideText(data_fname + '.csv', delim=',')

# Quit the experiment
