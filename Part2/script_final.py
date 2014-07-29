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

# Get subject name, gender, age, handedness through a dialog box
exp_name = 'Change Detection'
exp_info = {
            'participant': '', 
            'gender': ('male', 'female'), 
            'age':'', 
            'left-handed':False 
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)

# If 'Cancel' is pressed, quit
if dlg.OK == False:
    core.quit()

# Get date and time
exp_info['date'] = data.getDateStr()
exp_info['exp_name'] = exp_name

# Create a unique filename for the experiment data
if not os.path.isdir(datapath):
    os.makedirs(datapath)
data_fname = exp_info['participant'] + '_' + exp_info['date']
data_fname = os.path.join(datapath, data_fname)


#========================
# Prepare condition lists
#========================

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
win = visual.Window(size=scrsize, color='white', units='pix', fullscr=False)

# Define trial start text
start_message = visual.TextStim(win,
                                text="Press spacebar to start the trial",
                                color='red', height=20)

# Define two bitmap stimuli (contents can still change)
bitmap1 = visual.SimpleImageStim(win, 
                                 image=os.path.join(impath, imlist[0]+asfx))
bitmap2 = visual.SimpleImageStim(win, 
                                 image=os.path.join(impath, imlist[0]+bsfx))
                                
# Define a bubble (position and size can still change)
bubble = visual.Circle(win, fillColor='black', lineColor='black')


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

# Initialize two clocks:
#   - for image change time
#   - for response time
change_clock = core.Clock()
rt_clock = core.Clock()

# Run through the trials
for trial in trials:
    
    # Display trial start text
    start_message.draw()
    win.flip()
    
    # Wait for a spacebar press to start the trial, or escape to quit
    keys = event.waitKeys(keyList=['space', 'escape'])

    # Set the images, set the orientation
    im_fname = os.path.join(impath, trial['im'])
    bitmap1.setImage(im_fname + asfx)
    bitmap1.setFlipHoriz(trial['ori'])
    bitmap2.setImage(im_fname + bsfx)
    bitmap2.setFlipHoriz(trial['ori'])
    bitmap = bitmap1

    # Set the clocks to 0
    change_clock.reset()
    rt_clock.reset()

    # Empty the keypresses list
    # Leave an 'escape' press in for immediate exit
    if 'space' in keys:
        keys = []  
    
    # Start the trial
    # Stop trial if spacebar or escape has been pressed, or if 30s have passed
    while not keys and rt_clock.getTime() < timelimit: 
        
        # Switch the image
        if bitmap == bitmap1:
            bitmap = bitmap2
        else:
            bitmap = bitmap1
        
        bitmap.draw()

        # Draw bubbles of increasing radius at random positions                
        for radius in range(n_bubbles):
            bubble.setRadius(radius)
            bubble.setPos(((rnd.random()-.5) * scrsize[0],
                           (rnd.random()-.5) * scrsize[1] ))
            bubble.draw()

        # Show the new screen we've drawn
        win.flip()
        
        # For the duration of 'changetime',
        # Listen for a spacebar or escape press
        change_clock.reset()
        while change_clock.getTime() <= changetime:
            keys = event.getKeys(keyList=['space','escape'])
            if keys:
                break

    # Analyze the keypress
    if keys:
        if 'escape' in keys:
            # Escape press = quit the experiment
            break
        else:
            # Spacebar press = correct change detection; register response time
            acc = 1
            rt = rt_clock.getTime()            
            
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
win.close()