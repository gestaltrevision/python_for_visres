#==============================================
# Settings that we might want to tweak later on
#==============================================

datapath = 'data'                   # directory to save data in
impath = 'images'                   # directory where images can be found
imlist = ['1','2','3','4','5','6']  # image names without the suffixes
asfx = 'a.jpg'                      # suffix for the first image
bsfx = 'b.jpg'                      # suffix for the second image
scrsize = (600,400)                 # screen size in pixels
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
data_fname = exp_info['participant'] + '_' + exp_info['date']


#========================
# Prepare condition lists
#========================

# Check if all images exist
# Randomize the image order

# Create the orientations list: half upright, half inverted
orilist = [0,1]*(len(imlist)/2)

# Randomize the orientation order


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


#=====================
# Start the experiment
#=====================

for trial in trials:
    
    # Display trial start text
    
    # Wait for a spacebar press to start the trial, or escape to quit
    
    # Set the image filename, set the orientation    
    
    # Start the trial
    # Stop trial if spacebar or escape has been pressed, or if 30s have passed
    while not response and time < timelimit:
        
        # Switch the image
        
        # Draw bubbles of increasing radius at random positions

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
    else:
        # No press = failed change detection; maximal response time

    # Advance to the next trial


#======================
# End of the experiment
#======================

# Save all data to a file
# Quit the experiment
