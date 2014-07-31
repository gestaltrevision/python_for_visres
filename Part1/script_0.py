#==============================================
# Settings that we might want to tweak later on
#==============================================

# directory to save data in                          data
# directory where images can be found                image
# image names without the suffixes                   1,2,3,4,5,6
# suffix for the first image                         a.jpg
# suffix for the second image                        b.jpg
# screen size in pixels                              1200x800
# image freezing time in seconds                     30
# image changing time in seconds                     0.5
# number of bubbles overlayed on the image           40


#==========================================
# Store info about the experiment session
#==========================================

# Get subject name, gender, age, handedness through a dialog box
# If 'Cancel' is pressed, quit
# Get date and time
# Store this information as general session info
# Create a unique filename for the experiment data


#========================
# Prepare condition lists
#========================

# Check if all images exist
# Randomize the image order

# Create the orientations list: half upright, half inverted
# Randomize the orientation order


#===============================
# Creation of window and stimuli
#===============================

# Open a window
# Define trial start text
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

# Run through the trials. On each trial:
#   - Display trial start text
#   - Wait for a spacebar press to start the trial, or escape to quit
#   - Set the images, set the orientation
#   - Switch the image every 0.5s, and:
#        - Draw bubbles of increasing radius at random positions
#        - Listen for a spacebar or escape press
#   - Stop trial if spacebar or escape has been pressed, or if 30s have passed
#   - Analyze the keypress
#        - Escape press = quit the experiment
#        - Spacebar press = correct change detection; register response time
#        - No press = failed change detection; maximal response time
#   - Advance to the next trial


#======================
# End of the experiment
#======================

# Save all data to a file
# Quit the experiment
