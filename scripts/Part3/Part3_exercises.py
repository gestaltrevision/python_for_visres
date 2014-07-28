# EXERCISE 1: Try to compute the BMI of each subject, as well as the average BMI across subjects
# BMI = weight/(length/100)**2
n = len(subj_length)
summed = 0.
for subj in range(n):
    subj_bmi.append(subj_weight[subj]/(subj_length[subj]/100)**2)
    summed = summed + subj_bmi[subj]
print subj_bmi
print summed/n


# EXERCISE 2: Try to complete the program now!
# Hint: np.mean() computes the mean of an ndarray
# Note that unlike MATLAB, Python does not need the '.' before elementwise operators
subj_bmi = subj_weight/(subj_length/100)**2 
mean_bmi = np.mean(subj_bmi)
print subj_bmi
print mean_bmi


# EXERCISE 3: Create a 2x3 array containing the column-wise and the row-wise means of the original matrix
# Do not use a for-loop, and also do not use the np.mean() function for now.
arr = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype='float')

res = np.array([(arr[:,0]+arr[:,1]+arr[:,2])/3,(arr[0,:]+arr[1,:]+arr[2,:])/3])
print res
print res.shape


# EXERCISE 4: Create your own meshgrid3d function
# Like np.meshgrid(), it should take two vectors and replicate them; one into columns, the other into rows
# Unlike np.meshgrid(), it should return them as a single 3D array rather than 2D arrays
# ...do not use the np.meshgrid() function

def meshgrid3d(xvec, yvec):
    xlayer = np.tile(xvec,(len(yvec),1))
    ylayer = np.tile(yvec,(len(xvec),1)).T
    return np.dstack((xlayer,ylayer))

xvec = np.arange(10)
yvec = np.arange(5)
xy = meshgrid3d(xvec, yvec)
print xy
print xy[:,:,0] # = first output of np.meshgrid()
print xy[:,:,1] # = second output of np.meshgrid()


# EXERCISE 5: Make a better version of Exercise 3 with what you've just learned
arr = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype='float')

# What we had:
print np.array([(arr[:,0]+arr[:,1]+arr[:,2])/3,(arr[0,:]+arr[1,:]+arr[2,:])/3])

# Now the new version:
print np.vstack((np.mean(arr,1), np.mean(arr,0)))


# EXERCISE 6: Create a Gabor patch of 100 by 100 pixels
import numpy as np
import matplotlib.pyplot as plt

# Create x and y coordinates
vals = np.linspace(-np.pi, np.pi, 100)
x,y = np.meshgrid(vals, vals)

# Make grating, gaussian, gabor
grating = np.sin(x*10)
gaussian = np.exp(-((x**2)+(y**2))/2)
gabor = grating*gaussian

# Visualize your result
# (we will discuss how this works later)
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(grating, cmap='gray')
plt.subplot(132)
plt.imshow(gaussian, cmap='gray')
plt.subplot(133)
plt.imshow(gabor, cmap='gray')
plt.show()


# EXERCISE 7: Vectorize the above program

# You get these lines for free...
import numpy as np
throws = np.random.randint(1,7,(5000,2000))
one = (throws==1)
two = (throws==2)
three = (throws==3)

# Find out where all the 111 and 123 sequences occur
find111 = one[:,:-2] & one[:,1:-1] & one[:,2:]
find123 = one[:,:-2] & two[:,1:-1] & three[:,2:]

# Then at what index they /first/ occur in each sequence
first111 = np.argmax(find111, axis=1)
first123 = np.argmax(find123, axis=1)

# Compute the average first occurence location for both situations
avg111 = np.mean(first111)
avg123 = np.mean(first123)

# Print the result
print avg111, avg123


# EXERCISE 8: Visualize the difference between the PIL conversion to grayscale, and a simple average of RGB
# Display pixels where the average is LESS luminant in red, and where it is MORE luminant in shades green
# The luminance of these colors should correspond to the size of the difference
# Extra: Maximize the overall contrast in your image
# Extra2: Save as three PNG files, of different sizes (large, medium, small)

import numpy as np
from PIL import Image
im = Image.open('python.jpg')

# Do both grayscale conversions
im_avg = np.array(im)
im_avg = np.mean(im_avg,2)
im_pil = im.convert('L')
im_pil = np.array(im_pil)

# Compute the difference per pixel
imd = im_avg-im_pil

# Assign different colors according to the direction of difference
outp = np.zeros((im_avg.shape)+(3,))
outp[:,:,0][imd<0] = -imd[imd<0]
outp[:,:,1][imd>0] = imd[imd>0]

# Maximize contrast
outp = outp * (255./np.max(outp))

# Conversion back to a PIL image
outp = outp.astype('uint8')
outp_pil = Image.fromarray(outp, mode='RGB')

# Save with three different sizes
sz = np.array(outp_pil.size)
sz_name = ['large','medium','small']
for n,fct in enumerate([1,2,4]):
    outp_rsz = outp_pil.resize(sz/fct)
    outp_rsz.save('python_'+sz_name[n]+'.png')


# EXERCISE 9: Plot y=sin(x) and y=sin(x^2) in two separate subplots, one above the other
# Let x range from 0 to 2*pi

import numpy as np
import matplotlib.pyplot as plt

# X-axos values
x = np.linspace(0,2*np.pi,1000)

# Figure and Axes creation
fig = plt.figure(figsize=(10,5))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)

# Make the plots
ax0.plot(x,np.sin(x),'r-', linewidth=2)
ax1.plot(x,np.sin(x**2),'b-', linewidth=2)

# Finetune the plots
ax0.set_xlim([0,2*np.pi])
ax0.set_xticks([])
ax1.set_xlim([0,2*np.pi])

# Show the figure
fig.show()


# EXERCISE 10: Add regression lines
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

# Open image, convert to an array
im = Image.open('python.jpg')
im = im.resize((400,300))
arr = np.array(im, dtype='float')

# Split the RGB layers and flatten them
R,G,B = np.dsplit(arr,3)
R = R.flatten()
G = G.flatten()
B = B.flatten()

# Do the plotting
plt.figure(figsize=(5,5))
plt.plot(R, B, marker='x', linestyle='None', color=(0,0,0.6))
plt.plot(R, G, marker='.', linestyle='None', color=(0,0.35,0))

# Tweak the plot
plt.axis([0,255,0,255])
plt.xlabel('Red value')
plt.ylabel('Green/Blue value')

# Do the linear regressions
regRB = np.polyfit(R,B,1)
regRG = np.polyfit(R,G,1)

# Create the line objects
xaxv = np.arange(255.)
lRB = lines.Line2D(xaxv,regRB[1]+xaxv*regRB[0], color='k')
lRG = lines.Line2D(xaxv,regRG[1]+xaxv*regRG[0], color='k')

# Fetch the current Axes, and attach the lines to it
ax = plt.gca()
ax.add_artist(lRB)
ax.add_artist(lRG)

# Show the result
plt.show()
