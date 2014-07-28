import sys


class Logger(object):
    """Log all output
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("workshop_req_check.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

sys.stdout = Logger()
sys.stderr = sys.stdout


from distutils.version import LooseVersion as V

def check_reqs(cmod, reqs):
    """Check requirements for each session
    """
    reqs_fix = []
    for lvl,req in enumerate(reqs):
        for k,v in req.iteritems():
            if k in cmod:
                if v > cmod[k]:
                    reqs_fix.append(k)
        if reqs_fix:
            if lvl == 0:
                print "    NOT PREPARED, check error messages for:"
            elif lvl == 1:
                print "    MINIMALLY PREPARED, check error messages for:"
            elif lvl == 2:
                print "    MOSTLY PREPARED, check error messages for:"
            for k in reqs_fix:
                print "       -", k
            return
    print "    FULLY PREPARED"


def mod_check(mod, modname, req_ver=None, max_ver=None, fun=None):
    print '   ', modname
    try:
        __import__(mod)
    except ImportError:
        cmod[mod] = 0
        print '        Not found, please (re-)install'
    else:
        if req_ver or max_ver:
            m = sys.modules[mod]
            try:
                this_ver = m.__version__
            except AttributeError:
                this_ver = m.version

        if req_ver and V(this_ver) < V(req_ver):
            cmod[mod] = 1
            print '        Version %s+ recommended, now: %s' % (req_ver, this_ver)
        else:
            cmod[mod] = 2

        if max_ver and V(this_ver) > V(max_ver):
            cmod[mod] = 1
            print '        Version under %s recommended, now: %s' % (max_ver, this_ver)
        elif cmod[mod] != 1:
            cmod[mod] = 2

        if fun is not None:
            try:
                fun()
            except:
                print sys.exc_info()[1]
                print '        Failed test'
                cmod[mod] = 0
            else:
                print '        OK'
        else:
            print '        OK'

def test_psychopy():
    print
    print '*'*61
    print '*' + ' '*59 + '*'
    print '* A new window will open. Please follow instructions on it. *'
    print '*' + ' '*59 + '*'
    print '*'*61
    print
    from psychopy import visual, event
    win = visual.Window()
    text = visual.TextStim(win, "Press any key to continue...")
    text.draw()
    win.flip()
    event.waitKeys()
    win.close()

def test_ipynb():
    import subprocess, time
    print
    print '*'*61
    print '*' + ' '*59 + '*'
    print '* An IPython notebook should open in your browser.          *'
    print '* Please wait for this test to finish. Do not hit Control-C *'
    print '*' + ' '*59 + '*'
    print '*'*61
    print
    proc = subprocess.Popen(['ipython', 'notebook'], shell=False)
    time.sleep(10)
    proc.terminate()

def test_PIL():
    import PIL
    import PIL.Image

    this_ver = PIL.Image.VERSION
    req_ver = '1.1.7'

    if V(this_ver) < V(req_ver):
        cmod['PIL'] = 1
        print '        Version %s+ recommended, now: %s' % (req_ver, this_ver)
    else:
        cmod['PIL'] = 2

    try:
        this_ver = PIL.PILLOW_VERSION
        req_ver = '2.2'
        if V(this_ver) < V(req_ver):
            print '        Version %s+ recommended, now: %s' % (req_ver, this_ver)
            cmod['PIL'] = 1
        else:
            cmod['PIL'] = 2
    except AttributeError:
        print '        You are using plain PIL, Pillow is recommended instead'
        cmod['PIL'] = 1

# Print system info
print sys.platform
print sys.path

# Check the individual modules
# 0=broken, 1=suboptimal, 2=ok
cmod = {}

print
print '='*79
print 'MODULE CHECK'
print

mod_check('sys', 'Python: base installation', '2.6','2.8')
mod_check('spyderlib', 'Spyder: IDE', '2.2.5')
mod_check('numpy', 'NumPy: numerical computing', '1.6')
mod_check('scipy', 'SciPy: scientific functions', '0.10')
mod_check('matplotlib', 'Matplotlib: plot graphs', '1.0')
mod_check('psychopy_ext', 'PsychoPy_ext: streamline research', '0.5.2')
mod_check('seaborn', 'Seaborn: statistical data visualization', '0.2')
mod_check('docutils', 'Docutils: documentation utilities')
mod_check('svgwrite', 'Svgwrite: create svg images')
mod_check('pandas', 'Pandas: data analysis toolkit', '0.12')
mod_check('nibabel', 'NiBabel: access neuroimaging files')
mod_check('mvpa2', 'PyMVPA: fMRI MVPA package', '2.3')
mod_check('PIL', 'Pillow: handle images', None, None, test_PIL)
mod_check('psychopy', 'PsychoPy: build experiments', '1.79.01', None, test_psychopy)
mod_check('IPython', 'IPython: interactive notebooks', '0.13', None, test_ipynb)

print
print '='*79
print 'HOW WELL ARE YOU PREPARED?'
print

# Now check if requirements are met for each session
# Format of reqs: [minimally, mostly, fully] prepared, else not prepared

print "Session: Introduction to Python"
reqs = [{'sys':1},

        {'sys':1,
         'numpy':1,
         'psychopy':1},

        {'sys':2,
         'spyderlib':2,
         'numpy':2,
         'psychopy':2,
         'IPython':2}]
check_reqs(cmod, reqs)

# ***
print
print "Session: Introduction to PsychoPy"
reqs_psych = [{'sys':1,
             'psychopy':1},

            {'sys':2,
             'numpy':1,
             'scipy':1,
             'psychopy':1},

            {'sys':2,
             'spyderlib':2,
             'numpy':2,
             'scipy':2,
             'psychopy':2,
             'IPython':2}]
check_reqs(cmod, reqs_psych)

# ***
print
print "Session: Transitioning from MATLAB to Python"
reqs = [{'sys':1,
         'numpy':1,
         'scipy':1,
         'PIL':1,
         'matplotlib':1},

        {'sys':1,
         'spyderlib':1,
         'numpy':2,
         'scipy':1,
         'PIL':1,
         'matplotlib':1},

        {'sys':2,
         'spyderlib':2,
         'numpy':2,
         'scipy':2,
         'PIL':2,
         'IPython':2,
         'matplotlib':2}]
check_reqs(cmod, reqs)

# ***
print
print "Session: More practice with PsychoPy"
check_reqs(cmod, reqs_psych)

# ***
print
print "Session: Streamline research with psychopy_ext"
reqs = [{'sys':2,
         'numpy':2,
         'scipy':2,
         'matplotlib':2,
         'psychopy':2,
         'psychopy_ext':2,
         'pandas':2},

        {'sys':2,
         'numpy':2,
         'scipy':2,
         'matplotlib':2,
         'psychopy':2,
         'psychopy_ext':2,
         'seaborn':2,
         'docutils':1,
         'pandas':2},

        {'sys':2,
         'spyderlib':2,
         'numpy':2,
         'scipy':2,
         'matplotlib':2,
         'psychopy':2,
         'psychopy_ext':2,
         'seaborn':2,
         'docutils':2,
         'svgwrite':2,
         'pandas':2,
         'IPython':2}]
check_reqs(cmod, reqs)

# ***
print
print "Session: Natural image statistics"
reqs = [{'sys':1,
         'numpy':1,
         'scipy':1,
         'PIL':1,
         'matplotlib':1},

        {'sys':1,
         'numpy':2,
         'scipy':2,
         'PIL':1,
         'matplotlib':1},

        {'sys':2,
         'numpy':2,
         'scipy':2,
         'PIL':2,
         'matplotlib':2,
         'IPython':2}]
check_reqs(cmod, reqs)

# ***
print
print "Session: Multi-voxel pattern analysis"
reqs = [{'sys':2,
         'numpy':2,
         'psychopy_ext':2,
         'nibabel':2,
         'mvpa2':2},

        {'sys':2,
         'numpy':2,
         'psychopy_ext':2,
         'nibabel':2,
         'mvpa2':2},

        {'sys':2,
         'spyderlib':2,
         'numpy':2,
         'psychopy_ext':2,
         'nibabel':2,
         'mvpa2':2,
         'IPython':2}]
check_reqs(cmod, reqs)

# ***
print
print '='*79
print 'WHAT TO DO NOW?'
print
print "1. Check in the list above how well you're prepared for the sessions"
print "   you signed up."
print "2. Ideally, you should be fully prepared. Mostly prepared might"
print "   still suffice but not everything may work. Minimally prepared means"
print "   you will not be able to execute significant parts of the code."
print "3. If you're underprepared, download and install missing packages,"
print "   and rerun this script. You may find information at"
print "   http://gestaltrevision.be/wiki/python/check_install useful."
print "4. A file `workshop_req_check.txt` was generated in the same folder"
print "   where this script is. When ready, please **email** it to "
print "   <Maarten.Demeyer@ppw.kuleuven.be> so that we can verify that "
print "   you're ready for the workshop."
print
print '='*79
print

sys.stdout.log.close()
