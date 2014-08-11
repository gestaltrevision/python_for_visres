#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2014 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A wrapper of PyMVPA2 for simple fMRI analyses using SPM preprocessing.

Currently only signal, SVM, and correlational analyzes are stable. Other
features are not as extensively tested.

.. warning:: This library has not been thoroughly tested yet!
"""

import os, sys, glob, shutil, warnings
from datetime import datetime
import cPickle as pickle

import numpy as np
import scipy.stats
import pandas

try:
    import mvpa2.suite
except ImportError:
    raise ImportError('You must have pymvpa2 installed to run this.')
try:
    import nibabel as nb
except ImportError:
    raise ImportError('You must have nibabel installed to run this.')

# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict

# stuff from psychopy_ext
from psychopy_ext import exp, plot, stats
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt


class Analysis(object):

    def __init__(self,
                 paths,
                 tr,
                 info=None,
                 rp=None,
                 fmri_prefix='swa*',
                 fix=0,
                 rois=None,
                 offset=None,
                 dur=None,
                 condlabel='cond',
                 durlabel='dur'
                 ):
        """
        For conducting functional magnetic resonance imaging analyses.

        Assumptions:

        1. For beta- and t-values, analyses were done is SPM.
        2. Functional runs named as `func_<runno>`, anatomicals named as
           `struct<optional extra stuff>`, behavioral data files (or files with
           condition assignments) as `<something>_<runno>_<runtype>.csv`
        3. Beta-values model every condition, including fixation. But
           t-values are computed as each conditions versus a fixation.

        :Args:
            - paths (dict of str:str pairs)
                A dictionary of paths where data is stored. Expected to have at
                least the following keys:

                - 'analysis' (for storing analysis outputs),
                - 'data_behav' (behavioral data with condition labels),
                - 'data_fmri',
                - 'rec' (for ROIs from surface reconstruction in Caret or so),
                - 'data_rois' (for storing the extracted signals in these ROIs)

            - tr (int or float)
                Time of repetition during the fMRI scan. Usually 1, 2, or 3 seconds.
                This information is not
                reliably coded in NIfTI files, so you need to define it yourself.

        :Kwargs:
            - info (dict, default: None)
                All parameters related to participant information
            - rp (dict, default: None)
                All runtime parameters that you want to be able to access from GUI
                or CLI. Expected to have at least:

                - no_output
                - verbose
                - force

            - fmri_prefix (str, default: 'swa*')
                Prefix of SPM output functional scans that you want to analyze.
            - fix (int or str, default: 0)
                Label to identify fixation condition.
            - rois (list of str)
                A list of ROIs to analyze. See :func:`make_roi_pattern` for
                accepted formats.
            - offset (int or dict)
                E.g., {'V1': 4, 'V2': 4, 'V3': 4, 'LO': 3, 'pFs': 3}
            - dur (int or dict)
                Same format as 'offset'.
            - condlabel (str, default: 'cond')
                Name of the column in your data file where condition number of
                each trial is kept.
            - durlabel (str, default: 'dur')
                Name of the column in your data file where duration  of each trial
                is kept.
        """

        # minimal parameters that Analysis expects in info and rp
        self.info = OrderedDict([
            ('subjid', 'subj'),
            ('runtype', ('main', 'loc')),
            ])
        self.rp = OrderedDict([
            ('method', ('timecourse', 'corr', 'svm')),
            ('values', ('raw', 'beta', 't')),
            ('no_output', False),
            ('debug', False),
            ('verbose', True),
            ('plot', True),
            ('saveplot', False),
            ('visualize', False),
            ('force', None),
            #('retrain', False),
            ('dry', False),
            #('reuserois', True),
            ])
        if info is not None:
            self.info.update(info)
        if rp is not None:
            self.rp.update(rp)

        self.paths = paths
        self.tr = tr
        self.fmri_prefix = fmri_prefix
        self.fix = fix
        self.rois = make_roi_pattern(rois)
        self.offset = offset
        if self.rp['method'] == 'timecourse':
            self.rp['values'] = 'raw'
            if offset is None:
                self.offset = 0
        else:
            if offset is None:
                self.offset = 2
        self.dur = dur
        self.condlabel = condlabel
        self.durlabel = durlabel

    def run(self):
        """
        A wrapper for running an analysis specified in `self.rp`.

        Steps:
            - Try to load a saved analysis, unless a `force` flag is given
            - Otherwise, either generate synthetic data (values = `sim`) or
              extract it from the real data using :func:`run_method`.
            - Save a `pandas.DataFrame` in the analysis folder with the
              filename like `df_<method>_<values>.pkl`

        :Returns:
            A DataFrame with the output of a particular analysis in the
            `subj_resp` column, and a file name where that data is stored.

        """
        df, df_fname, loaded = self.get_fmri_df()
        if self.rp['plot']:
            self.plot(df)
        return df, df_fname, loaded

    def get_fmri_df(self, avg_iters=True):
        df_fname = (self.paths['analysis']+'df_%s_%s.pkl' %
                    (self.rp['method'], self.rp['values']))
        try:
            if self.rp['force'] is not None:
                raise  # redo the analysis
            else:
                df = pickle.load(open(df_fname,'rb'))
                if self.rp['verbose']:
                    mtime = os.path.getmtime(df_fname)
                    mtime = datetime.fromtimestamp(mtime).ctime()
                    print ('loaded stored dataset of %s %s results '
                           '[saved on %s]' % (self.rp['values'],
                           self.rp['method'], mtime))
                    print 'subjids: %s' % ', '.join(df.subjid.unique())
        except:
            res_fname = self.paths['analysis']+'%s_%s_%s.pkl'
            # generate some fake data to check a particular hypothesis
            if self.rp['values'] == 'sim':
                simds = self.genFakeData()
            else:
                simds = None
            header, results = self.run_method(self.info['subjid'],
                self.info['runtype'], self.rois, offset=self.offset,
                dur=self.dur, filename=res_fname,
                method=self.rp['method'], values=self.rp['values'],
                simds=simds)
            df = pandas.DataFrame(results, columns=header)
            #if not self.rp['no_output']:
                #try:
                    ##import pdb; pdb.set_trace()
                    #os.makedirs(os.path.dirname(df_fname))
                #except:
                    #pass
                #pickle.dump(df, open(df_fname,'wb'))
                #if self.rp['verbose']:
                    #print ("saved dataset of %s %s results to %s" %
                          #(self.rp['values'], self.rp['method'],
                              #df_fname))
            loaded = False
        else:
            loaded = True

        return df, df_fname, loaded

    def get_behav_df(self, pattern='%s'):
        """
        Extracts data from files for data analysis.

        :Kwargs:
            pattern (str, default: '%s')
                A string with formatter information. Usually it contains a path
                to where data is and a formatter such as '%s' to indicate where
                participant ID should be incorporated.

        :Returns:
            A `pandas.DataFrame` of data for the requested participants.
        """
        return exp.get_behav_df(self.info['subjid'], pattern=pattern)


    def plot(self, df, cols='cond', **kwargs):
        if self.rp['method'] == 'timecourse':
            plt = plot_timecourse(df, cols=cols)
            plt.tight_layout()
        else:
            if self.rp['method'] == 'corr':
                title = '(1-correlation) / 2'
            else:
                title = 'SVM decoding'

            agg = self.get_agg(df, **kwargs)
            plt = plot.Plot()
            axes = plt.plot(agg, kind='bar', title=title, ylabel='dissimilarity')
            # mark chance level
            if self.rp['method'] == 'svm':
                try:
                    iter(axes)
                except:
                    axes = [axes]
                for ax in axes:
                    ax.axhline(y=.5, color='.2', ls='--', lw=2, marker='None')
            plt.tight_layout()

            if self.rp['method'] in ['corr','svm']:
                mtx_agg = self.get_agg(df, kind='matrix')
                mtx = plot.Plot(kind='matrix')
                axes = mtx.plot(mtx_agg, kind='matrix', title=title)
                try:
                    iter(axes)
                except:
                    axes = [axes]
                for ax in axes:
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                mtx.tight_layout()

        plt.show()  # will show mtx as well :/
        if self.rp['saveplot'] and not self.rp['no_output']:
            plt.savefig(self.paths['analysis']+'%s_%s.png' %
                (self.rp['method'], self.rp['values']))
            if self.rp['method'] in ['corr','svm']:
                mtx.savefig(self.paths['analysis']+'%s_%s_matrix.png' %
                    (self.rp['method'], self.rp['values']))

    #def get_data(self, df):
        #raise NotImplementedError

    def get_agg(self, df, kind=None, **kwargs):
        #raise NotImplementedError('You should define get_agg method yourself.')
        if kind == 'matrix':
            agg = stats.aggregate(df, values='subj_resp',
                                        cols=['stim1.cond'],
                                        rows=['stim2.cond'],
                                        yerr='subjid')
        else:
            df['group'] = 'within'
            df['group'][df['stim1.cond'] != df['stim2.cond']] = 'across'
            agg = stats.aggregate(df, values='subj_resp',
                                        cols='group',
                                        yerr='subjid')
        return agg

    def run_method(self, subjids, runtype, rois, method='svm', values='raw',
                offset=None, dur=None, filename = 'RENAME.pkl', simds=None):
        """
        A wrapper for running a specified analysis.

        Process:
            1. Attempt to load stored results from the analysis that was done
               before. (stored in teh analysis folder in a file
               `<method>_<values>_<subjid>.pkl`
            2. If that fails, it's probably because the analysis has
               not been performed yet or, in rare cases, because the data
               file is corrupt or missing. So a new analysis is initiated.

                1. First, Regions of Interest (ROIs) are loaded from ``PATHS['data_rois']``
                2. If that is not possible, then ROIs are extracted from
                   anatomical data using functional localizer data from SPM.
                3. Extracted ROIs are stored in ``PATHS['data_rois']``.
                4. Finally, the specified analysis is performed.

        :Args:
            - subjids (str of list of str)
                Which participants should be analyzed
            - runtype (str)
                Which run type should be taken. Usually you have a few runs,
                such as main experimental runs and localizer runs. They should
                have be labeled data file

        :Kwargs:
            - method: {'timecourse', 'univariate', 'signal', 'corr',  'svm'} (default: 'svm'}
                Method to analyze data.
            - values: {'raw', 'beta', 't'}
                fMRI signal values to use. If 'raw', you have to pass offset
                and duration. If you intend to try a few different parameters
                for 'raw', e.g. a duration of 1 TR and 3 TRs, you may indicate
                that in the value parameter like ``values='raw_3'`` which will
                be useful in order not to confuse output files (they get
                prefixes based on the value name).
                e.g.::

                    offset = {'V1': 4, 'V2': 4, 'V3': 4, 'LO': 3, 'pFs': 3}
                    dur = 1
        """

        if type(subjids) not in [list, tuple]:
            subjids = [subjids]
        results = []
        #loadable = []
        ## quick way to see if we need to import mvpa2.suite
        #for sNo, subjid in enumerate(subjids):
            #try:
                #filename_full = filename % (method, values, subjid)
            #except:
                #pass
            #loadable.append(os.path.isfile(filename_full))
        #import pdb; pdb.set_trace()
        #if not np.all(loadable):

        for subjid in subjids:
            print subjid,
            #sys.stdout.write("\n%s" % subjid)
            sys.stdout.flush()
            try:
                out_fname = filename % (method, values, subjid)
            except:
                raise
            loaded = False
            if method in ['corr', 'svm']:
                try:
                    if self.rp['force'] in ['retrain']:
                        raise  # redo the analysis
                    else:
                        header, result = pickle.load(open(out_fname,'rb'))
                        results.extend(result)
                        # result = pickle.load(open(filename,'rb'))
                        # header = [i[0] for i in result[0]]
                        # for res in result:
                        #     results.append([r[1] for r in res])
                        print ': loaded stored %s results' % values
                        loaded = True
                except:
                    print
                    print "Could not load or find the results file %s" % out_fname
                    print "Will proceed to do %s analysis from scratch" % method
            if not loaded:
                temp_res = []
                for r, ROI_list in enumerate(rois):
                    print ROI_list[1],
                    sys.stdout.flush()
                    if simds is not None:
                        values = 'sim'
                    else:
                        if type(offset) == dict:  # different offsets for ROIs
                            off = offset[ROI_list[1]]
                        else:
                            off = offset
                        ds = self.extract_samples(subjid, runtype, ROI_list,
                                                  values=values, offset=off)

                    if values.startswith('raw'):
                        ds = self.detrend(ds)
                        #ds = np.nan_to_num(ds, value=0)
                        evds = self.ds2evds(ds, dur=dur)
                    elif values in ['t', 'beta', 'sim']:
                        # SPM sets certain voxels to NaNs
                        # we just gonna convert them to 0
                        ds.samples = np.nan_to_num(ds.samples)
                        evds = ds

                    if method == 'timecourse':
                        header, result = self.timecourse(evds, offset=off)
                    elif method in ['signal', 'univariate']:
                        header, result = self.signal(evds, values)
                    elif method == 'corr':
                        evds = evds[evds.sa.targets != self.fix]
                        header, result = self.correlation(evds, nIter=100)
                    elif method == 'svm':
                        evds = evds[evds.sa.targets != self.fix]
                        header, result = self.svm(evds, nIter=100)
                    elif method is None:
                        sys.exit()
                    else:
                        try:
                            func = getattr(self, method)
                        except:
                            raise NotImplementedError('Analysis for %s '
                                    'values is not implemented' % values)
                        else:
                            header, result = func(evds, values)

                    header = ['subjid', 'roi'] + header
                    for line in result:
                        line = [subjid, ROI_list[1]] + line
                        temp_res.append(line)
                print
                results.extend(temp_res)

                if not self.rp['no_output'] and method in ['corr', 'svm']:
                    # mvpa2.suite.h5save(rp.o, results)
                    try:
                        os.makedirs(self.paths['analysis'])
                    except:
                        pass

                    pickle.dump([header,temp_res], open(out_fname,'wb'))
        return header, results

    #def time_course(self):
        #ds = self.extract_samples(subjid, runtype, ROI_list,
                                                  #values=values)


        #return thisloop

    #@loop
    # def time_course(self, subjid, runtype, ROI_list):
    #     ds = self.extract_samples(subjid, runtype, ROI_list)
    #     ds = self.detrend(ds)
    #     evds = self.ds2evds(ds, offset=0, dur=8)
    #     # mvpamod.plotChunks(ds,evds,chunks=[0])
    #     return self.get_psc(evds)

    # def univariate(self, subjid, runtype, ROI_list):
    #     ds = self.extract_samples(subjid, runtype, ROI_list)
    #     ds = self.detrend(ds)
    #     evds = self.ds2evds(ds, offset=3, dur=3)
    #     # mvpamod.plotChunks(ds,evds,chunks=[0])
    #     return self.psc_diff(evds)

    # #@loop
    # def mvpa(self, subjid, runtype, ROI_list, offset, dur):
    #     """Performs an SVM classification.

    #     **Parameters**

    #         clf: 'SMLR', 'LinearCSVMC', 'LinearNuSVMC', 'RbfNuSVMC', or 'RbfCSVMC', or a list of them
    #             A name of the classifier to be used

    #     """
    #     ds = self.extract_samples(subjid, runtype, ROI_list)
    #     ds = self.detrend(ds)
    #     evds = self.ds2evds(ds, offset=offset, dur=dur)
    #     evds = evds[evds.sa.targets != 0]
    #     return self.svm(evds)

    def get_mri_data(self, filename):
        """
        Get MRI data with the affine transformation (world coordinates) applied.

        :Args:
            filename (str)
                A filename of data to load
        """
        nim = nb.load(filename)
        data = get_data_world(nim)
        # reorient data based on the affine information in the header
        #ori = nb.io_orientation(nim.get_affine())
        #data = nb.apply_orientation(data, ori)
        data = np.squeeze(data)  # remove singular dimensions (useful for ROIs)
        return data

    def extract_samples(self,
        subjid,
        # runno,
        runtype,
        ROIs,
        values='raw',
        offset=0
        ):
        """
        Produces a detrended dataset with info for classifiers.

        :Args:
            - subjid (str)
                participant ID
            - runtype (str)
                run type (useful if, for example, you also have
                localizer runs which you probably want to analyze separately from
                the experimental runs)
            - ROIs (list)
                A pattern of ROI file patterns to be combined into one ROI

        :Kwargs:
            values (str, default: 'raw')
                What kind of values should be used. Usually you
                have 'raw', 'beta', and 't'.

        :Returns:
            ds (Dataset)

        """
        if values.startswith('raw'):
            add = ''
        else:
            add = '_' + values
        suffix = ROIs[1] + add + '.gz.hdf5'
        roiname = self.paths['data_rois'] %subjid + suffix
        #if self.rp['reuserois'] and os.path.isfile(roiname):
            #ds = mvpa2.suite.h5load(roiname)
            #print '(loaded)',
            #return ds

        # else
        # make a mask by combining all ROIs
        allROIs = []
        for ROI in ROIs[2]:
            theseROIs = glob.glob((self.paths['rois'] + ROI + '.nii*') %subjid)
            allROIs.extend(theseROIs)
        if len(allROIs) == 0:
            raise Exception('No matching ROIs were found in %s' %
                            (self.paths['rois'] % subjid))

        # add together all ROIs -- and they should not overlap too much
        thismask = sum([self.get_mri_data(roi) for roi in allROIs])
        #thismask = thismask[::-1]

        if values.startswith('raw'):
            # find all functional runs of a given runtype
            allimg = glob.glob((self.paths['data_fmri'] + self.fmri_prefix + \
                               runtype + '.nii*') % subjid)
            allimg.sort()
            data_path = self.paths['data_behav']+'data_%02d_%s.csv'
            labels = self.extract_labels(allimg, data_path, subjid, runtype)
            ds = self.fmri_dataset(allimg, labels, thismask, offset=offset)
        elif values == 'beta':
            data_path = self.paths['data_behav'] + 'data_*_%s.csv'
            behav_data = self.read_csvs(data_path %(subjid, runtype))
            labels = np.unique(behav_data[self.condlabel]).tolist()
            nruns = len(np.unique(behav_data['runno']))
            analysis_path = self.paths['spm_analysis'] % subjid + runtype + '/'
            betaval = np.array(sorted(glob.glob(analysis_path + 'beta_*.img')))
            if len(betaval) != (len(labels) + 6) * nruns + nruns:
                raise Exception('Number of beta value files is incorrect '
                    'for participant %s' % subjid)
            select = [True]*len(labels) + [False]*6
            select = np.array(select*nruns + [False]*nruns)
            allimg = betaval[select]

            ds = []
            nLabels = len(labels)
            for runno in range(nruns):
                ds.append( mvpa2.suite.fmri_dataset(
                    samples = allimg[runno*nLabels:(runno+1)*nLabels].tolist(),
                    targets = labels,
                    chunks = runno,
                    mask = thismask
                    ))
            ds = mvpa2.suite.vstack(ds)
        elif values == 't':
            data_path = self.paths['data_behav'] + 'data_*_%s.csv'
            behav_data = self.read_csvs(data_path %(subjid, runtype))
            labels = np.unique(behav_data[self.condlabel]).tolist()
            # t-values did not model all > fixation, so we skip it now
            labels = labels[1:]
            nruns = len(np.unique(behav_data['runno']))
            analysis_path = self.paths['spm_analysis'] % subjid + runtype + '/'
            tval = np.array(sorted(glob.glob(analysis_path + 'spmT_*.img')))
            if len(tval) != (nruns + 1) * len(labels):
                raise Exception('Number of t value files is incorrect '
                    'for participant %s' % subjid)
            allimg = tval[np.arange(len(tval)) % (nruns+1) != nruns]
            ds = mvpa2.suite.fmri_dataset(
                samples = allimg.tolist(),
                targets = np.repeat(labels, nruns).tolist(),
                chunks = np.tile(np.arange(nruns), len(labels)).tolist(),
                mask = thismask
                )
        else:
            raise Exception('values %s are not recognized' % values)
        if not self.rp['no_output']:  # save the extracted data
            try:
                os.makedirs(self.paths['data_rois'] %subjid)
            except:
                pass
            mvpa2.suite.h5save(roiname, ds, compression=9)

        return ds

    def extract_labels(self, img_fnames, data_path, subjid, runtype):
        """
        Extracts data labels (targets) from behavioral data files.

        .. note:: Assumes that each block/condition is a multiple of TR.
        """
        labels = []
        for img_fname in img_fnames:
            runno = int(img_fname.split('_')[-2])
            behav_data = pandas.read_csv(data_path %(subjid, runno, runtype))
            # indicate which condition was present for each acquisition
            # FIX: !!!ASSUMES!!! that each block/condition is a multiple of TR
            run_labels = []
            for lineNo, line in behav_data.iterrows():
                # how many TRs per block or condition
                repeat = int(line[self.durlabel] / self.tr)  # FIX
                run_labels.extend( [line[self.condlabel]] * repeat )  #FIX
            labels.append(run_labels)

        return labels

    def fmri_dataset(self, samples, labels, thismask=None, offset=0):
        """
        Create a dataset from an fMRI timeseries image.

        Overrides `mvpa2.datasets.mri.fmri_dataset` which has a buggy multiple
        images reading.
        """
        ## Load in data for all runs and all ROIs
        chunkcount = 0
        first = True
        mvpa2.datasets.mri._img2data = _img2data
        for thisimg, thislabel in zip(samples,labels):
            # load the appropriate func file with a mask
            tempnim = mvpa2.suite.fmri_dataset(
                    samples = thisimg,
                    targets = np.roll(thislabel, offset),
                    chunks = chunkcount,  #FIXME: change to runno?
                    mask = thismask
                    )
            # combine all functional runs into one massive NIfTI Dataset
            if first:
                ds = tempnim
                first = False
            else:
                ds = mvpa2.suite.vstack((ds,tempnim))
            chunkcount += 1

        #ds = []
        ##for chunkcount, (thisimg, thislabel) in enumerate(zip(samples,labels)):
            ##print chunkcount
        #ds = mvpa2.suite.fmri_dataset(
                #samples = samples,
                #targets = labels,
                #chunks = range(len(labels)),  #FIXME: change to runno?
                #mask = thismask
                #)
        #ds = mvpa2.suite.vstack(ds)
        return ds

    def detrend(self, ds):
        """
        Second-order detrending of data per chunk with the mean added back for
        a convenient percent signal change calculation.
        """
        dsmean = np.mean(ds.samples)
        mvpa2.suite.poly_detrend(ds, polyord=2, chunks_attr='chunks')
        ds.samples += dsmean # recover the detrended mean
        return ds

    def ds2evds(self, ds, dur=2):
        """
        Converts a dataset to an event-related dataset.

        :Args:
            ds

        :Kwargs:
            - offset (int, default: 2)
                How much labels should be shifted due to the hemodynamic lag. A
                good practice is to first plot data to see where the peaks are.
                Default is 2 as a typical TR is between 2 and 3 secs and the
                lag is around 6 seconds.
            - dur (int, default: 2)
                How many timepoints per condition. You may want to use a couple
                because the peak response may occupy more than a single
                timepoint (thus the default is 2).
        """

        # convert to an event-related design
        events = mvpa2.suite.find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
        # Remove the first and the last fixation period of each block
        # Also, we don't want any spillover between adjacent chunks
        events_temp = []
        for evno, ev in enumerate(events):
            if evno != 0 and evno != len(events)-1:
                if ev['chunks'] == events[evno-1]['chunks'] and \
                ev['chunks'] == events[evno+1]['chunks']:
                    events_temp.append(ev)
        events = events_temp
        for ev in events:
            #try:
                #ev['onset'] += offset
            #except:
                #import pdb; pdb.set_trace()
            if dur is not None:
                ev['duration'] = dur
        evds = mvpa2.suite.eventrelated_dataset(ds, events=events)

        durs = [ev['duration'] for ev in events]
        evds.sa['durations'] = mvpa2.suite.ArrayCollectable(name='durations',
            value=durs, length=len(durs))

        if self.rp['visualize']:
            plot_chunks(ds, evds, chunks=[0], shift_tp=0, tr=self.tr, fix=self.fix)

        return evds

    def timecourse(self, evds, offset=0):
        """
        For each condition, extracts all timepoints as specified in the evds
        window, and averages across voxels
        """
        baseline = evds[evds.sa.targets == self.fix]
        conds = evds[evds.sa.targets != self.fix]
        if np.min(baseline.sa.durations) < np.max(conds.sa.durations):
            warnings.warn('Some (all?) baseline events are shorter than '
                          'condition events, thus percent signal change is '
                          'computed w.r.t. the mean of all baseline events.')
            baseline = np.mean(baseline.samples)
        else:  # FIXME: what if baseline > conds?
            #import pdb; pdb.set_trace()
            baseline = evds.a.mapper[-1].reverse(baseline.samples)
            # average across all voxels and all blocks
            baseline = np.mean(np.mean(baseline,2),0)
        if np.any(baseline<0):
            warnings.warn('Some baseline values are negative')
        # now plot the mean timeseries and standard error
        header = ['cond', 'time', 'subj_resp']
        results = []
        for cond in conds.UT:
            evds_mean = conds[np.array([t == cond for t in conds.sa.targets])].samples
            # recover 3D evds structure: measurements x time points x voxels
            evds_mean = evds.a.mapper[-1].reverse(evds_mean)
            # average across all voxels and measurements
            evds_mean = np.mean(np.mean(evds_mean,2),0)
            thispsc = (evds_mean - baseline) / baseline * 100
            #time = np.arange(len(thispsc))*self.tr
            for pno, p in enumerate(thispsc):
                results.append([cond, (pno+self.offset)*self.tr, p])
        return header, results

    def signal(self, evds, values):
        """
        Extracts fMRI signal.

        .. warning:: must be reviewed

        :Args:
            - evds (event-related mvpa dataset)
            - values {'raw', 'beta', 't'}

        :Returns:
            fMRI signal for each condition (against the fixation condition)
        """
        header = ['cond', 'subj_resp']
        #results = []

        # calculate the mean per target per chunk (across trials)
        run_averager = mvpa2.suite.mean_group_sample(['targets','chunks'])
        evds_avg = evds.get_mapped(run_averager)

        if values.startswith('raw') or values == 'beta':
            # take fixation trials only
            baseline = evds_avg[evds_avg.sa.targets == self.fix].samples
            baseline = np.mean(baseline, 1)  # mean across voxels
            # replicate across chunks
            baseline = np.tile(baseline, len(evds_avg.UT))
            # replicate across voxels
            baseline = np.tile(baseline, (evds_avg.shape[1], 1)).T
            if values.startswith('raw'):
                evds_avg.samples = (evds_avg.samples - baseline) / baseline * 100
            else:
                evds_avg.samples = evds_avg.samples - baseline

        # calculate mean across conditions per target per voxel
        #target_averager = mvpa2.suite.mean_group_sample(['chunks'])
        #mean = evds_avg.get_mapped(target_averager)
        chunk_averager = mvpa2.suite.mean_group_sample(['targets'])
        mean = evds_avg.get_mapped(chunk_averager)
        results = [[i,j] for i,j in zip(mean.sa.targets, np.mean(mean, 1))]

        #if values.startswith('raw') or values == 'beta':
            #baseline = mean[mean.sa.targets == 0].samples
            #baseline = np.mean(baseline, 1)
        #for cond in mean.UT:
            #if cond != 0:
                #sel = np.array([t == cond for t in mean.sa.targets])
                #mean_cond = mean[sel].samples
                #mean_cond = np.mean(evds_cond, 1)
                #if values.startswith('raw'):
                    #mean_cond = (mean_cond - baseline) / baseline * 100
                #elif values == 'beta':
                    #mean_cond = mean_cond - baseline
                #evdsMean = np.mean(mean_cond)
                #results.append([cond, evdsMean])
        return header, results

    def univariate(self, evds, values):
        """Alias for :func:`signal`
        """
        return self.signal(evds, values)

    def correlation(self, evds, nIter=100):
        """
        Computes pairwise correlations between the two data splits in half.

        Reported as one minus a correlation over two to provide a dissimilarity
        measure between 0 and 1 as in :func:`~psychopy_ext.fmri.Analysis.svm()`.

        Data is normalized by subtracting the mean across conditions (targets)
        per chunk per voxel.

        :Args:
            evds (event-related mvpa dataset)

        :Kwargs:
            nIter (int, default: 100)
                Number of random splits in half of the entire dataset.

        :Returns:
            A header and a results matrix with four columns:
                - iter: iteration number
                - stim1.cond: first condition
                - stim2.cond: second condition
                - subj_resp: one minus the correlation value over two
                    (0: patterns identical, 1: patterns have nothing in common)
        """

        # calculate the mean per target per chunk (across trials)
        run_averager = mvpa2.suite.mean_group_sample(['targets','chunks'])
        evds_avg = evds.get_mapped(run_averager)
        numt = len(evds_avg.UT)

        # calculate mean across conditions per chunk per voxel
        target_averager = mvpa2.suite.mean_group_sample(['chunks'])
        mean = evds_avg.get_mapped(target_averager)
        # subtract the mean chunk-wise
        evds_avg.samples -= np.repeat(mean, numt, 0)

        if len(evds_avg.UC) == 1:
            raise Exception('You have only a single fMRI run. You need more '
                            'than one to run a correlational analysis.')

        #results = np.zeros((nIter,numt,numt))
        runtype = [0,1] * (len(evds_avg.UC)/2) + \
                   [-1] * (len(evds_avg.UC)%2)
                   # for odd number of chunks (will get rid of one)
        targets = evds_avg.UT
        header = ['iter', 'stim1.cond', 'stim2.cond', 'subj_resp']
        results = []
        for n in range(nIter):
            np.random.shuffle(runtype)
            evds_avg.sa['runtype'] = np.repeat(runtype,numt)

            evds_split1 = evds_avg[np.array([i==0 for i in evds_avg.sa.runtype])]
            run_averager = mvpa2.suite.mean_group_sample(['targets'])
            evds_split1 = evds_split1.get_mapped(run_averager)

            evds_split2 = evds_avg[np.array([i==1 for i in evds_avg.sa.runtype])]
            run_averager = mvpa2.suite.mean_group_sample(['targets'])
            evds_split2 = evds_split2.get_mapped(run_averager)

            result = mvpa2.clfs.distance.one_minus_correlation(evds_split1.samples,
                evds_split2.samples) / 2

            for i in range(0, numt):
                for j in range(0, numt):
                    results.append([n, targets[i], targets[j], result[i,j]])

        return header, results


    def svm(self, evds, nIter=100, clf=mvpa2.suite.LinearNuSVMC()):
        """
        Runs a support vector machine pairwise.

        .. note: Might be not the most efficient implementation of SVM, but
                 it is at least intuitive.

        Process:
            - Normalize data by subtracting the mean across voxels
              per chunk per condition (target).
            - Split data into a training set (about 75% of all values) and a testing
              set (about 25% of values), unless there are only two runs, in
              which case it is 50% training and 50% testing.
            - For each pair of conditions, train the classifier.
            - Then test on the average of the testing set, i.e., only on two
              samples. This trick usually boosts the performance (credit:
              Hans P. Op de Beeck)

        :Args:
            evds (event-related mvpa dataset)

        :Kwargs:
            - nIter (int, default: 100)
                Number of random splits into a training and testing sets.
            - clf (mvpa classfier, default: Linear Nu SVM)

        :Returns:
            A header and a results matrix with four columns:
                - iter: iteration number
                - stim1.cond: first condition
                - stim2.cond: second condition
                - subj_resp: one minus the correlation value
        """
        # calculate the mean per target per chunk (across trials)
        run_averager = mvpa2.suite.mean_group_sample(['targets','chunks'])
        evds_avg = evds.get_mapped(run_averager)
        numT = len(evds_avg.UT)

        # subtract the mean across voxels (per target per chunk)
        evds_avg.samples -= np.tile(np.mean(evds_avg, 1), (evds_avg.shape[1],1) ).T
        # and divide by standard deviation across voxels
        evds_avg.samples /= np.tile(np.std(evds_avg, axis=1, ddof=1),
            (evds_avg.shape[1],1) ).T

        ## NEW
        ntest_runs = len(evds_avg.UC) / 4
        if ntest_runs == 0:
            if len(evds_avg.UC) == 1:
                raise Exception('You have only a single fMRI. You need more '
                                'than one to run an SVM analysis.')
            ntest_runs = 1

        if len(evds_avg.UC)%2:
            runtype = [0]*(len(evds_avg.UC)-ntest_runs-1) + [1]*ntest_runs + [-1]
            # for odd number of chunks (will get rid of one)
        else:
            runtype = [0]*(len(evds_avg.UC)-ntest_runs) + [1]*ntest_runs
        ###

        ## OLD
        # if len(evds_avg.UC)%2:
        #     runtype = [0]*(len(evds_avg.UC)-3) + [1]*2 + [-1]
        #     # for odd number of chunks (will get rid of one)
        # else:
        #    runtype = [0]*(len(evds_avg.UC)-2) + [1]*2
        ###

        #targets = evds_avg.UT
        header = ['iter', 'stim1.cond', 'stim2.cond', 'subj_resp']
        results = []
        for n in range(nIter):
            sys.stdout.write('\r %d %%' % (100*n/float(nIter)))
            sys.stdout.flush()
            np.random.shuffle(runtype)
            evds_avg.sa['runtype'] = np.repeat(runtype,numT)

            evds_train = evds_avg[np.array([i==0 for i in evds_avg.sa.runtype])]
            evds_test = evds_avg[np.array([i==1 for i in evds_avg.sa.runtype])]
            ## NEW
            # boost results by averaging test patterns over chunks
            run_averager = mvpa2.suite.mean_group_sample(['targets'])
            evds_test = evds_test.get_mapped(run_averager)
            ###

            for i in range(0, numT):
                for j in range(0, numT):
                    targets = (evds_train.UT[i], evds_train.UT[j])
                    if i==j:
                        pred = None
                    else:
                        ind_train = np.array([k in targets for k in evds_train.sa.targets])
                        evds_train_ij = evds_train[ind_train]

                        ind_test = np.array([k in targets for k in evds_test.sa.targets])
                        # keep = np.logical_not(np.isnan(evds_test))
                        evds_test_ij = evds_test[ind_test]
                        # evds_test_ij = evds_test_ij[:,keep]
                        # fsel = mvpa2.suite.StaticFeatureSelection(keep)
                        # clf = mvpa2.suite.LinearNuSVMC()
                        # clf = mvpa2.suite.FeatureSelectionClassifier(clf, fsel)
                        clf.train(evds_train_ij)
                        #fsel = mvpa2.suite.SensitivityBasedFeatureSelection(
                            #mvpa2.suite.OneWayAnova(),
                            #mvpa2.suite.FractionTailSelector(0.05, mode='select', tail='upper'))
                        #fclf = mvpa2.suite.FeatureSelectionClassifier(clf, fsel)
                        #fclf.train(evds_train_ij)

                        # sensana = clf.get_sensitivity_analyzer()
                        # sens = sensana(evds_train_ij)
                        # inds = np.argsort(np.abs(sens.samples))
                        # inds = np.squeeze(inds)
                        # evds_train_ij.samples = evds_train_ij.samples[:,inds>=len(inds)-100]
                        # #import pdb; pdb.set_trace()
                        # clf.train(evds_train_ij)

                        # test_samp = evds_test_ij.samples[:,inds>=len(inds)-100]
                        # predictions = clf.predict(test_samp)
                        predictions = clf.predict(evds_test_ij.samples)
                        pred = np.mean(predictions == evds_test_ij.sa.targets)
                    results.append([n, targets[0], targets[1], pred])
        print

        return header, results


    def dissimilarity(self,
                   evds,
                   method = 'svm',
                   nIter = 10,  # how many iterations for  # have more for SVM
                   meanFunc = 'across voxels',
                   ):
        """
        DEPRECATED.
        Computes a dissimilarity (0 - very similar, 1 - very dissimilar) between
        two splits of data over multiple iterations. If method is correlation,
        dataset is split in half. If svm, leave-one-chunk.
        """

        numT = len(evds.UT)
        results = np.zeros((nIter,numT,numT))

        # prepare split of data
        # runtype is either 0 (train data or split1) or 1 (test data or split2)
        if method=='corr':  # splitHalf
            # if odd, then one of the runs will be spared (value -1)
            runtype = [0,1] * (len(evds.UC)/2) + [-1] * (len(evds.UC)%2)
        elif method=='svm':  # nFold split
            if len(evds.UC)%2:
                runtype = [0]*(len(evds.UC)-3) + [1,1] + [-1] # for odd
            else:
                runtype = [0]*(len(evds.UC)-2) + [1,1]

        # if corr: cvtype = len(evds.UC)/2
        # else: cvtype = 1
        # nfolds = mvpa2.suite.NFoldPartitioner(cvtype=cvtype,count=10,selection_strategy='equidistant')
        # import pdb; pdb.set_trace()

        for n in range(nIter):
            print n,
            # we want each iteration to have a different (random) split
            np.random.shuffle(runtype)
            # for each datapoint within a chunk, assign the same runtype
            evds.sa['runtype'] = np.repeat(runtype,len(evds.sa.chunks)/len(evds.UC))
            # create an average per target per chunk (per voxel)
            run_averager = mvpa2.suite.mean_group_sample(['targets','chunks'])
            evds_avg = evds.get_mapped(run_averager)

            # calculate mean and standard deviation across conditions per voxel
            ds_split_train = evds_avg[np.array([i==0 for i in evds_avg.sa.runtype])]
            mean_train = np.mean(ds_split_train,0)  # mean per voxel
            sd_train = np.std(ds_split_train, axis=0, ddof=1)

            ds_split_test = evds_avg[np.array([i==1 for i in evds_avg.sa.runtype])]
            mean_test = np.mean(ds_split_test,0)
            sd_test = np.std(ds_split_test, axis=0, ddof=1)

            targets = ds_split_train.UT
            if np.sum(targets != ds_split_test.UT)>0:
                sys.exit("Targets on the two splits don't match. Unbalanced design?")

            # filling in the results matrix
            for index,value in np.ndenumerate(results[n]):
                # target pair for that particular matrix cell
                indexT = (targets[index[0]],  targets[index[1]])

                ind_train = np.array([i in indexT  for i in ds_split_train.sa.targets])
                ds_train = ds_split_train[ind_train]
                ds_train.samples -= mean_train
                ds_train.samples /= sd_train

                ind_test = np.array([i in indexT  for i in ds_split_test.sa.targets])
                ds_test = ds_split_test[ind_test]
                ds_test.samples -= mean_test
                ds_test.samples /= sd_test

    #            if index[0] == index[1]:
    #                # import pdb; pdb.set_trace()
    #                halfT1 = len(ds_train.sa.targets)/2
    #                ds_train.sa.targets = np.array([1,2]*halfT1)
    #                halfT2 = len(ds_test.sa.targets)/2
    #                ds_test.sa.targets = np.array([1,2]*halfT2)

                if method=='corr':
                    cr = mvpa2.clfs.distance.one_minus_correlation(ds_train.samples,ds_test.samples)
                    # if one target then there's one correlation only
                    if index[0] == index[1]: acc = cr
                    else: acc = np.mean([ cr[0,1], cr[1,0] ])
                    results[n,index[0],index[1]] = acc

                elif method=='svm':
                    if index[0] == index[1]:  # can't do svm, so assume
                        results[n,index[0],index[1]] = 1
                    else:
                        clf = mvpa2.suite.LinearNuSVMC()
                        clf.train(ds_train)
                        predictions = clf.predict(ds_test.samples)
                        results[n,index[0],index[1]] = np.mean(predictions == ds_test.sa.targets)


                    # nfold = mvpa2.suite.NFoldPartitioner(cvtype=5,count=10,selection_strategy='equidistant')
                    # cvte = mvpa2.suite.CrossValidation(clf, HalfPartitioner(attr='runtype'),
                        # errorfx = lambda p, t: np.mean(p == t),  # this makes it report accuracy, not error
                        # enable_ca=['stats'])
                    # cvte(ds_pair)


        print
        if self.visualize:
            meanPerIter = np.mean(np.mean(results, 2), 1)
            cumMean = np.cumsum(meanPerIter)/range(1, len(meanPerIter)+1)
            plt.plot(cumMean)
            plt.show()

        return np.mean(results,0) # mean across folds


    def searchlight(self, ds):
        """ Basic searchlight analysis

        .. warning:: does not work yet
        """
        run_averager = mvpa2.suite.mean_group_sample(['targets', 'chunks'])
        ds = ds.get_mapped(run_averager)
        clf = mvpa2.suite.LinearNuSVMC()
        cvte = mvpa2.suite.CrossValidation(clf, mvpa2.suite.NFoldPartitioner(),
                            errorfx = lambda p, t: np.mean(p == t),enable_ca=['stats'])
        sl = mvpa2.suite.sphere_searchlight(cvte, radius=3, postproc=mvpa2.suite.mean_sample())

        pairs = [
            [(1,2),(1,3),(2,3)],
            [(4,5),(4,6),(5,6)],
            [(7,8),(7,9),(8,9)],
            [(10,11),(10,12),(11,12)]
            ]
        chance_level = .5
        for pair in pairs:
            thisds = ds[np.array([i in pair for i in ds.sa.targets])]
            res = sl(ds)
            resOrig = res.a.mapper.reverse(res.samples)
            print res_orig.shape
            fig = plt.figure()
            fig.subplot(221)
            plt.imshow(np.mean(resOrig.samples,0), interpolation='nearest')
            fig.subplot(222)
            plt.imshow(np.mean(resOrig.samples,1), interpolation='nearest')
            fig.subplot(223)
            plt.imshow(np.mean(resOrig.samples,2), interpolation='nearest')
            plt.show()
            sphere_errors = res.samples[0]
            res_mean = np.mean(res)
            res_std = np.std(res)
            import pdb; pdb.set_trace()

            sphere_errors < chance_level - 2 * res_std
        mri_args = {
            'background' : os.path.join(datapath, 'anat.nii.gz'),
            #'background_mask' : os.path.join(datapath, 'mask_brain.nii.gz'),
            #'overlay_mask' : os.path.join(datapath, 'mask_gray.nii.gz'),
            'cmap_bg' : 'gray',
            'cmap_overlay' : 'autumn', # YlOrRd_r # pl.cm.autumn
            'interactive' : cfg.getboolean('examples', 'interactive', True),
            }
        fig = plot_lightbox(overlay=map2nifti(dataset, sens),
                  vlim=(0, None), slices=18, **mri_args)


    def _plot_slice(self, volume_path, title='', rois=None, coords=None, fig=None):
        """
        Plots a slice from the three sides.

        .. note:: ROIs (masks) are averaged across all slices so that you
        would definitely get to see the ROIs independent of the plotted slice.

        :Args:
            volume_path (str)
                Path to the volume you want to plot.

        :Kwargs:
            - mask (str, default: None)
                Path to the ROI data. If it contains data (i.e., it comes from
                the `data_roi` folder), the data is
            - rois
            - coords (tuple of 3 or 4 ints; default: None)

            - fig (:class:`plot.Plot`; default: None)
                Pass an existing plot if you want to plot in it.

        """
        if fig is None:
            fig = plot.Plot(ncols=3)
            showplot = True
        else:
            showplot = False  # the caller probably has more plots to do

        labels = ['parasagittal', 'coronal', 'horizontal']
        allvols = glob.glob(volume_path)
        if len(allvols) == 0:
            raise Exception('Volume not found at %s' % volume_path)

        #for vol in allvols:
        vol = allvols[0]
        data = self.get_mri_data(vol)
        #import pdb; pdb.set_trace()
        if rois is not None:
            mask = sum([self.get_mri_data(roi) for roi in rois])
            coords = [int(np.mean(c)) for c in mask.nonzero()]

        if coords is None or len(coords) <= 2:
            coords = [m/2 for m in data.shape]  # middle
        if data.ndim == 4:  # time volume
            if len(coords) == 4:
                data = data[:,:,:,coords[3]]
            else:
                data = data[:,:,:,0]

        for i in range(3):
            if i == 0:
                mf = data[coords[i]]
            elif i == 1:
                mf = data[:, coords[i]]
            else:
                mf = data[:, :, coords[i]]
            ax = fig.next()
            ax.imshow(mf.T, cmap='gray', origin='lower',
                      interpolation='nearest')

            crosshair = [c for j,c in enumerate(coords) if j!=i]
            ax.axvline(x=crosshair[0], color='g', alpha=.5)
            ax.axhline(y=crosshair[1], color='g', alpha=.5)

            ax.set_title('%s at %s' % (labels[i], coords[i]))

            if rois is not None:
                mean_mask = np.mean(mask, i).T
                # make it uniform color
                mean_mask[np.nonzero(mean_mask)] = 1.  # ROI voxels are 1
                mean_mask[mean_mask==0] = np.nan  # non-ROI voxels are nan
                mask_rgba = np.zeros(mean_mask.shape + (4,))  # add transparency
                mask_rgba[:] = np.nan  # default is nan
                mask_rgba[:,:,0] = mean_mask  # make mask red
                mask_rgba[:,:,3] = mean_mask  # transparency should have nans
                ax.imshow(mask_rgba, alpha=.5,
                        origin='lower', interpolation='nearest')
                # could also use the following:
                #import matplotlib.patches as mpatches
                #from matplotlib.collections import PatchCollection
                #patches = []
                #rect = mpatches.Rectangle((0,0), 5,5)
                ##ax.add_patch(rect)
                #patches.append(rect)
                #collection = PatchCollection(patches, facecolor=(1,0,0,1), edgecolor='none')
                #ax.add_collection(collection)
        if showplot:
            fig.show()

    def _check_exists(self, filepatt):
        files = glob.glob(filepatt)
        if len(files) > 0:
            return True
        else:
            return False

    def plot_roi(self, roi):
        """
        Plots Regions of Interest (ROIs) on the functional data.
        """
        subjid = self.info['subjid']
        if not isinstance(subjid, str):
            raise TypeError('subjid is supposed to be a string, '
                            'but got %s instead' % subjid)
        #allROIs = []
        #for ROIs in self.rois:
            #for ROI in ROIs[2]:
                #theseROIs = glob.glob((self.paths['rois'] + ROI + '.nii*') %subjid)
                #theseROIs.sort()
                #allROIs.extend(theseROIs)
        #if len(allROIs) == 0:
            #raise Exception('Could not find matching ROIS at %s' %
                             #(self.paths['rois'] %subjid))
        #else:
            #allROIs = (None, '-'.join([r[1] for r in self.rois]), allROIs)

        filepatts = [{'name': 'Anatomical',
                      'path': self.paths['data_struct'] %subjid + 'wstruct*'},
                     {'name': 'Functional',
                      'path': self.paths['data_fmri'] %subjid + 'func_*_main.nii*'},
                     {'name': 'Slice timing corrected',
                      'path': self.paths['data_fmri'] %subjid + 'afunc_*_main.nii*'},
                     {'name': 'Motion corrected (realigned)',
                      'path': self.paths['data_fmri'] %subjid + 'r*func_*_main.nii*'},
                     {'name': 'Normalized',
                      'path': self.paths['data_fmri'] %subjid + 'w*func_*_main.nii*'},
                     {'name': 'Smoothed',
                      'path': self.paths['data_fmri'] %subjid + 's*func_*_main.nii*'}
                      ]
        exist = [f for f in filepatts if self._check_exists(f['path'])]
        #import pdb; pdb.set_trace()
        sns.set(style='dark')
        fig = plot.Plot(nrows=len(exist), ncols=3, sharex=False, sharey=False)

        for f in exist:
            self._plot_slice(f['path'], title=f['name'], fig=fig,
                             rois=[self.paths['rois'] %subjid +roi])

        ## plot ROI values
        #ds = self.extract_samples(subjid, self.info['runtype'],
            #allROIs, values=self.rp['values'])
        #if not self.rp['values'].startswith('raw'):
            #nans = np.sum(np.isnan(ds)) * 100. / ds.samples.size
            #title = '%d%% of ROI voxels are nans' % nans
        #else:
            #title = ''
        #ax = fig.next()
        #ax.hist(ds.samples.ravel(), label=title)
        #ax.set_xlabel('signal')
        #ax.set_ylabel('# of voxels (all timepoints)')
        #fig.hide_plots([-2,-1])
        fig.show()

    def plot_ds(self, ds, chunk=0):
        sns.set(style='dark')
        plt.imshow(ds.samples[ds.sa.chunks == 0], interpolation='none', cmap='coolwarm')
        plt.xlabel('voxels')
        plt.ylabel('trials')
        plt.title('run %d' % chunk)
        plt.show()

    def _calc_nans(self):
        pass


    def genFakeData(self, nchunks = 4):

        def fake(nconds = 12,nvoxels = 100):
            # each voxel response per condition
            fakecond1 = np.array([0.5,1.]*(nvoxels/2))
            # fakecond1 = np.random.normal( loc=1,scale=1,size=(nvoxels,) )
            # ROI's response to each condition
            fakecond1 = np.tile( fakecond1, (nconds/2,1) )
            # add noise
            fakeds1 = fakecond1 + np.random.random((nconds/2,nvoxels))/10.

            fakecond2 = np.array([1.,.5,1.,5]*(nvoxels/4))
    #        fakecond2 = np.random.normal(loc=3,scale=1,size= (nvoxels,) )
            fakecond2 = np.tile( fakecond2, ( nconds/2,1 ) )
            fakeds2 = fakecond2 + np.random.random((nconds/2,nvoxels))/10.

            fakechunk = np.vstack((fakeds1,fakeds2,fakeds2[:,::-1],fakeds1[:,::-1]))
            targets = range(1,nconds+1) + range(nconds,0,-1)
            fakechunk = mvpa2.suite.dataset_wizard(samples=fakechunk, targets=targets)
            return fakechunk

        fakeds = mvpa2.suite.multiple_chunks(fake,nchunks)
        return fakeds

    def read_csvs(self, path):
        """
        Reads multiple CSV files and concatinates tehm into a single
        `pandas.DataFrame`

        :Args:
            path (str)
                Where to find the data
        """
        df_fnames = glob.glob(path)
        dfs = []
        for dtf in df_fnames:
            dfs.append(pandas.read_csv(dtf))
        return pandas.concat(dfs, ignore_index=True)

    def roi_params(self,
        rp,
        subROIs = False,
        suppressText = True,
        space = 'talairach',
        spm = False
        ):

        """
        Calculates mean coordinates and the number of voxels of each given ROI.

        **Parameters**
            rp: Namespace (required)
                Run parameters that are parsed from the command line
            subROIs: True or False
                If True, then subROIs are not combined together into an ROI
            suppressText: True or False
                If True, then nothing will be printed out
            space: talairach or native
                Choose the output to be either in native voxel space or in Talairach coordinates
            spm: True or False
                If True, then the coordinates in the voxel space are provided with
                indices +1 to match MatLab's convention of starting arrays from 1.

        """

        if subROIs: names = ['subjid','roi','subROI','x','y','z','numVoxels']
        else: names = ['subjid','roi','x','y','z','numVoxels']
        recs = []

        # allCoords = np.zeros((1,4))
        for subjidno, subjid in enumerate(rp.subjid_list):

            for ROI_list in rp.rois:

                allROIs = []
                for thisROI in ROI_list[2]:
                    allROIs.extend(q.listDir(scripts.core.init.paths['recDir'] %subjid,
                        pattern = thisROI + '\.nii', fullPath = True))
                #import pdb; pdb.set_trace()
                if allROIs != []:
                    SForm = nb.load(allROIs[0]).get_header().get_sform()

                    # check for overlap
                    # if subjid == 'twolines_06': import pdb; pdb.set_trace()
                    print [os.path.basename(subROI) for subROI in allROIs]
                    #
                    mask = sum([np.squeeze(nb.load(subROI).get_mri_data()) for subROI in allROIs])
                    if not suppressText:
                        overlap = mask > 2
                        if np.sum(overlap) > 0:
                            print 'WARNING: Overlap in %(subjid)s %(ROI)s detected.'\
                            %{'subjid': subjid, 'ROI': ROI_list[1]}


                    if not subROIs: allROIs = [mask]
                    for subROI in allROIs:

                        if subROIs: subROIname = os.path.basename(os.path.abspath(subROI)).split('.')[0]
                        else: subROIname = ROI_list[1]
                        #import pdb; pdb.set_trace()
                        if subROIs: thisROI = nb.load(subROI).get_mri_data()
                        else: thisROI = subROI
                        transROI = np.transpose(thisROI.nonzero())

                        meanROI = np.mean(transROI,0)[1:]
                        meanROI = meanROI[::-1] # reverse the order per convention

                        # convert to the Talairach coordinates
                        if space == 'talairach':
                            meanROI = np.dot(SForm, np.concatenate((meanROI,[1]))) # convert
                            meanROI = meanROI[:-1] # remove the last coordinate (slice number)
                        else:
                            meanROI = [m+spm for m in meanROI] # +1 to correct for SPM coords

                        if subROIs:
                            recs.append((subjid,ROI_list[1],subROIname)+tuple(meanROI)+(transROI.shape[0],))
                        else:
                            recs.append((subjid,subROIname)+tuple(meanROI)+(transROI.shape[0],))



        ROIparams = tb.tabarray(records = recs, names = names)

        if not suppressText:
            if subROIs: on = ['ROI','subROI']
            else: on = ['ROI']
            ROImean = ROIparams.aggregate(On = on, AggFunc = np.mean,
                AggFuncDict = {'subjid': lambda x: None})

            xyz = ROIparams[['x','y','z']].extract().reshape((len(rp.subjid_list),-1,3))
            xyzErr = np.std(xyz, axis = 0, ddof = 1)

            # sort ROImean
            numPerSubj = xyz.shape[1]
            order = ROIparams[:numPerSubj][on]
            order = order.addcols(range(len(order)), names=['order'])
            order.sort(order=on)
            ROImean.sort(order=on)
            ROImean = ROImean.addcols(order[['order']].extract(), names = 'order')
            ROImean.sort(order = 'order')

            lenROI = min([len(ROI) for ROI in ROImean['ROI']])
            if subROIs: lenSubROI = min([len(ROI) for ROI in ROImean['subROI']])
            print
            print ROIparams.dtype.names[1:]
            for i, line in enumerate(ROImean):
                print line['ROI'].ljust(lenROI+2),
                if subROIs: print line['subROI'].ljust(lenSubROI+2),
                print '%3d' %np.round(line['x']),
                print u'\xb1 %d  ' %np.round(xyzErr[i,0]),
                print '%3d' %np.round(line['y']),
                print u'\xb1 %d  ' %np.round(xyzErr[i,1]),
                print '%3d' %np.round(line['z']),
                print u'\xb1 %d  ' %np.round(xyzErr[i,2]),
                print '%4d' %np.round(line['numVoxels'])

        return ROIparams




class Preproc(object):

    def __init__(self,
                 paths,
                 info=None,
                 rp=None
                 ):
        """
        Generates batch scripts from SPM preprocessing.

        .. note:: Presently, only batch scripts for statistical analyses in SPM
                  are available.

        :Args:
            paths (dict of str:str pairs)
                A dictionary of paths where data is stored. Expected to have at
                least the following keys:

                - 'fmri_root' for moving the original realignment parameter
                  (prefix `rp`) file
                - 'data_behav' - where to find behavioral data with condition
                  labels (passed`condcol` variable), onsets, and durations
                - 'data_fmri' - where to find fMRI functional data
                - 'rec' (for ROIs from surface reconstruction in Caret or so)
                - 'data_rois' (for storing the extracted signals in these ROIs)

        """
        self.info = OrderedDict([
            ('subjid', 'subj'),
            ('runtype', 'main'),
            ])
        self.rp = OrderedDict([
            ('method', 'timecourse'),
            ('values', 'raw'),
            ('no_output', False),
            ('debug', False),
            ('verbose', True),
            ('visualize', False),
            ('force', False),
            ('dry', False),
            ('reuserois', True),
            ])
        self.paths = paths
        if info is not None:
            self.info.update(info)
        if rp is not None:
            self.rp.update(rp)

    def split_rp(self, subjid):
        """
        Splits the file that has realignment information by run.

        This is used for stats as each run with its covariates has to be
        entered separately.

        Assumptions:
            - Realignment parameters are supposed to be called like
              `rp_afunc_<runno>.txt`
            - Functional data is expected to be in the `paths['data_fmri']` folder
            - `paths['fmri_root']` should also be specified so that the original
              rp file would be backuped there.

        :Args:
            subjid (str)
                For which subject the split is done.
        """
        func_img = glob.glob(self.paths['data_fmri'] % subjid + 'func_*_*.nii')
        func_img.sort()
        rp_pattern = self.paths['data_fmri'] % subjid + 'rp_afunc_*.txt'
        rpfiles = glob.glob(rp_pattern)
        rpfiles.sort()

        if len(rpfiles) == 0:  # probably split_rp has been done before
            if self.rp['verbose']:
                print 'No rp files like %s found' % rp_pattern
        else:
            rp = []
            for rpfile in rpfiles:
                f = open(rpfile)
                rp.extend(f.readlines())
                f.close()
                rp_bck = self.paths['fmri_root'] % subjid
                rp_bck += os.path.basename(rpfile)
                if not self.rp['dry']:
                    shutil.move(rpfile, rp_bck)
                else:
                    print '%s --> %s' % (rpfile, rp_bck)

            last = 0
            for func in func_img:
                runno = func.split('.')[0].split('_')[-2]
                dynscans = self.get_mri_data(func).shape[3] # get number of acquisitions

                runtype = func.split('.')[0].split('_')[-1]
                outname = self.paths['data_fmri']%subjid + 'rp_%s_%s.txt' %(runno,runtype)

                if not self.rp['dry']:
                    f = open(outname, 'w')
                    f.writelines(rp[last:last+dynscans])
                    f.close()
                else:
                    print '%s: %s' % (func, outname)

                last += dynscans

            if len(rp) != last:
                warnings.warn('Splitting was performed but the number of '
                       'lines in the rp file did not match the total number of '
                       'scans in the functional runs.')


    def gen_stats_batch(self, condcol='cond', descrcol='name'):
        """
        Generates a batch file for statistical analyses in SPM.

        :Kwargs:
            - condcol (str)
                Column in the data files with condition labels (numbers)
            - descrcol (str)
                Column in the data files with condition names

        """
        subjid = self.info['subjid']
        runtype = self.info['runtype']
        if isinstance(runtype, str):
            runtype = [runtype]

        self.split_rp(subjid)
        # set the path where this stats job will sit
        # all other paths will be coded as relative to this one
        curpath = os.path.join(self.paths['fmri_root'] %subjid,'jobs')
        f = open(os.path.join(curpath,'stats.m'),'w')
        f.write("spm('defaults','fmri');\nspm_jobman('initcfg');\nclear matlabbatch\n\n")

        for rtNo, runtype in enumerate(runtype):
            analysis_dir = os.path.normpath(os.path.join(os.path.abspath(self.paths['fmri_root']%subjid),
            'analysis',runtype))
            try:
                os.makedirs(analysis_dir)
            except:
                print ('WARNING: Analysis folder already exists at %s' %
                        os.path.abspath(analysis_dir))
            # make analysis path relative to stats.m
            analysis_dir_str = ("cellstr(spm_select('CPath','%s'))" %
                                os.path.relpath(analysis_dir, curpath))
            dataFiles = glob.glob(self.paths['data_behav'] % subjid +\
                                  'data_*_%s.csv' %runtype)
            dataFiles.sort()
            regressorFiles = glob.glob(self.paths['data_fmri'] % subjid +\
                                       'rp_*_%s.txt' %runtype)
            regressorFiles.sort()
            f.write("matlabbatch{%d}.spm.stats.fmri_spec.dir = %s;\n" %
                    (3*rtNo+1, analysis_dir_str))
            f.write("matlabbatch{%d}.spm.stats.fmri_spec.timing.units = 'secs';\n" %
                    (3*rtNo+1))
            f.write("matlabbatch{%d}.spm.stats.fmri_spec.timing.RT = 2;\n" %
                    (3*rtNo+1))

            for rnNo, dataFile in enumerate(dataFiles):
                runno = int(os.path.basename(dataFile).split('_')[1])

                data = np.recfromcsv(dataFile, case_sensitive = True)
                swapath = os.path.relpath(self.paths['data_fmri']%subjid, curpath)
                f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).scans = "
                        "cellstr(spm_select('ExtFPList','%s',"
                        "'^swafunc_%02d_%s\.nii$',1:168));\n" %
                        (3*rtNo+1, rnNo+1, swapath, runno, runtype))
                # "cellstr(spm_select('ExtFPList','%s','^swafunc_%02d_%s\.nii$',1:168));\n" %(os.path.abspath(self.paths['data_fmri']%subjid),runno,runtype))
                conds = np.unique(data[condcol])
                if runtype == 'mer':
                    conds = conds[conds!=0]

                for cNo, cond in enumerate(conds):
                    agg = data[data[condcol] == cond]
                    f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d)."
                            "cond(%d).name = '%d|%s';\n" %
                            (3*rtNo+1, rnNo+1, cNo+1, cond, agg[descrcol][0]))
                    if 'blockNo' in agg.dtype.names:
                        onsets = []
                        durs = []
                        for block in np.unique(agg['blockNo']):
                            onsets.append(agg[agg['blockNo']==block]['onset'][0])
                            durs.append(np.around(sum(agg[agg['blockNo']==block]['dur']),
                             decimals=1))
                    else:
                        onsets = np.round(agg['onset'])
                        durs = agg['dur']
                        # for fixation we remove the first and the last blocks
                        if cond == 0:
                            onsets = onsets[1:-1]
                            durs = durs[1:-1]

                    f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).cond(%d).onset = %s;\n" %(3*rtNo+1,rnNo+1,cNo+1,onsets))
                    f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).cond(%d).duration = %s;\n" %(3*rtNo+1,rnNo+1,cNo+1,durs))

                regpath = os.path.relpath(regressorFiles[rnNo], curpath)
                regpath_str = "cellstr(spm_select('FPList','%s','^%s$'))" % (os.path.dirname(regpath), os.path.basename(regpath))
                f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).multi_reg = %s;\n\n" %(3*rtNo+1,rnNo+1,regpath_str))

            spmmat = "cellstr(fullfile(spm_select('CPath','%s'),'SPM.mat'));\n" % os.path.relpath(analysis_dir, curpath)
            f.write("matlabbatch{%d}.spm.stats.fmri_est.spmmat = %s" % (3*rtNo+2, spmmat))
            f.write("matlabbatch{%d}.spm.stats.con.spmmat = %s" %(3*rtNo+3,
                spmmat))

            if runtype == 'loc':
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.name = 'all > fix';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.convec = [-2 1 1];\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{2}.tcon.name = 'objects > scrambled';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{2}.tcon.convec = [0 1 -1];\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{2}.tcon.sessrep = 'repl';\n\n\n" %(3*rtNo+3))
            elif runtype == 'mer':
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.name = 'hor > ver';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.convec = [1 -1];\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';\n\n\n" %(3*rtNo+3))
            else:
                # f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.name = 'all > fix';\n" %(3*rtNo+3))

                conds = np.unique(data[condcol])
                descrs = []
                # skip fixation condition as it's our baseline
                for cond in conds[1:]:
                    descrs.append((cond,
                        data[data[condcol]==cond][descrcol][0]))
                # descrs = np.unique(data['descr'])
                # descrs = descrs[descrs != 'fixation']

                # thisCond = ' '.join(['-1']+['1']*len(descrs))
                # f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.convec = [%s];\n" %(3*rtNo+3,thisCond))
                # f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';\n" %(3*rtNo+3))

                # dNo is corrected with +2: +1 for Matlab and +1 because we
                # have 'all > fix'
                # for now disabled
                for dNo, descr in enumerate(descrs):
                    f.write("matlabbatch{%d}.spm.stats.con.consess{%d}.tcon.name = '%d|%s';\n" %(3*rtNo+3,dNo+1,descr[0],descr[1]))
                    thisCond = [-1] + [0]*dNo + [1] + [0]*(len(descrs)-dNo-1)
                    f.write("matlabbatch{%d}.spm.stats.con.consess{%d}.tcon.convec = %s;\n" %(3*rtNo+3,dNo+1,thisCond) )
                    f.write("matlabbatch{%d}.spm.stats.con.consess{%d}.tcon.sessrep = 'both';\n" %(3*rtNo+3,dNo+1))
                f.write('\n\n')

        f.write("save('stats.mat','matlabbatch');\n")
        f.write("%%spm_jobman('interactive',matlabbatch);\n")
        f.write("spm_jobman('run',matlabbatch);")
        f.close()


def make_full(distance):
    res = np.nan*np.ones(distance.shape)
    iu = np.triu_indices(len(distance),k=1)  # upper triangle less diagonal
    il = np.tril_indices(len(distance),k=-1)  # lower triangle less diagonal
    res[iu] = distance[iu]
    res = res.T
    res[iu] = distance[iu]
    return res

def plot_timecourse(df, plt=None, cols='name', **kwargs):
    """Plots an fMRI time course for signal change.

    :Args:
        df (:class:`pandas.DataFrame`)
            A DataFrame with fMRI signal change computed.

    :Kwargs:
        - title (str, default: '')
            Title for the plot (i.e., for the current axis, not the whole figure)
        - plt (:class:`plot.Plot`, default: None)
            The plot you're working on.
        - cols (str or list of str, default: 'name')
            Column names to plot as separate conditions (different curves)

    """
    if plt is None:
        plt = plot.Plot(sharex=True, sharey=True)
    #f, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    #axes = axes.ravel()
    #for i, ax in enumerate(plt.axes):
        #ax.plot(range(5), np.arange(5)*i)
    agg = stats.aggregate(df, subplots='roi', values='subj_resp', rows='time',
                          cols=cols, yerr='subjid')
    ax = plt.plot(agg, kind='line',
        xlabel='Time since trial onset, s',
        ylabel='Signal change, %', **kwargs)
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for thisax in ax:
        thisax.axhline(linestyle='-', color='0.6')
    plt.tight_layout()
    #plt.show()
    return plt

def plot_similarity(similarity, names=None, percent=False):
    similarity = make_symmetric(similarity)
    trace = similarity.trace()/len(similarity)
    offdiag = (np.sum(similarity) - similarity.trace()) / len(similarity) / (len(similarity)-1)
    print '%.2f' %trace,
    print '%.2f' %offdiag,
    iu = np.triu_indices(len(similarity),k=1)  # upper triangle less diagonal
    rel = np.corrcoef(similarity[iu],similarity.T[iu])[0,1]
    print '%.2f' %rel
#    import pdb; pdb.set_trace()
    if percent: plot_data = similarity*100
    else: plot_data = similarity
    im = plt.imshow(plot_data,interpolation='none',vmin=.45,vmax=.86)
    plt.colorbar(im, use_gridspec=True)
    # plt.tight_layout()

    if not names is None:
        names = [n[1] for n in names]
        locs, labels = plt.xticks(range(plot_data.shape[1]), names)
        plt.setp(labels, 'rotation', 'vertical')
        locs, labels = plt.yticks(range(plot_data.shape[0]), names)
    for index,value in np.ndenumerate(plot_data):
        if np.isnan(value): h = ''
        else:
            if percent: h = '%d' %(value*100)
            else: h = '.%d' %(value*100)
        plt.text(index[1]-.5,index[0]+.5,h)
    return im

def plot_hcluster(similarity, names):
    import hcluster
    similarity = make_symmetric(similarity)
    sim2 = similarity - .5
    sim2[sim2<0] = 0
    # distance = Orange.core.SymMatrix(1-similarity)
    # root = Orange.clustering.hierarchical.HierarchicalClustering(distance)
    tree = hcluster.hcluster(sim2)
    imlist = [ str(i[0]) + '-' + i[1] for i in names]
    dendogram = hcluster.drawdendrogram(tree,imlist,jpeg='sunset.jpg')
    plt.imshow(dendogram, cmap=plt.cm.gray)

def plot_mds(similarity, names):
    #
    similarity = make_symmetric(similarity)
    sim2 = similarity - .5
    sim2[sim2<0] = 0
    #import pdb; pdb.set_trace()
    distance = Orange.core.SymMatrix(sim2)
    #import pdb; pdb.set_trace()
    mds = Orange.projection.mds.MDS(distance)
    mds.run(100)
    for (x, y), name in zip(mds.points,names):
        plt.plot((x,),(y,),'ro')
        plt.text(x,y,name[1])

def mean_diag_off(matrix):
    trace = matrix.trace()/len(matrix)
    offdiag = (np.sum(matrix) - matrix.trace()) / len(matrix) / (len(matrix)-1)
    return [trace,offdiag]

def avg_blocks(matrix, coding):
    coding = np.array(coding)
    coding_int = coding[np.not_equal(coding, None)] # remove nones
#    try:
#        np.bincount(coding_int)>1
#    except:
#        import pdb; pdb.set_trace()
    coding_int = coding_int.astype(np.int)
    if not np.all(np.bincount(coding_int)>1):
        print np.bincount(coding_int)
        sys.exit('You have a single occurence of some entry')
    else:
        uniquec = np.unique(coding_int)
        avg = np.zeros( (len(uniquec),len(uniquec)) )
        for i,ui in enumerate(uniquec):
            indi = coding == ui
            for j,uj in enumerate(uniquec):
                indj = coding == uj
                ind = np.outer(indi,indj)
                np.fill_diagonal(ind,False)
                avg[i,j] = np.mean(matrix[ind])
    return avg

def plot_psc(*args, **kwargs):
    """
    DEPRECATED. Plots percent signal change of raw data
    """
    ax = plot.pivot_plot(marker='o', kind='line', *args, **kwargs)
    ax.set_xlabel('Time since trial onset, s')
    ax.set_ylabel('Signal change, %')
    ax.axhline(linestyle='--', color='0.6')
    # plt.xlim(( 0,evdsMean.shape[1]+1 ))
    # plt.ylim((-.5,2.))
    ax.legend(loc=0).set_visible(False)
    return ax

def plot_vol(struct=None, func=None, rois=None, coords=None, paths=None):
    """
    Plots Regions of Interest (ROIs) on the functional data.
    """
    if paths is None:
        paths = {'data_roi': '', 'data_struct': '', 'data_func': ''}

    allROIs = []
    for ROIs in self.rois:
        for ROI in ROIs[2]:
            theseROIs = glob.glob((self.paths['rois'] + ROI + '.nii*') %subjid)
            theseROIs.sort()
            allROIs.extend(theseROIs)
    if len(allROIs) == 0:
        raise Exception('Could not find matching ROIS at %s' %
                         (self.paths['rois'] %subjid))
    else:
        allROIs = (None, '-'.join([r[1] for r in self.rois]), allROIs)

    fig = plot.Plot(nrows=2, ncols=3, sharex=False, sharey=False)
    try:
        self._plot_slice(self.paths['data_struct']
                 %subjid + 'wstruct*', fig=fig)
    except:
        pass
    #try:
    self._plot_slice(self.paths['data_fmri']
                 %subjid + 'afunc_01_main.nii*', rois=allROIs[2], fig=fig) #swmean
    #except:
        #pass

    # plot ROI values
    ds = self.extract_samples(subjid, self.info['runtype'],
        allROIs, values=self.rp['values'])
    if not self.rp['values'].startswith('raw'):
        nans = np.sum(np.isnan(ds)) * 100. / ds.samples.size
        title = '%d%% of ROI voxels are nans' % nans
    else:
        title = ''
    ax = fig.next()
    ax.hist(ds.samples.ravel(), label=title)
    ax.set_xlabel('signal')
    ax.set_ylabel('# of voxels (all timepoints)')
    fig.hide_plots([-2,-1])
    fig.show()

def plot_chunks(ds, evds, chunks=None, shift_tp=0, tr=None, fix=0):
    events = mvpa2.suite.find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
    # which chunks to display
    if chunks == None: chunks = ds.UC

    # get colors and assign them to targets
    ncolors = len(ds.UT)
    cmap = mpl.cm.get_cmap('Paired')
    norm = mpl.colors.Normalize(0, 1)
    z = np.linspace(0, 1, ncolors + 2)
    z = z[1:-1]
    colors_tmp = cmap(norm(z))
    colors = {}
    for target, color in zip(ds.UT,colors_tmp): colors[target] = color
    colors[fix] = (1,1,1,.1)

    chunk_len = ds.shape[0] / len(ds.UC)
    event_dur = evds.a.mapper[1].boxlength

    plt = plot.Plot(nrows=len(chunks))
    for chunkno, chunk in enumerate(chunks):
        sel = np.array([i==chunk for i in evds.sa.chunks])
        evds_sel = evds[sel]
        sel_ds = np.array([i==chunk for i in ds.sa.chunks])
        mean_per_chunk = np.mean(ds[sel_ds],1) # mean across voxels

        if shift_tp == 0:
            title = 'Run %s' % chunk
        else:
            title = 'Run %s with conditions shifted by %d' %(chunk, shift_tp)

        if tr is None:
            time = np.arange(len(mean_per_chunk))
            xlabel = 'acquisition number'
        else:
            time = np.arange(0, len(mean_per_chunk), tr)
            xlabel = 'time (s)'

        plt.plot(mean_per_chunk, kind='line',
            title=title, xlabel=xlabel,
            ylabel='signal intensity (absolute)')

        legends = []
        labels = []
        for evno in range(len(evds_sel.sa.event_onsetidx)):
            xmin = evds_sel.sa.event_onsetidx[evno] % chunk_len + shift_tp
            label = evds_sel.sa.targets[evno]
            h = plt.axvspan(
                xmin = xmin + shift_tp - .5,
                xmax = xmin + evds_sel.sa.durations[evno] - .5,
                facecolor = colors[label],
                alpha=0.5)
            if label not in labels:
                legends.append(h)
                labels.append(label)

            plt.axvline(x=evds_sel.sa.event_onsetidx[evno] % chunk_len + shift_tp - .5,
                        color='black', alpha=.3)

        order = np.argsort(labels)
        labels = np.array(labels)[order]
        legends = np.array(legends)[order]
        plt.legend(legends, labels, frameon=True, framealpha=.5)

    plt.show()


"""
Other tools
"""
def make_roi_pattern(rois):
    """
    Takes ROI names and expands them into a list of:
        - ROI name as given
        - Pretty ROI name for output
        - ROI names with * prepended and appended for finding these ROIs easily
            using `glob`

    :Args:
        rois (list of str or tuples):
            A list of ROI names, e.g., `['V1', (['rh_V2','lh_V2'], 'V2')]`.
            If an element is a tuple, the first element is ROI names and the
            second one is their "pretty" (unifying) name for printing.

    :Returns:
        A list of ROI names in the format described above, e.g.
        `[('V1','V1','*V1*'), (['rh_V2','lh_V2'], 'V2', ['*rh_V2*','*lh_V2*'])]`
    """
    def makePatt(ROI):
        """Expands ROI patterns by appennding *"""
        return ['*'+thisROI+'*' for thisROI in ROI]

    if not isinstance(rois, list) and not isinstance(rois, tuple):
        rois = [rois]
    ROIs = []
    for ROI in rois:
        if type(ROI) == tuple:  # renaming is provided
            ROIs.append(ROI + (makePatt(ROI[0]),))
        elif type(ROI) == list:  # a list of ROIs is provided
            ROIs.append((ROI, '-'.join(ROI), makePatt(ROI)))
        else:  # just a single ROI name provided
            ROIs.append((ROI, ROI, makePatt([ROI])))
    return ROIs

def _img2data(src):
    """Modified from mvpa2.datasets.mri to resolve
    world-coordinates issue.
    """
    # break early of nothing has been given
    # XXX feels a little strange to handle this so deep inside, but well...
    if src is None:
        return None

    # let's try whether we can get it done with nibabel
    if isinstance(src, basestring):
        # filename
        img = nb.load(src)
    else:
        # assume this is an image already
        img = src
    if isinstance(img, nb.spatialimages.SpatialImage):
        header = img.get_header()
        data = get_data_world(img)
        # AFNI stuff omitted as we only work with SPM
        # ...
        # nibabel image, dissect and return pieces
        return mvpa2.datasets.mri._get_txyz_shaped(data), header, img.__class__
    else:
        # no clue what it is
        return None

def get_data_world(img):
    data = img.get_data()
    ori = nb.io_orientation(img.get_affine())
    data = nb.apply_orientation(data, ori)
    return data


class GenHRF(object):

    def __init__(self,
                 paths,
                 #info=None,
                 #rp=None,
                 subjid='subj_01'
                 ##roi_coords=np.s_[50:60, 55:60, 14:18]
                 ):
        #if paths is None:
            #self.paths = PATHS
        #else:
        self.paths = paths
        self.subjid = subjid
        #nruns = 8  # will simulate 8 runs
        #tr = 2  # TR in sec
        ##ntr = 60*5/tr  # simulate 5 min runs
        #blocklen = 4  # TRs

        ##conds = OrderedDict([()])
        #nconds = 4
        #labels = ['fixation', 'cond1', 'cond2', 'cond3', 'cond4']
        #weights = [0, 1, 2, 1.5, .5]  # how much each condition is preferred (relatively)
        #dur = tr

    def gen_test_data(self,
                      nruns=8,
                      tr=2,
                      blocklen=4,
                      nconds=4,
                      include_fix=True,
                      labels=None,
                      weights=None,
                      roi_coords=np.s_[60:64,17:21,10:18]  # right LO
                      ):
        if weights is None:
            if include_fix:
                weights = np.ones(nconds+1) #np.arange(nconds+1)
            else:
                weights = np.ones(nconds) #np.arange(1, nconds+1)
        if labels is None:
            labels = ['cond%d' % i for i in range(nconds)]
            if include_fix:
                labels = ['fix'] + labels

        if len(weights) != len(labels):
            raise Exception

        exp.try_makedirs(self.paths['data_fmri'] % self.subjid)
        exp.try_makedirs(self.paths['data_behav'] % self.subjid)
        exp.try_makedirs(self.paths['rois'] % self.subjid)
        #nmeasr = len(conds)*blocklen
        self.prepare_roi(roi_coords)
        print 'Generating fMRI data...',
        for runno in range(nruns):
            print runno+1,
            df = self._gen_behav_data(labels, runno, blocklen=4)

            weights_full = (df.cond>0) * np.take(weights, df.cond)
            self._gen_fmri_data(df, weights_full, roi_coords, runno, blocklen)

    def _gen_behav_data(self, labels, runno=1, blocklen=4):
        nconds = len(labels)
        if nconds % 2 == 1:  # guess you've got a fixation
            nconds -= 1
        paras = exp.make_para(n=nconds)
        para = paras[runno%nconds].tolist()
        conds = np.array(para[:-1] * 3 + [0])
        conds = np.repeat(conds, blocklen)

        df = pandas.DataFrame({'subjid': self.subjid,
                               'runtype': 'main',
                               'runno': runno+1,
                               'cond':conds,
                               'dur': [2.]*len(conds),
                               'condname': np.take(labels,conds)})
        df.to_csv(self.paths['data_behav'] % self.subjid + 'data_%02d_main.csv' % (runno+1))
        return df

    def prepare_roi(self, roi_coords):
        mask = nb.load('slice_red.nii')
        mask_data = get_data_world(mask)
        mask_data[:,:,:] = 0
        mask_data[roi_coords] = 1
        #mask_data = mask_data.squeeze()
        #mask_data[:,:,:] = mask_data[::-1]
        nb.save(mask, self.paths['rois'] % self.subjid + 'rh_LO.nii')
        #r = nb.load(self.paths['rois'] % self.subjid + 'roi.nii')
        #d = r.get_data()
        #print np.sum(d>0)

    #def prepare_vol(self, paths, nmeasr):
        ## apply a weight for each stimulus condition
        ## fixation gets a zero weight
        #weights_full = (conds>0) * np.take(weights, conds)
        #nmeasr = len(conds)*blocklen
        #coords = np.s_[50:60, 55:60, 14:18, 0:nmeasr]  # an ROI of a shape of a box + time
        ## concat measurements for a single timepoint

    def _gen_fmri_data(self, df, weights, roi_coords, runno, blocklen):
        nmeasr = len(df)
        # make a time series of the slices
        nim = nb.concat_images(['slice_red.nii']*nmeasr)
        data = nim.get_data()
        ori = nb.io_orientation(nim.get_affine())
        data = nb.apply_orientation(data, ori)

        #import pdb; pdb.set_trace()

        # make an empty ROI
        coords = roi_coords + (np.s_[0:len(df)],)
        #import pdb; pdb.set_trace()
        #roi = np.zeros(data.shape)[coords]
        roi = np.zeros([c.stop - c.start for c in coords])
        #print roi.shape

        # compute response in a single voxel
        resp = self.hrfs(weights, blocklen) #* weights_full
        #plt.plot(resp); plt.show()
        resp = np.tile(resp, roi.shape[:3] + (1,))

        nvox = roi[:,:,:,0].size
        vox_sel = np.zeros(roi.shape)
        pos = [0]*(nvox-1) + [10] # np.random.random(nvox)
        cov = np.eye(nvox)
        for cond in np.unique(df.cond):
            np.random.shuffle(pos)
            #cov = np.diag(pos)
            #cov = (cov +cov.T) / 2  # make symmetric
            # "1" is here only to prevent warnings from Numpy 1.8
            sel = np.random.multivariate_normal(pos, cov, 1)
            sel[sel<0] = 0
            sel = sel.reshape(roi.shape[:3])
            inds = np.arange(len(df))
            inds = inds[np.array(df.cond==cond)]
            for ind in inds:
                vox_sel[:,:,:,ind] = sel

        #vox_sel = np.dstack(vox_sels)
        #import pdb; pdb.set_trace()

        #vox_sel = np.random.random(roi.shape[:3])  # how selective the voxel is
        #vox_sel = np.tile(vox_sel, (roi.shape[-1], 1,1,1))
        #vox_sel = vox_sel.transpose((1,2,3,0))
        roi = 10 * resp * vox_sel * np.random.random(roi.shape) + 1000
        #plt.imshow(roi[:,0,0,:]);plt.show()
        #import pdb; pdb.set_trace()
        data[coords] = roi

        nb.save(nim, self.paths['data_fmri'] % self.subjid + 'swafunc_%02d_main.nii' % (runno+1))  # export this to a NIfTI file

    def hrfs(self, weights, blocklen):
        hrf = self.spm_hrf_compat(np.arange(0,24,2), peak_delay=10, peak_disp=2)
        hrf_full = np.zeros(len(weights))
        hrf_full[:len(hrf)] = hrf
        #plt.plot(hrf); plt.show()
        #import pdb; pdb.set_trace()
        full_resp = np.zeros(len(weights))
        #full_resp[:len(hrf)] = hrf

        for offset in range(0, len(weights), blocklen):
            #toend = len(full_resp) - off
            full_resp += np.roll(hrf_full, offset) * weights[offset]
        return full_resp

    #def run(self):


        #for runno in range(nruns):
            #self.get_behav_df()
            #self._gen_fmri_data(runno)

    def spm_hrf_compat(self, t,
                       peak_delay=6,
                       under_delay=16,
                       peak_disp=1,
                       under_disp=1,
                       p_u_ratio = 6,
                       normalize=True,
                      ):
        """ SPM HRF function from sum of two gamma PDFs

        This function is designed to be partially compatible with SPMs `spm_hrf.m`
        function.

        The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
        `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
        and dispersion `under_disp`, and divided by the `p_u_ratio`).

        From: hrf_estimation package (https://github.com/fabianp/hrf_estimation/blob/master/hrf_estimation/hrf.py)
        which used nipy.modalities.fmri.hemodynamic_models
        https://github.com/nipy/nipy/blob/master/nipy/modalities/fmri/hemodynamic_models.py

        Parameters
        ----------
        t : array-like
            vector of times at which to sample HRF
        peak_delay : float, optional
            delay of peak
        peak_disp : float, optional
            width (dispersion) of peak
        under_delay : float, optional
            delay of undershoot
        under_disp : float, optional
            width (dispersion) of undershoot
        p_u_ratio : float, optional
            peak to undershoot ratio.  Undershoot divided by this value before
            subtracting from peak.
        normalize : {True, False}, optional
            If True, divide HRF values by their sum before returning.  SPM does this
            by default.

        Returns
        -------
        hrf : array
            vector length ``len(t)`` of samples from HRF at times `t`

        Notes
        -----
        See ``spm_hrf.m`` in the SPM distribution.
        """
        if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
                if v <= 0]):
            raise ValueError("delays and dispersions must be > 0")
        # gamma.pdf only defined for t > 0
        hrf = np.zeros(t.shape, dtype=np.float)
        pos_t = t[t > 0]
        peak = scipy.stats.gamma.pdf(pos_t,
                             peak_delay / peak_disp,
                             loc=0,
                             scale=peak_disp)
        undershoot = scipy.stats.gamma.pdf(pos_t,
                                   under_delay / under_disp,
                                   loc=0,
                                   scale=under_disp)
        hrf[t > 0] = peak - undershoot / p_u_ratio
        if not normalize:
            return hrf
        return hrf / np.max(hrf)

