#!/usr/bin/env python
'''
A module containing data-model validation tools.
'''

import unittest

######################### METRICS #########################
def bias(o, m):
    '''
    For paired data arrays o (observed) and m (model), calculate the
    bias or mean systematic error.  See Jolliffe and Stephenson, Chapter 5, 
    page 99 for details.  

    A positive bias indicates that the model is consistently overpredicting
    the observations; a negative value indicates underprediction.
    '''

    from numpy import nansum

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    return nansum( m - o )/m.size
    
def mse(o, m):
    '''
    For paired data arrays o (observed) and m (model), calculate the 
    mean squared error.  
    It is a popular metric and can be found easily in review literature.

    Both arguments must be numpy arrays of the same shape with the
    same number of values.  

    '''

    from numpy import nansum, sqrt

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    mse = nansum((o-m)**2.0)/o.size
    
    return mse

def rmse(o, m):
    '''
    For paired data arrays o (observed) and m (model), calculate the 
    root mean squared error (the square root of MSE.)  The units of the
    returned value is the same as the input units.
    It is a popular metric and can be found easily in review literature.

    Both arguments must be numpy arrays of the same shape with the
    same number of values.  

    '''

    from numpy import sqrt

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    rmse = sqrt(mse(o, m))
    
    return rmse

def nrmse(o, m, factor=None):
    '''
    For paired data arrays *o* (observed) and *m* (model), calculate the
    normalized root-mean-squared error.  This is the rmse value divided by
    the range of the data values.  This value is the more widely accepted
    version of nRMSE.

    Alternatively, the normalization factor can be overrided by setting the
    *factor* keyword, which defaults to **None** (and, therefore, the RMSE
    is normalized to the range of the observations).
    '''

    from numpy import nanmin, nanmax
    
    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'
    
    if not factor:
        factor = (nanmax(o)-nanmin(o))

    return rmse(o, m)/factor

def nrmse_old(o, m):           # 
    '''
    For paired data arrays o (observed) and m (model), calculate the 
    normalised root-mean-squared error.  The resulting value is zero for
    a perfect prediction and infinity for an infinitely bad prediction.
    A score of 1 means that the model has the same predictive power
    as a persistance forecast with an amplitude equal to the mean of the
    observations.

    Both arguments must be numpy arrays of the same shape with the
    same number of values.  

    This value is described in detail by Welling and Ridley, 2010.
    '''

    from numpy import sum, sqrt

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    nrmse = sqrt( sum((o-m)**2.0) / sum(o**2.0) )
    
    return nrmse

def p_corr(o, m):
    '''
    For paired time series vectors o (observed) and m (model), calculate
    the Pearson correlation coefficient.  Note that this function is
    just a convenience wrapper for the Scipy function that does the same.

    Pearson correlation coefficient is a well documented value that is 
    easily found in the literature.  A value of zero implies no correlation
    between the data and the model.  A value of [-]1 implies perfect
    [anti-]correlation.
    '''
    
    from scipy.stats import pearsonr

    # Check input values.
    assert o.size == m.size,   'Input arrays must have same size!'
    assert (len(o.shape) == 1) and (len(m.shape) == 1), \
        'Input arrays must be vectors!'

    r = pearsonr(o, m)
    
    return r[0]

def predicteff(o, m):
    '''
    For paired time series vectors o (observed) and m (model), calculate
    the prediction efficiency.  This metric is a measure of skill and is
    defined as 1 - (MSE/theta**2) where MSE is the Mean Square Error
    and theta**2 is the variance of the observations.  A value of 1
    indicates a perfect forecast.
    
    For more information, see
    http://www.swpc.noaa.gov/forecast_verification/Glossary.html#skill
    '''

    from scipy.ndimage.measurements import variance

    # Check input values.
    assert o.size == m.size,   'Input arrays must have same size!'
    assert (len(o.shape) == 1) and (len(m.shape) == 1), \
        'Input arrays must be vectors!'

    var  = variance(o)
    peff = 1.0 - mse(o,m)/var 
    
    return peff

def pairtimeseries_linear(time1, data, time2, **kwargs):
    '''
    Use linear interpolation to pair two timeseries of data.  Data set 1
    (data) with time t1 will be interpolated to match time set 2 (t2). 
    The returned values, d3, will be data set 1 at time 2.
    No extrapolation will be done; t2's boundaries should encompass those of
    t1.

    This function will correctly handle masked functions such that masked
    values will not be considered.

    **kwargs** will be handed to scipy.interpolate.interp1d
    A common option is to set fill_value='extrapolate' to prevent 
    bounds errors.
    '''

    from numpy import bool_
    from numpy.ma import MaskedArray
    from scipy.interpolate import interp1d
    from matplotlib.dates import date2num, num2date

    # Dates to floats:
    t1=date2num(time1); t2=date2num(time2)

    # Remove masked values (if given):
    if type(data) == MaskedArray:
        if type(data.mask) != bool_:
            d =data[~data.mask]
            t1=t1[~data.mask]
        else:
            d=data
    else:
        d=data
    func=interp1d(t1, d, **kwargs)
    return func(t2)

class BinaryEventTable(object):
    '''
    For two unpaired timeseries, observations *Obs* with datetimes *tObs*,
    and model values *Mod* with datetimes *tMod*, create a binary event
    table using timewindow *window* (a datetime.timedelta object) using value
    threshold *cutoff*.  Such tables are powerful for creating validation
    metrics for predictive models.

    Time windows are handled so that the start of the window is inclusive
    while the end of the window is exclusive (i.e., [winStart, winStop) ).
    The start and stop times of windows are not relative to the start/stop
    times of the data but relative to universal time.  For example, if your data
    starts at 5:24UT but a 20 minute time window is selected, the first window
    will start at 5:20, the second at 5:40, etc. until every data point
    (from both model and observations) lies in a window.  If any time window
    is devoid of model and/or observation data, it is ignored.

    This class is fairly robust in that it can handle several non-standard
    situations.  Data gaps spanning a width greater than *window* are 
    dropped from the calculation altogether (i.e., the number of windows
    evaluated is reduced by one).  If either *Mod* and *Obs* are masked 
    arrays, masked values are removed.
    '''

    from numpy import nan
    
    def __repr__(self):
        return 'Binary Event Table with {:.0f} entries'.format(self['n'])

    def __str__(self):
        return '{:.0f} hits, {:.0f} misses,'.format(self['a'], self['c']) + \
            ' {:.0f} false positives, {:.0f} true negatives.'.format(self['b'], 
                                                                     self['d'])

    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
         self.table[key] = value

    def __iadd__(self, table):

        from numpy import append
        
        # Only add to similar objects:
        if type(table) != type(self):
            raise TypeError(
                "unsupported operand type(s) for +: {} and {}".format(
                    type(self), type(table)))

        # Only add if window and cutoff are identical.
        if (self.window != table.window) or (self.threshold !=table.threshold):
            raise ValueError("Threshold and window must be equivalent")

        # Ensure no overlapping times: DISABLED.
        #if (self.start<table.end and self.start>=table.start) or \
        #   (self.end>table.start and self.end<=table.end):
        #    raise ValueError(
        #        "Cannot add two temporally overlapping data sets in place.")
        
        # Combine observations, predictions, and times:
        self.Obs  = append(self.Obs,  table.Obs)
        self.Mod  = append(self.Mod,  table.Mod)
        self.tObs = append(self.tObs, table.tObs)
        self.tMod = append(self.tMod, table.tMod)

        # Combine timings:
        self.nWindow += table.nWindow
        self.start = min(self.start, table.start)
        self.end   = min(self.end,   table.end)

        # Combine hits/misses/etc.
        self['hit']    += table['hit']
        self['miss']   += table['miss']
        self['falseP'] += table['falseP']
        self['trueN']  += table['trueN']

        self['n'] += table['n']
        
        # Update letter-designated values:
        self['a'], self['b'] = self['hit'],  self['falseP']
        self['c'], self['d'] = self['miss'], self['trueN']
        
        return self
        
    def __init__(self, tObs, Obs, tMod, Mod, cutoff, window):
        '''
        Build binary event table from scratch.
        '''

        from datetime import timedelta
        
        from numpy import array
        from numpy.ma.core import MaskedArray, bool_
        from numpy import min, max, ceil, where, zeros, logical_not
        from matplotlib.dates import date2num, num2date

        # If window not a time delta, assume it is seconds.
        if type(window) != timedelta: window=timedelta(seconds=window)
        
        # If list type data, convert to numpy arrays:
        if type(tObs)==list: tObs=array(tObs)
        if type(tMod)==list: tMod=array(tMod)
        if type(Obs) ==list: Obs =array(Obs)
        if type(Mod) ==list: Mod =array(Mod)

        # If handed masked arrays, collapse them to remove bad data.
        if type(Obs) == MaskedArray:
            if type(Obs.mask)!=bool_:
                mask=logical_not(Obs.mask)
                Obs = Obs.compressed()
                tObs= tObs[mask]
        if type(Mod) == MaskedArray:
            if type(Mod.mask)!=bool_:
                mask=logical_not(Mod.mask)
                Mod = Mod.compressed()
                tMod= tMod[mask]

        # Using the start and stop time of the file, obtain the start and stop
        # times of our analysis (time rounded up/down according to *window*).
        start = date2num(min([tObs.min(), tMod.min()]) )
        end   = date2num(max([tObs.max(), tMod.max()]) )
        dT    = window.total_seconds()

        # Offsets for start and end times to make time windows align
        # correctly.  Round to nearest second to avoid precision issues.
        start_offset = timedelta(seconds = round(start*24*3600)%dT)
        end_offset   = timedelta(seconds = round(start*24*3600)%dT) + window

        # Generate start and stop time.
        winstart = (num2date(start) - start_offset).replace(tzinfo=None)
        winend   = (num2date(end)   + end_offset  ).replace(tzinfo=None)

        # With start and stop times, create window information.
        nWindow = int(ceil( (date2num(winend)-date2num(winstart)) \
                            / (dT/3600./24.) ))
        nTime    = nWindow+1

        
        # Create boundaries of time windows.
        time = [winstart+i*window for i in range(int(nTime))]
        
        # Store these values in the object.
        self.Obs, self.tObs = Obs, tObs
        self.Mod, self.tMod = Mod, tMod
        self.window = window
        self.start, self.end = winstart, winend
        self.nWindow = nWindow
        self.threshold = cutoff

        # Convert data to binary format: +1 for above or equal to threshold,
        # -1 for below threshold.  
        # A "hit" is 2*Obs+Model=3.
        # A "true negative" is 2*Obs+Model=-3.
        # A "miss" is 2*Obs+Model=1.
        # A "false positive" is 2*Obs+Model=-1
        Obs = where(Obs >= cutoff, 1,-1)
        Mod = where(Mod >= cutoff, 1,-1)
        
        # Create an empty table and a dictionary of keys:
        table = {'hit':0., 'miss':0., 'falseP':0., 'trueN':0., 'n':0.}
        result={3:'hit', 1:'miss', -1:'falseP', -3:'trueN'}

        # Perform binary analysis.
        for i in range(nWindow):
            #print('Searching from {} to {}'.format(time[i], time[i+1]))
            # Get points inside window:
            subObs = Obs[(tObs>=time[i]) & (tObs<time[i+1])]
            subMod = Mod[(tMod>=time[i]) & (tMod<time[i+1])]
            
            # No points?  No metric!
            if not subObs.size*subMod.size:
                #print('NO RESULT from {} to {}'.format(time[i], time[i+1]))
                continue

            # Determine contigency result and increment it.
            val = 2*int(subObs.max()) + int(subMod.max())
            table[result[val]] += 1
            table['n'] += 1
            #print('{} from {} to {}'.format(result[val],time[i], time[i+1]))
            
        # For convenience, use the definitions from Jolliffe and Stephenson.
        table['a'], table['b'] = table['hit'],  table['falseP']
        table['c'], table['d'] = table['miss'], table['trueN']

        # Place results into object.
        self.table=table
        self.n    =table['n']

    def latex_table(self, value='values', units=''):
        '''
        Return a string that, if printed into a LaTeX source file, 
        would yield the results in tabular format.

        The kwarg *value* should be set to the variable being investigated,
        e.g., tornado occurence, 40$keV$ proton flux, etc.

        If kwarg *units* is provided, add units to the threshold value
        in the table caption.
        '''
        
        table = r'''
        \begin{table}[ht]
        \centering
        \begin{tabular}{r|c c}
        \hline \hline
        \multicolumn{1}{c|}{Event} & \multicolumn{2}{c}{Event}\\
        \multicolumn{1}{c|}{Forecasted?} & \multicolumn{2}{c}{Observed?}\\
        & Yes & No\\
        \hline
        '''
        
        table +='''
        Yes   & {a:.0f}  &  {b:.0f} \\\\
        No    & {c:.0f}  &  {d:.0f} \\\\
        \\hline
        Total & {n:.0f}\\\\
            '''.format(**self.table)

        table+=r'''
        \end{tabular}
        \caption{
        '''

        table+='''Binary event table for predicted {0} using a threshold of
        {1.threshold:G}{2}.  Under these conditions, the model yielded a
        Hit Rate of {3:05.3f}, a False Alarm Rate of {4:05.3f}, and a 
        Heidke Skill Score of {5:05.3f}.'''.format(
            value, self, units, self.calc_HR(), 
            self.calc_FARate(), self.calc_heidke())

        table+='''}
        \end{table}
        '''

        return table

    def add_timeseries_plot(self, target=None, loc=111, xlim=None, ylim=None, 
                            doLog=False):
        '''
        
        '''
        pass


    def calc_PC(self):
        '''
        Calculate and return Proportion Correct, or, the proportion of
        correct forecasts, defined as "hits" plus "true negatives" divided
        by total number of occurrences.
        '''
        return (self['a']+self['d'])/self['n']

    def calc_HR(self):
        '''
        Calculate and return Hit Rate, or the proportion of occurrences that
        were correctly forecast.  This is also known as probability of 
        detection, and is the "hits" divided by "hits"+"misses".
        '''
        from numpy import nan
        
        if self['a']+self['c'] > 0:
            return self['a']/(self['a']+self['c'])
        else:
            return nan

    def calc_FARate(self):
        '''
        Calculate False Alarm Rate, also known as probability of false
        detection, definded as "False Positives" divided by "False Positives"
        plus "True Negatives".  It is the proportion of non-events
        incorrectly forecasted as events.
        '''

        from numpy import nan
        
        if (self['b']+self['d']) > 0:
            return self['b']/(self['b']+self['d'])
        else:
            return nan

    def calc_PCE(self):
        '''
        Calculate and return the Proportion Correct for a random 
        forecast.  This value is a baseline, unskilled PC and is the 
        basis for the Heidke Skill Score calculation.
        '''

        return \
            (self['a']+self['c'])/self['n'] * \
            (self['a']+self['b'])/self['n'] + \
            (self['b']+self['d'])/self['n'] * \
            (self['c']+self['d'])/self['n']


    def calc_heidke(self):
        '''
        Calculate and return the Heidke Skill Score, a measure of proportion 
        correct adjusted for the number of forecasts that would be correct by
        random chance (i.e., in the absence of skill.)  The value ranges
        from [-1,1], where zero is no skill (the model performs as well as
        one that relies on random chance), 1 is a perfect forecast.  
        A negative value, or negative "skill", is not worse than a score
        of zero; rather, it implies positive skill if the binary event
        categories were rearranged.
        '''
        from numpy import nan
        
        PC = self.calc_PC()
        E  = self.calc_PCE()

        if (1-E) != 0:
            return (PC-E)/(1.-E)
        else:
            return nan


###############################################################################
#TEST SUITE #
###############################################################################
class TestBinaryTable(unittest.TestCase):
    '''
    Test building binary event tables, combining them with others, and 
    calculating final metrics.
    '''
    import datetime as dt
    
    # Create some time vectors:

    start = dt.datetime(2000, 5, 2, 23, 56, 0)
    # UPDATED: LIMITED SCOPE FOR LIST COMP & EXEC WITHIN CLASS DEF
    #t1 = [start+dt.timedelta(minutes=i) for i in range(8)]
    #t2 = [t + dt.timedelta(seconds=10)  for t in t1]
    #t3 = [t + dt.timedelta(minutes=10)for i, t in enumerate(t1)]
    t1, t2, t3 = [], [], []
    for i in range(8):
        t1.append(start+dt.timedelta(minutes=i))
        t2.append(t1[-1]+ dt.timedelta(seconds=10))
        t3.append(t1[-1]+ dt.timedelta(minutes=10))

    # Artificial data vectors: every category (hit, miss, etc.) is used.
    # This will give us a final binary table where every value is "2".
    d1 = [0, 1, 0, 1, 0, 1, 0, 1]
    d2 = [0, 0, 1, 1, 0, 0, 1, 1]

    # Alternative "observation" that yields better scores:
    d3 = [0, 1, 0, 1, 0, 0, 0, 0]

    # Cutoff and time window for tables:
    cutoff=.5
    window=dt.timedelta(minutes=1)
    
    def testTable(self):
        '''
        Build a table, test results.
        '''

        tab = BinaryEventTable(self.t1, self.d1, self.t2, self.d2,
                               self.cutoff, self.window)

        # Test values in table:
        for x in 'ac':
            self.assertEqual(2, tab[x])
        
        # Test metric calculation:
        self.assertEqual( 0.5, tab.calc_PC() )
        self.assertEqual( 0.5,  tab.calc_HR() )
        self.assertEqual( 0.5,  tab.calc_FARate() )
        self.assertEqual( 0.5,  tab.calc_PCE() )
        self.assertEqual( 0.0,  tab.calc_heidke() )

    def testTableExactTime(self):
        '''
        Build a table where the observed and model times are identical.
        Test results.  This was creating issues in some applications.
        This error was caused by precision limitations when converting
        datetimes to floats using Matplotlib's dates.date2num function.
        Very small (1E-5s) errors was causing a shift in the time windows
        and missing some data-model comparisons.
        '''

        tab = BinaryEventTable(self.t3, self.d1, self.t3, self.d2,
                               self.cutoff, self.window)

        # Test values in table:
        for x in 'abcd':
            self.assertEqual(2, tab[x])
        
        # Test metric calculation:
        self.assertEqual( 0.5, tab.calc_PC() )
        self.assertEqual( 0.5, tab.calc_HR() )
        self.assertEqual( 0.5, tab.calc_FARate() )
        self.assertEqual( 0.5, tab.calc_PCE() )
        self.assertEqual( 0.0, tab.calc_heidke() )

    def testTableMetric(self):
        '''
        Test metrics when skill is nonzero.
        '''
        tab = BinaryEventTable(self.t1, self.d1, self.t2, self.d3,
                               self.cutoff, self.window)

        # Test values in table:
        for x in 'ac':
            self.assertEqual(2, tab[x])
        self.assertEqual(0, tab['b'])
        self.assertEqual(4, tab['d'])

        # Test metric calculation:
        self.assertEqual( 0.75, tab.calc_PC() )
        self.assertEqual( 0.5,  tab.calc_HR() )
        self.assertEqual( 0.0,  tab.calc_FARate() )
        self.assertEqual( 0.5,  tab.calc_PCE() )
        self.assertEqual( 0.5,  tab.calc_heidke() )

    def testTableAddInPlace(self):
        '''
        Test our ability to add to existing table in-place.
        '''

        # Two individual tables.  These are tested individually above.
        tab1=BinaryEventTable(self.t1, self.d1, self.t2, self.d2,
                              self.cutoff, self.window)
        tab2=BinaryEventTable(self.t3, self.d1, self.t3, self.d3,
                              self.cutoff, self.window)

        # Combined table (manual):
        tab3=BinaryEventTable(self.t1+self.t3, 2*self.d1,
                              self.t2+self.t3, self.d2+self.d3,
                              self.cutoff, self.window)

        # Combined table (via sum):
        tab1+=tab2
        
        # Table 3 should be equivalent to tab1+tab2.
        for x in 'abcd':
            self.assertEqual(tab1[x], tab3[x])
        
        self.assertEqual(tab1.calc_PC(),     tab3.calc_PC())
        self.assertEqual(tab1.calc_HR(),     tab3.calc_HR())
        self.assertEqual(tab1.calc_FARate(), tab3.calc_FARate())
        self.assertEqual(tab1.calc_PCE(),    tab3.calc_PCE())
        self.assertEqual(tab1.calc_heidke(), tab3.calc_heidke())

    #def testTableAddOverlap(self):
    #    '''
    #    Ensure that adding temporally overlapping tables raises exception.
    #    '''
    #
    #    tab1=BinaryEventTable(self.t1, self.d1, self.t2, self.d2,
    #                          self.cutoff, self.window)
    #    tab2=BinaryEventTable(self.t1, self.d1, self.t2, self.d3,
    #                          self.cutoff, self.window)
    #    
    #    self.assertRaises(ValueError, tab1.__iadd__, tab2)
        
if __name__=='__main__':
    print( 10*'=' + 'TESTING VALIDATION PACKAGE' + 10*'=')
    unittest.main()
