#!/usr/bin/env python
'''
A module for obtaining (eventually) and parsing data from 
http://omniweb.gsfc.nasa.gov/
'''

import re
import datetime as dt
import numpy as np
from spacepy.datamodel import dmarray

def read_ascii(filename):
    '''
    Load data into dicti3onary.
    '''

    if filename[-4:] == '.lst': # These if elif statements get both the .lst and .fmt files for the given filename.
        datafile   = filename
        formatfile = filename[:-4]+'.fmt'
    elif filename[-4:] == '.fmt':
        formatfile = filename
        datafile   = filename[:-4]+'.lst'
    else:
        formatfile = filename+'.fmt'
        datafile   = filename+'.lst'

    try:
        fmt = open(formatfile, 'r')
    except:
        fmt = False

    info = {} # a dictionary
    var  = [] # a list of variables
    flag = [] # a list of flags
        
    if fmt: # if fmt is a TextIOWrapper (the above try / except was successful)
        raw = fmt.readlines()
        fmt.close() # read in the data to raw and close the reader.
        # Determine time resolution: hour or minute:
        tres = 'Hour' # asign hour and then check each line for if it is actually a minute
        for line in raw:
            if 'Minute' in line: tres='Minute' # could this be made efficent by checking a smaller range of raw?
        
        # Skip header.
        while True:
            raw.pop(0)
            if tres in raw[0]: break # pop all lines until the time resolutin line is reached.
        last = raw.pop(0) # gets rid of all lines up to and inculding the time resolution line of the .fmt file.

        # Parse rest of header:
        for l in raw: # l is short for line.
            x=re.search('\d+\s+(.+?),\s*(\S+)', l) # the pattern being searched for: \d Any decimal digit,+ matches 1
            # or more (greedy, match as many repetitions as possible) of previous RE (in this case decimal digits),
            # \s any whitespace character, '.' any character except a newline, ? matches 0 or 1 (greedy) of preceding
            # RE, * matches 0 or more (greedy), \S matches any non-whitespace character. Returns object if match found,
            # or none if match wasn't found.
            if x: # if a match of the above string was found in the line l.
                info[x.groups()[0]] = x.groups()[-1] # assigns the variable name as the key and the unit as the value stored with it in the dictionary
                var.append(x.groups()[0]) # appends the variable to the list of variables
                fmt_code = l.split()[-1] # gets the fmt_code from the last entry of the split line.
                if 'F' in fmt_code: # not sure what F means, but I've seen either F or I in the fmt_code spot along with some numbers.
                    i1, i2 = int(fmt_code[1]), -1*int(fmt_code[-1]) #takes the numbers in the F code and does some math to them.
                    flag.append(10**(i1+i2-2) - 10**(i2))# this flag command appends a value of 9999.99 for a F8.2 fmt_code
                elif 'I' in fmt_code:
                    i1 = int(fmt_code[-1])-1
                    flag.append(10**i1 -1) # appends a int with as many 9s as one less than the number in the I code.
            else:
                break
    # the flag then is a list of numbers, whose values can be used to figure out the fmt_code used in that variable.
    # Read in data.
    raw = open(datafile, 'r').readlines()
    nLines = len(raw)

    # Create container.
    data = {}
    data['time'] = np.zeros(nLines, dtype=object)
    for k in info:
        data[k] = np.zeros(nLines)
        #data[k] = dmarray(np.zeros(nLines), attrs={'units':info[k]})

    # Now save data.
    t_index = 3 + (tres == 'Minute')*1 # t_index == 3 if time resolution is Hour, or 4 if tres is Minute.
    for i, l in enumerate(raw): # i is the index of the raw list that line l is.
        parts = l.split() # l is again the lines of raw, being the lines of the datafile (.lst)
        doy = dt.timedelta(days=int(parts[1])-1)# gives int one value less than the day of year recorded in the line.
        minute = int(float(parts[3]))*(tres == 'Minute') # if time resolution is mintues gives the minute value to minute.
        # if time resolution is Hour, sets minute to 0.
        data['time'][i]=dt.datetime(int(parts[0]), 1, 1, int(parts[2]),
                                minute, 0) + doy # lines doy to this one change date format from YYYY/DOY to YYYY/MM/DD
        for j, k in enumerate(var):
            data[k][i] = parts[j+t_index] # assigns the variable info from the line to the apporaite key in the dict and
            # index of the array each key has. t_index is the skip the time data in the line now contained within
            # data['time']

    # Mask out bad data.
    for i, k in enumerate(var):
        data[k] = np.ma.masked_greater_equal(data[k], flag[i]) # masked arrays hide invalid or missing data for each
        # variable in data go through their entire array of values and if any has a value equal to or greater than the
        # flag for the variable (as determined by the fmt_code) mask it, the value will be ignored if any operations
        # are preformed on the array such as mean, median, mode, etc.

            
    return data


def omni_to_swmf(infile, pair, outfile=None):
    '''
    Given an input omni file, *infile*, read its contents and convert to an
    SWMF input file.  The omni input file **must** have the following
    variables: BX, BY, BZ (all GSM); Proton Density; Speed; Temperature.

    If the kwarg *outfile* is set, the resulting conversion will be written
    to the file specified by the kwarg. 

    Both the omni data and the ImfInput object are returned to the caller.

    This function will attempt to linearly interpolate over bad data values.
    If the first or last line of the data contains bad data flags, this will
    fail.

    GSE coordinates V_Y and V_Z will be rotated to GSM coordinates.

    Finally, always inspect your results.
    '''

    from spacepy.pybats import ImfInput
    from spacepy.time import Ticktock
    from spacepy.coordinates import Coords
    #from validate.py import pairtimeseries_linear as pair

    # Variable name mapping:
    omVar = ['BX', 'BY', 'BZ', 'Vx Velocity', 'Vy Velocity', 'Vz Velocity', 'Proton Density', 'Temperature']
    swVar = ['bx', 'by', 'bz', 'ux', 'uy', 'uz', 'rho', 'temp']
    # omVar = ['BX', 'BY', 'BZ', 'Speed', 'Proton Density', 'Temperature']

    # Load observations:
    omdat = read_ascii(infile)
    nPts = omdat['time'].size

    # Create empty ImfInput object:
    swmf = ImfInput(npoints=nPts, load=False, filename=outfile)

    # Convert data and load into ImfInput object:
    swmf['time'] = omdat['time']
    npts = swmf['time'].size
    for vo, vs in zip(omVar, swVar):
        if vo in omdat:
            swmf[vs] = pair(omdat['time'], omdat[vo], omdat['time']) # interpolates data from omni to swmf.
        else:
            print('WARNING: Did not find {} in Omni file.'.format(vo))
            swmf[vs] = np.zeros(npts)

    # Do rotation of Vy, Vz GSE->GSM:
    rot = Coords(np.array([swmf['ux'], swmf['uy'], swmf['uz']]).transpose(),
                 'GSE', 'car', ['Re', 'Re', 'Re'],
                 Ticktock(swmf['time'])).convert('GSM', 'car')

    # Copy rotated values into original structure:
    swmf['ux'] = rot.data[:, 0]
    swmf['uy'] = rot.data[:, 1]
    swmf['uz'] = rot.data[:, 2]

    # Flip sign of velocity:
    if swmf['ux'].min() > 0: swmf['ux'] *= -1

    if outfile: swmf.write()

    return omdat, swmf


def qindent_to_swmf(infiles, outfile):
    '''
    Given either a single input file name or a list of multiple file names
    corresponding to JSON-headed ASCII Qin-Denton OMNI data (obtainable from
    http://www.rbsp-ect.lanl.gov/data_pub/QinDenton), this function will
    produce a SWMF IMF input file.

    If the kwarg *outfile* is set, the resulting conversion will be written
    to the file specified by the kwarg. 

    Both the omni data and the ImfInput object are returned to the caller.

    This function will attempt to linearly interpolate over bad data values.
    If the first or last line of the data contains bad data flags, this will
    fail.

    There are many limitations to this approach.  First, V_Y and V_Z are
    set to zero as this information is not given in the Qin-Denton files.
    The same is true for IMF Bx.  Finally, and most importantly, solar
    wind temperature is not present, making this whole thing really bad.

    A limitation of this approach is that solar V_Y, V_Z are neglected.
    Additionally, there is no B_X information, so these values are set to
    zero as well.  Use with care!

    Finally, always inspect your results.
    '''

    from spacepy.omni import readJSONheadedASCII as readit
    from spacepy.pybats import ImfInput
    # from validate import pairtimeseries_linear as pair

    raise NotImplemented('This function is incomplete.')

    return False

    # Variable names:
    omVar = ['ByIMF', 'BzIMF', 'Vsw', 'Den_P', 'Temperature']
    swVar = ['by', 'bz', 'ux', 'rho', 'temp']

    # Read omni files:
    omdat = readit(infiles)

    # Create empty ImfInput object:
    nPts = omdat['time'].size
    swmf = ImfInput(npoints=nPts, load=False, filename=outfile)

    return omdat, swmf


