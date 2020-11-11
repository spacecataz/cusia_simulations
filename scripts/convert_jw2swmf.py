#!/usr/bin/env python

'''
Convert a set of propagated solar wind files as produced by James Weygand
into SWMF input files.

All bad data gaps will be filled via linear interpolation.

This script requires Spacepy to build the SWMF input file.

The input files should have corresponding time values.
'''

from argparse import ArgumentParser

parser = ArgumentParser(description=__doc__)
parser.add_argument('filename', type=str, help='Either a plamsa or mag ' +
                    'file for conversion.  The matching mag/plasma file '+
                    'will be found automatically.')

# Get args from caller, collect arguments into a convenient object:
args = parser.parse_args()

# Continue with imports:
import numpy as np
import datetime as dt

from spacepy.pybats import ImfInput

def lin_interp(time, x):
    '''
    A wrapper for scipy.interpolate.interp1d to interpolate over masked values
    in a numpy masked array.

    The first and last values are set to be the first and last (respectively)
    non-masked values.
    '''
    
    from scipy.interpolate import interp1d
    from matplotlib.dates import date2num

    # If there are no masked values, there's no work to do.
    if type(x.mask) == np.bool_:
        return(x)
    
    time = date2num(time)
    
    # Set first and last items if they are masked to first (last) not
    # masked item:
    if x.mask[ 0]: x[0] = x[~x.mask][ 0]
    if x.mask[-1]: x[-1]= x[~x.mask][-1]

    func = interp1d(time[~x.mask], x[~x.mask])
    
    # Interpolate linearly:
    return func(time)
    
def read_jwascii(filename, flag=1E34):
    '''
    Read a JW-generated propagated solar wind data file and load to 
    numpy arrays.  

    A dictionary of variable name-numpy masked array pairs is returned.

    Parameters
    ----------
    filename : string
        Name of file to open.

    Other Parameters
    ----------------
    flag : float
        Bad data flag; values will be filtered from final result.


    Returns
    --------
    data : dict
        Dict of variable names and associated masked arrays of values.

    '''

    # Slurp data contents:
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Determine if this is a plasma or magnetic field file:
    nVar = len(lines[0].split())

    # Set variable names appropriately:
    if nVar>12:  #plasma data
        namevar = ['temp','rho', 'ux','uy','uz','x','y','z']
    else:
        namevar = ['bx','by','bz','x','y','z']

    # Number of time records:
    nTime = len(lines)
        
    # Create output container:
    data = {}
    data['time'] = np.zeros(nTime, dtype='object')
    for v in namevar: data[v] = np.zeros(nTime)

    # Read data:
    for i, l in enumerate(lines):
        var = l.split()
        
        # Get time:
        data['time'][i]=dt.datetime.strptime(' '.join(var[:5]),'%d %m %Y %H %M')

        # Get rest of values:
        for j,v in enumerate(var[6:]):
            data[namevar[j]][i] = v

    # Mask data:
    for v in namevar:
        data[v] = lin_interp(data['time'], np.ma.masked_equal(data[v], flag))
        
            
    # All done:
    return data

if __name__ == '__main__':

    constant = 60.57372293754603 # m_proton*1000^2/2/Boltzman's constant
    
    if 'plasma' in args.filename:
        pfile = args.filename
        mfile = args.filename.replace('plasma','mag')
    elif 'mag' in args.filename:
        pfile = args.filename.replace('mag', 'plasma')
        mfile = args.filename
    else:
        print(f'File name does not match expected format:\n\t{args.filename}')
        raise ValueError("'plasma' or 'mag' not found in file name.")

    print(f'Converting these files:\n\t{pfile}\n\t{mfile}')

    # Open/read files:
    pdata = read_jwascii(pfile)
    mdata = read_jwascii(mfile)

    # Some basic checks that our files line up reasonably well:
    if pdata['time'].size != mdata['time'].size:
        raise ValueError('Files do not have these same number of entries.')
    if pdata['time'][ 0]!=mdata['time'][ 0] or \
       pdata['time'][-1]!=mdata['time'][-1]:
        raise ValueError('Differing start/stop times for plasma/mag files.')
    
    # Convert values for temperature: km/s -> K
    pdata['temp'] = 60.57372293754603 * pdata['temp']**2

    # Build our IMF object:
    print("Building SWMF IMF file...")
    imf = ImfInput(load=False)

    # Copy values over:
    imf['time'] = mdata['time']
    for b in ['bx', 'by', 'bz']:
        imf[b] = mdata[b]
    for v in ['ux', 'uy', 'uz', 'rho', 'temp']:
        imf[v] = pdata[v]

    # Set filename, attributes:
    imf.attrs['file'] = f'imf_{mdata["time"][0]:%Y%m%d}.dat'
    imf.attrs['header'].append("Converted from Weygand propagted files:")
    imf.attrs['header'].append(f"\n\t{pfile}")
    imf.attrs['header'].append(f"\n\t{mfile}")

    # Write out:
    print(f"Saving file as {imf.attrs['file']}")
    imf.write()
