"""
A collection of functions for reading the pvp files by using the already
developped octave functions from PetaVision/mlab/HyPerLCA/ and
Petavision/mlab/utils. 

Design philosophy:
- just get the things from what octave files do into Python
- Make sure you don't run out of memory by keeping the data in both Octave and
  Python

NOTE: For some misterious reason "from oct2py importy octave" works much faster
then if we were to initiate the Oct2Py object and use its call method.
"""

from oct2py import octave

def readpvpfile(filename):
    # start a new octave session
    octave.restart()
    [data, header] = octave.readpvpfile(filename)
    # close that octave session to save on memory
    octave.close()
    return data, header

