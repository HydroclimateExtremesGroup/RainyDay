#==============================================================================
# WELCOME
#==============================================================================


#    Welcome to RainyDay, a framework for coupling gridded precipitation
#    fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
#    Copyright (C) 2017  Daniel Benjamin Wright (danielb.wright@gmail.com)
#

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.#




#==============================================================================
# THIS DOCUMENT CONTAINS VARIOUS FUNCTIONS NEEDED TO RUN RainyDay
#==============================================================================
#
# RainyDay_Py3.py imports this module as "RainyDay" and calls these functions
# as RainyDay.<function>. Every function has a docstring; the "Status" line at
# the end of each says whether RainyDay_Py3.py currently uses it.
#
# Rough organization of this file:
#   1. Small utilities: smoothing, KDE bandwidth, JSON duplicate-key check
#   2. Storm catalog search: find where the watershed-weighted rainfall is
#      largest (catalogFFT_irregular is the active one; others are legacy)
#   3. Transposition / resampling: basin rainfall for each transposed
#      position, with optional rescaling (SSTalt, SSTalt_normalized,
#      SSTalt_singlecell) and kernel-based location sampling (numbakernel*)
#   4. Grid and mask setup: findsubbox, rastermask
#   5. NetCDF input/output: read input rainfall, write/read storm catalogs,
#      write scenario files
#   6. File-list, date and misc. helpers
#
# Conventions used throughout:
#   - Rainfall arrays are ordered (time, lat, lon) and stored south-up, i.e.
#     row 0 is the southernmost latitude.
#   - Grid coordinates are treated as the upper-left corner of each cell.
#   - Transposition positions (x, y) are the column/row index of the
#     upper-left corner of the watershed's bounding rectangle ("trimmask").
#   - Rainfall is a rate in mm/hr. Summing over time and multiplying by
#     timeres/60 gives depth in mm; dividing a mask-weighted sum by
#     mnorm = sum(trimmask) gives a basin average.
#   - -9999 (sometimes -999) is the missing-data / "no storm" flag.
#
#==============================================================================
#%%                                               
import os
import sys
import numpy as np
import scipy as sp
import glob
import re     
import fiona
import copy
from netCDF4 import Dataset, num2date
from rasterio.transform import from_origin
from rasterio.mask import mask
from rasterio.io import MemoryFile
import pandas as pd
from numba import prange,jit
import pyproj
import shapely
from shapely.geometry import shape
import json
import xarray as xr
import linecache
from scipy.signal import correlate
from datetime import datetime



# =============================================================================
# FFT-based catalog creator from Gabriel Perez, added by DBW 10 July 2025, edited by BLF 15 September 2026
# Edits added restriction that storms footprint must fully be in domain (mirrors storm placement changes)
# =============================================================================

def catalogFFT_irregular(temparray, trimmask, valid_anchor):
    """
    Find the location of maximum basin-averaged rainfall in one accumulated
    rainfall field. This is the core search used during storm catalog creation.

    The watershed mask (``trimmask``) is slid over every possible position in
    the transposition domain using FFT/direct cross-correlation, which is
    mathematically the same as looping over every position and computing
    ``nansum(temparray[y:y+h, x:x+w] * trimmask)``, but much faster. Only
    "anchor" positions where the whole watershed footprint lies inside the
    domain (``valid_anchor``) are eligible.

    Added by DBW (July 2025, from Gabriel Perez); edited by BLF (Sept 2026) to
    require that the full watershed footprint fall inside the domain.

    Parameters
    ----------
    temparray : np.ndarray, shape (ny, nx)
        Rainfall field accumulated over the catalog duration (sum of rates;
        NaNs are treated as zero).
    trimmask : np.ndarray, shape (maskheight, maskwidth)
        Watershed mask trimmed to its bounding rectangle. Values are weights
        (0-1 for a fractional mask).
    valid_anchor : np.ndarray of bool, shape (ny-maskheight+1, nx-maskwidth+1)
        True where the upper-left corner of the mask can be placed such that
        the entire watershed footprint is inside the transposition domain.

    Returns
    -------
    rmax : float
        Maximum mask-weighted rainfall sum (not yet normalized by mask area).
    ymax, xmax : int
        Row/column index of the upper-left corner of the mask at the maximum.

    Notes
    -----
    Exits the program if the watershed fits nowhere in the domain. Ties are
    resolved by ``np.argmax`` (first occurrence in row-major order).
    Status: used by RainyDay_Py3.py (catalog creation and duration trimming).
    """
    # Clean NaNs
    temparray_clean = np.nan_to_num(temparray)
    trimmask_clean = np.nan_to_num(trimmask)
    if not valid_anchor.any():
        sys.exit("Watershed fits nowhere inside the domain")

    # Cross-correlation (no flipping of mask)
    result = correlate(temparray_clean, trimmask_clean, mode='valid',method='auto')
    #result = fftconvolve(temparray_clean, trimmask_clean, mode='valid')
    #result = oaconvolve(temparray_clean, trimmask_clean, mode='valid')
    
    # Mask out invalid anchor points (where the watershed footprint does not fit)
    result = np.where(valid_anchor, result, -np.inf)

    # Find max value and its location
    rmax = np.max(result)
    ymax, xmax = np.unravel_index(np.argmax(result), result.shape)
    
    return float(rmax), int(ymax), int(xmax)



def mysmoother(inarray,sigma=[3,3]):
    """
    NaN-aware Gaussian smoothing ("normalized convolution").

    NaNs are set to zero, the field and a matching weight array (1 where valid,
    0 where NaN) are both Gaussian-filtered, and the smoothed field is divided
    by the smoothed weights. This prevents NaNs from spreading and avoids the
    edges being biased toward zero. Cells that were NaN in the input are NaN
    in the output.

    Parameters
    ----------
    inarray : np.ndarray
        Array to smooth (any dimension; typically 2D).
    sigma : list of float, optional
        Gaussian standard deviation (in grid cells) for each dimension of
        ``inarray``. Must have the same length as ``inarray.shape``.

    Returns
    -------
    np.ndarray
        Smoothed array, same shape as ``inarray``.

    Notes
    -----
    Status: used by RainyDay_Py3.py (transposition kernel and rescaling fields).
    """
    if len(sigma)!=len(inarray.shape):
        sys.exit("there seems to be a mismatch between the sigma dimension and the dimension of the array you are trying to smooth")
    V=inarray.copy()
    V[np.isnan(inarray)]=0.
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma)

    W=0.*inarray.copy()+1.
    W[np.isnan(inarray)]=0.
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma)
    outarray=VV/WW
    outarray[np.isnan(inarray)]=np.nan
    return outarray



def my_kde_bandwidth(obj, fac=1):     # this 1.5 choice is completely subjective :(
    #We use Scott's Rule, multiplied by a constant factor
    """
    Bandwidth rule passed to ``scipy.stats.gaussian_kde(bw_method=...)``.

    Returns Scott's rule factor, ``n**(-1/(d+4))``, times a constant ``fac``.
    It is used to build the kernel density estimate of storm-center locations
    that defines the non-uniform transposition probability map.

    Parameters
    ----------
    obj : scipy.stats.gaussian_kde
        The KDE object (supplies ``n`` = number of points, ``d`` = dimensions).
    fac : float, optional
        Multiplier on Scott's rule. Default 1 (the inline comment about 1.5
        refers to an older default).

    Returns
    -------
    float
        Bandwidth factor.

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    return np.power(obj.n, -1./(obj.d+4)) * fac



def find_nearest(array,value):
    """
    Return the index of the element of ``array`` closest to ``value``.

    Parameters
    ----------
    array : np.ndarray
        1D array to search (e.g., the return-period array).
    value : float
        Target value.

    Returns
    -------
    int
        Index of the nearest element (first one in case of ties).

    Notes
    -----
    Status: used by RainyDay_Py3.py (selecting RETURNLEVELS / RETURNTHRESHOLD).
    """
    idx = (np.abs(array-value)).argmin()
    return idx

def convert_3D_2D(geometry):
    """
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D
    Multi/Polygons by dropping the z-coordinate.

    Parameters
    ----------
    geometry : iterable of shapely geometries
        Typically a GeoSeries.

    Returns
    -------
    list of shapely.geometry.Polygon or MultiPolygon

    Notes
    -----
    Only geometries with a z-coordinate are returned; 2D geometries in the
    input are skipped. Iterating ``for ap in p`` over a MultiPolygon uses the
    Shapely 1.x API (Shapely 2 requires ``p.geoms``).
    Status: not currently called.
    """
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = shapely.geometry.Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = shapely.geometry.Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(shapely.geometry.MultiPolygon(new_multi_p))
    return new_geo



# adapted from https://pythonadventures.wordpress.com/2016/03/06/detect-duplicate-keys-in-a-json-file/
def dict_raise_on_duplicates(ordered_pairs):
    """
    ``object_pairs_hook`` for ``json.loads`` that rejects duplicate keys.

    Standard JSON parsing silently keeps the last value when a key appears
    twice, which can hide mistakes in a RainyDay parameter file. This hook
    stops the program instead.

    Parameters
    ----------
    ordered_pairs : list of (key, value) tuples
        Supplied by ``json.loads``.

    Returns
    -------
    dict

    Notes
    -----
    Adapted from https://pythonadventures.wordpress.com/2016/03/06/detect-duplicate-keys-in-a-json-file/
    Status: used by RainyDay_Py3.py when reading the parameter (.json) file.
    """
    d = {}
    for k, v in ordered_pairs:
        if k in d:
           sys.exit("duplicate key: %r" % (k,))
        else:
           d[k] = v
    return d

#==============================================================================
# LOOP TO DO SPATIAL SEARCHING FOR MAXIMUM RAINFALL LOCATION AT EACH TIME STEP
# THIS IS THE CORE OF THE STORM CATALOG CREATION TECHNIQUE
#==============================================================================
    

def catalogAlt(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask):
    """
    Brute-force (pure Python) version of the storm-catalog spatial search for
    rectangular domains. Slides ``trimmask`` over every position and returns
    the location of maximum mask-weighted rainfall.

    Parameters
    ----------
    temparray : np.ndarray, shape (ny, nx)
        Accumulated rainfall field.
    trimmask : np.ndarray, shape (maskheight, maskwidth)
        Trimmed watershed mask.
    xlen, ylen : int
        Number of candidate positions in x and y.
    maskheight, maskwidth : int
        Dimensions of ``trimmask``.
    rainsum : np.ndarray, shape (ylen, xlen)
        Work array; overwritten with the mask-weighted sum at each position.
    domainmask : np.ndarray
        Unused here (kept for a common call signature).

    Returns
    -------
    rmax : float
    ymax, xmax : int

    Notes
    -----
    LEGACY: superseded by ``catalogFFT_irregular``. Status: not currently called.
    """
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i//xlen
        x=i-y*xlen
        #print x,
        rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    return rmax, wheremax[0][0], wheremax[1][0]

def catalogAlt_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask):
    """
    Brute-force version of the storm-catalog spatial search for irregular
    transposition domains. A position is considered only if the mask's
    horizontal and vertical center lines intersect the domain.

    Parameters
    ----------
    See ``catalogAlt``. ``domainmask`` (1 inside the domain, 0 outside) is
    used to screen positions.

    Returns
    -------
    rmax : float
    ymax, xmax : int

    Notes
    -----
    LEGACY: superseded by ``catalogFFT_irregular``. Uses ``maskheight/2`` as an
    array index, which produces a float and would fail under Python 3.
    Status: not currently called.
    """
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i//xlen
        x=i-y*xlen
        #print x,y
        if np.any(np.equal(domainmask[y+maskheight/2,x:x+maskwidth],1.)) and np.any(np.equal(domainmask[y:y+maskheight,x+maskwidth/2],1.)):
            rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
        else:
            rainsum[y,x]=0.
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    
    return rmax, wheremax[0][0], wheremax[1][0]


# Testing (parallel, without symmetric positions). FASTEST for large domains
# @njit(parallel=True, fastmath=True)
# def catalogNumba_irregular(temparray, trimmask, xlen, ylen, xloop, yloop,
#                            maskheight, maskwidth, rainsum, stride=1):
#     # Since we are not doing symmetric positions, we can simplify the logic
#     """
#     Numba-parallel version of the storm-catalog spatial search.

#     Each row of candidate positions is processed in parallel; the mask-weighted
#     sum at each position is accumulated explicitly while skipping NaNs. A
#     parallel per-row max is then reduced serially to the global max.

#     Parameters
#     ----------
#     temparray : np.ndarray, shape (ny, nx)
#         Accumulated rainfall field.
#     trimmask : np.ndarray, shape (maskheight, maskwidth)
#         Trimmed watershed mask.
#     xlen, ylen : int
#         Number of candidate positions in x and y.
#     xloop, yloop : int
#         Ignored (overwritten with ``xlen``/``ylen``); kept for signature
#         compatibility with ``catalogNumba``.
#     maskheight, maskwidth : int
#         Dimensions of ``trimmask``.
#     rainsum : np.ndarray, shape (ylen, xlen)
#         Work array, overwritten.
#     stride : int, optional
#         Step between candidate x positions (CATALOGACCELERATOR). Default 1.

#     Returns
#     -------
#     rmax : float
#     ymax, xmax : int

#     Notes
#     -----
#     Despite its name, it does not use a domain mask. Superseded by
#     ``catalogFFT_irregular``. Status: not currently called.
#     """
#     xloop = int(xlen)
#     yloop = int(ylen)

#     rainsum[:, :] = 0.0

#     # Parallel storm scan
#     for y in prange(0, yloop):
#         for x in range(0, xloop, stride):
#             if y + maskheight > temparray.shape[0] or x + maskwidth > temparray.shape[1]:
#                 continue

#             val = 0.0
#             for i in range(maskheight):
#                 for j in range(maskwidth):
#                     a = temparray[y + i, x + j]
#                     b = trimmask[i, j]
#                     if not np.isnan(a) and not np.isnan(b):
#                         val += a * b

#             rainsum[y, x] = val

#     # --- Parallel row-wise reduction ---
#     row_max = np.full(rainsum.shape[0], -1e30)
#     row_x = np.full(rainsum.shape[0], -1, dtype=np.int32)

#     for y in prange(rainsum.shape[0]):
#         max_val = -1e30
#         max_x = -1
#         for x in range(rainsum.shape[1]):
#             val = rainsum[y, x]
#             if val > max_val:
#                 max_val = val
#                 max_x = x
#         row_max[y] = max_val
#         row_x[y] = max_x

#     # --- Serial reduction over rows ---
#     rmax = -1e30
#     ymax = -1
#     xmax = -1
#     for y in range(rainsum.shape[0]):
#         if row_max[y] > rmax:
#             rmax = row_max[y]
#             ymax = y
#             xmax = row_x[y]

#     return rmax, ymax, xmax



@jit(nopython=True,  fastmath =  True)
def catalogNumba(temparray,trimmask,xlen,ylen,xloop,yloop,maskheight,maskwidth,rainsum,stride=1):
    """
    Numba version of the storm-catalog spatial search that exploits symmetry:
    each loop iteration fills four positions (from each corner of the domain
    inward), so only about a quarter of the loop iterations are needed.

    Parameters
    ----------
    temparray : np.ndarray, shape (ny, nx)
        Accumulated rainfall field.
    trimmask : np.ndarray, shape (maskheight, maskwidth)
        Trimmed watershed mask.
    xlen, ylen : int
        Number of candidate positions in x and y.
    xloop, yloop : int
        Number of loop iterations in x and y (about half of xlen/ylen).
    maskheight, maskwidth : int
        Dimensions of ``trimmask``.
    rainsum : np.ndarray, shape (ylen, xlen)
        Work array, overwritten.
    stride : int, optional
        Step between candidate x positions. Default 1.

    Returns
    -------
    rmax : float
    ymax, xmax : int

    Notes
    -----
    LEGACY: superseded by ``catalogFFT_irregular``. The mirrored positions are
    written to ``rainsum[..., xlen-x-1]`` but computed from a window starting
    at ``xlen-x`` (off by one); see the review notes.
    Status: not currently called.
    """
    for y in range(0, int32(yloop)):
        for x in range(0, int32(xloop),stride):

            rainsum[y, x] = np.nansum(np.multiply(temparray[y:(y+maskheight), x:(x+maskwidth)], trimmask))

            rainsum[y, xlen-x-1] = np.nansum(np.multiply(temparray[y:(y+maskheight), xlen-x:(xlen-x+maskwidth)], trimmask))
            rainsum[ylen-y-1, x] = np.nansum(np.multiply(temparray[ylen-y-1:(ylen-y-1+maskheight), x:(x+maskwidth)], trimmask))


            rainsum[ylen-y-1, xlen-x-1] = np.nansum(np.multiply(temparray[ylen-y:(ylen-y+maskheight), xlen-x:(xlen-x+maskwidth)], trimmask))

    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(np.equal(rainsum,rmax))
    return rmax, wheremax[0][0], wheremax[1][0]



@jit(nopython=True)
def DistributionBuilder(intenserain,tempmax,xlen,ylen,checksep):
    """
    Maintain, at every grid cell, a running list of the N largest storm totals
    ("intensity distribution"), with a separation check so the same storm is
    not counted twice.

    For each cell, if the cell is flagged in ``checksep`` (the previous time
    step already contributed to this storm), the stored value is updated in
    place if the new value is larger. Otherwise, if the new value exceeds the
    smallest stored value, it replaces it and the cell is flagged.

    Parameters
    ----------
    intenserain : np.ndarray, shape (N, ny, nx)
        Current top-N storm totals at each cell (updated in place).
    tempmax : np.ndarray, shape (ny, nx)
        Candidate storm totals for the current time step.
    xlen, ylen : int
        Grid dimensions.
    checksep : np.ndarray of bool, shape (N, ny, nx)
        Flags marking which slot the ongoing storm occupies at each cell.

    Returns
    -------
    intenserain, checksep : np.ndarray

    Notes
    -----
    LEGACY: supported the old intensity-file workflow. Status: not currently called.
    """
    for y in np.arange(0,ylen):
        for x in np.arange(0,xlen):
            if np.any(checksep[:,y,x]):
                #fixind=np.where(checksep[:,y,x]==True)
                for i in np.arange(0,checksep.shape[0]):
                    if checksep[i,y,x]==True:
                        fixind=i
                        break
                if tempmax[y,x]>intenserain[fixind,y,x]:
                    intenserain[fixind,y,x]=tempmax[y,x]
                    checksep[:,y,x]=False
                    checksep[fixind,y,x]=True
                else:
                    checksep[fixind,y,x]=False
            elif tempmax[y,x]>np.min(intenserain[:,y,x]):
                fixind=np.argmin(intenserain[:,y,x])
                intenserain[fixind,y,x]=tempmax[y,x]
                checksep[fixind,y,x]=True
    return intenserain,checksep

# slightly faster numpy-based version of above
def DistributionBuilderFast(intenserain,tempmax,xlen,ylen,checksep):
    """
    Vectorized NumPy version of ``DistributionBuilder`` (same inputs/outputs).

    Notes
    -----
    LEGACY. The line ``intenserain[minsep,flatsep][islarger]=...`` uses chained
    fancy indexing, which assigns into a temporary copy, so that update has no
    effect; see the review notes. Status: not currently called.
    """
    minrain=np.min(intenserain,axis=0)
    if np.any(checksep):
        
        flatsep=np.any(checksep,axis=0)
        minsep=np.argmax(checksep[:,flatsep],axis=0)
        
        islarger=np.greater(tempmax[flatsep],intenserain[minsep,flatsep])
        if np.any(islarger):
            intenserain[minsep,flatsep][islarger]=tempmax[flatsep][islarger]
            checksep[:]=False
            checksep[minsep,flatsep]=True
        else:
            checksep[minsep,flatsep]=False
    elif np.any(np.greater(tempmax,minrain)):
        #else:
        fixind=np.greater(tempmax,minrain)
        minrainind=np.argmin(intenserain,axis=0)
        
        intenserain[minrainind[fixind],fixind]=tempmax[fixind]
        checksep[minrainind[fixind],fixind]=True
    return intenserain,checksep



# LEGACY (commented out, not used): old SSTalt without rescaling; superseded by SSTalt below. Candidate for removal.
#def SSTalt(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intense_data=False):
#    rainsum=np.zeros((len(sstx)),dtype='float32')
#   nreals=len(rainsum)
#
#    for i in range(0,nreals):
#        rainsum[i]=np.nansum(np.multiply(passrain[(ssty[i]) : (ssty[i]+maskheight) , (sstx[i]) : (sstx[i]+maskwidth)],trimmask))
#    return rainsum




# LEGACY (debugging leftovers): variable assignments used to test SSTalt interactively. Candidate for removal.
# sstx=whichx[whichstorms==i,pt]
# ssty=whichy[whichstorms==i,pt]
# durcheck=durcorrection                
# intensemean=None
# intensestd=None
# intensecorr=None
# homemean=None
# homestd=None



#@jit(nopython=True,fastmath=True)
def SSTalt(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intensemean=None,intensestd=None,intensecorr=None,homemean=None,homestd=None,durcheck=False):
    """
    Compute the basin-averaged rainfall for a set of transposed positions of
    one parent storm, optionally applying a "ratio rescaling" multiplier.

    For each position k, the storm field ``passrain`` is sampled at
    ``[ssty[k]:ssty[k]+maskheight, sstx[k]:sstx[k]+maskwidth]`` and multiplied
    by ``trimmask``. If every value in that window is below 0.5 (mm/hr summed
    over time), the result is set to 0 as a shortcut.

    Three rescaling modes are supported, chosen by which optional arguments
    are given:

    * none (default): multiplier = 1.
    * deterministic (``intensemean`` and ``homemean``): multiplier =
      exp(homemean - intensemean[y, x]), i.e. the ratio of the (log-space)
      mean storm total at the target location to that at the source location.
    * stochastic (also ``intensestd``, ``intensecorr``, ``homestd``): the
      multiplier is drawn from a lognormal distribution whose log-mean is the
      deterministic ratio and whose log-std accounts for the correlation
      between the two locations.

    Multipliers above ``maxmultiplier`` (1.5) are reset to 1.

    Parameters
    ----------
    passrain : np.ndarray
        Parent storm rainfall. Shape (ny, nx) when ``durcheck`` is False
        (already summed over time); shape (nt, ny, nx) when ``durcheck`` is
        True (rolling sums, one per candidate start time).
    sstx, ssty : np.ndarray of int, shape (npos,)
        Upper-left x/y indices of each transposition.
    trimmask : np.ndarray, shape (maskheight, maskwidth)
    maskheight, maskwidth : int
    intensemean, intensestd, intensecorr : np.ndarray, optional
        Gridded log-space mean, std, and home-location correlation of storm
        totals across the domain.
    homemean, homestd : float, optional
        Log-space mean and std of storm totals at the watershed ("home").
    durcheck : bool, optional
        If True, apply the duration correction: take the maximum over all
        time windows in ``passrain`` and record which window.

    Returns
    -------
    rainsum : np.ndarray of float32, shape (npos,)
        Mask-weighted sum (still needs ``* timeres/60 / mnorm`` to become a
        basin-average depth in mm).
    multiout : np.ndarray, shape (npos,)
        Only returned when rescaling. -999 where the storm had no rain.
    whichstep : np.ndarray of int32, shape (npos,)
        Index of the maximizing time window (0 if ``durcheck`` is False).

    Notes
    -----
    The number of returned values (2 or 3) depends on the rescaling mode.
    Status: used by RainyDay_Py3.py.
    """
    maxmultiplier=1.5
    
    rainsum=np.zeros((len(sstx)),dtype='float32')
    whichstep=np.zeros((len(sstx)),dtype='int32')
    nreals=len(rainsum)
    nsteps=passrain.shape[0]
    multiout=np.empty_like(rainsum)
    if (intensemean is not None) and (homemean is not None):
        domean=True
    else:
        domean=False

    if (intensestd is not None) and (intensecorr is not None) and (homestd is not None):
        #rquant=np.random.random_integers(5,high=95,size=nreals)/100.
        rquant=np.random.random_sample(size=nreals)
        doall=True
    else:
        doall=False
        rquant=np.nan
        
    
    if durcheck==False:
        exprain=np.expand_dims(passrain,0)
    else:
        exprain=passrain
        

    for k in range(0,nreals):
        y=int(ssty[k])
        x=int(sstx[k])
        if np.all(np.less(exprain[:,y:y+maskheight,x:x+maskwidth],0.5)):
            rainsum[k]=0.
            multiout[k]=-999.
        else:
            if domean:
                #sys.exit('need to fix short duration part')
                muR=homemean-intensemean[y,x]      #LY: we don't use mean, so here we need to revise and use trimmask
                if doall:
                    # std of the log-ratio of two correlated normals: var(A-B) = sA^2 + sB^2 - 2*rho*sA*sB
                    stdR=np.sqrt(np.power(homestd,2)+np.power(intensestd[y,x],2)-2.*intensecorr[y,x]*homestd*intensestd[y,x])
                   # multiplier=sp.stats.lognorm.ppf(rquant[k],stdR,loc=0,scale=np.exp(muR))     
                    #multiplier=10.
                    #while multiplier>maxmultiplier:       # who knows what the right number is to use here...
                    # Lognormal draw via the inverse normal CDF: for u ~ U(0,1),
                    # sqrt(2)*erfinv(2u-1) is a standard normal quantile, so
                    # multiplier = exp(muR + stdR*z). (erfinv is recomputed for all
                    # positions on every k; only element k is used.)
                    inverrf=sp.special.erfinv(2.*rquant-1.)
                    multiplier=np.exp(muR+np.sqrt(2.*np.power(stdR,2))*inverrf[k])
                    
                    #multiplier=np.random.lognormal(muR,stdR)
                    if multiplier>maxmultiplier:
                        multiplier=1.    
                else:
                    multiplier=np.exp(muR)
                    if multiplier>maxmultiplier:
                        multiplier=1.
            else:
                multiplier=1.
#            print("still going!")
            if multiplier>maxmultiplier:
                sys.exit("Something seems to be going horribly wrong in the multiplier scheme!")
            else:
                multiout[k]=multiplier
        
            if durcheck==True:            
                storesum=0.
                storestep=0
                for kk in range(0,nsteps):
                    #tempsum=numba_multimask_calc(passrain[kk,:],rsum,train,trimmask,ssty[k],maskheight,sstx[k],maskwidth)*multiplier
                    tempsum=numba_multimask_calc(passrain[kk,:],trimmask,y,x,maskheight,maskwidth)*multiplier
                    if tempsum>storesum:
                        storesum=tempsum
                        storestep=kk
                rainsum[k]=storesum
                whichstep[k]=storestep
            else:
                rainsum[k]=numba_multimask_calc(passrain,trimmask,y,x,maskheight,maskwidth)*multiplier
    if domean:
        return rainsum,multiout,whichstep
    else:
        return rainsum,whichstep

# =========================================================================================
# added Lei 02122025: Dimensionless rescaling
# updated Lei 04012025: extract top n storm/multiplier for writing scenarios (reduce memory)
# =========================================================================================
# LEGACY (commented out, not used): earlier SSTalt_normalized without top-N scenario tracking; superseded by the active version below. Candidate for removal.
# @jit(fastmath=True)
# def SSTalt_normalized(passrain, sstx, ssty, trimmask, maskheight, maskwidth, intensegrid=None, homegrid=None, durcheck=False):
#     #maxmultiplier = 1.5  #LY: should we use this?
#
#     rainsum = np.zeros((len(sstx)), dtype='float32')
#     whichstep = np.zeros((len(sstx)), dtype='int32')
#     nreals = len(rainsum)
#     nsteps = passrain.shape[0]
#     multiout = np.full((len(sstx), maskheight, maskwidth), np.nan, dtype='float32')
#
#     if (intensegrid is not None) and (homegrid is not None):
#         rescale = True
#     else:
#         rescale = False
#
#     if durcheck == False:
#         exprain = np.expand_dims(passrain, 0)
#     else:
#         exprain = passrain
#
#     for k in range(0, nreals):
#         y = int(ssty[k])
#         x = int(sstx[k])
#         if np.all(np.less(exprain[:, y:y + maskheight, x:x + maskwidth], 0.5)):
#             rainsum[k] = 0.
#             multiout[k] = -9999.
#         else:
#             if rescale:
#                 # sys.exit('need to fix short duration part')
#                 intensegrid_trans = intensegrid[y:y + maskheight, x:x + maskwidth] * trimmask
#                 multiplier=np.exp( homegrid - intensegrid_trans )
#                 # multiplier[multiplier > maxmultiplier] = 1.5
#                 valid_mask = (trimmask != 0)
#                 valid_multiplier = multiplier[valid_mask]
#
#                 sorted_arr = np.sort(valid_multiplier)
#                 n = len(sorted_arr)
#                 p10 = sorted_arr[max(0, int(0.1 * n) - 1)]
#                 p90 = sorted_arr[min(n - 1, int(0.9 * n))]
#
#                 multiplier = np.clip(multiplier, p10, p90)
#                 multiout[k, :, :] = multiplier
#             else:
#                 multiplier = 1.
#
#             if durcheck == True:
#                 storesum = 0.
#                 storestep = 0
#                 for kk in range(0, nsteps):
#                     if rescale:
#                         tempsum = numba_multimask_calc_rescale(passrain[kk, y:y + maskheight, x:x + maskwidth], trimmask, multiplier)
#                     else:
#                         tempsum = numba_multimask_calc(passrain[kk, :], trimmask, y, x, maskheight, maskwidth) * multiplier
#
#                     if tempsum > storesum:
#                         storesum = tempsum
#                         storestep = kk
#
#                 rainsum[k] = storesum
#                 whichstep[k] = storestep
#             else:
#                 if rescale:
#                     rainsum[k] = numba_multimask_calc_rescale(passrain[y:y + maskheight, x:x + maskwidth], trimmask, multiplier)
#                 else:
#                     rainsum[k] = numba_multimask_calc(passrain, trimmask, y, x, maskheight, maskwidth) * multiplier
#     if rescale:
#         return rainsum, multiout, whichstep
#     else:
#         return rainsum, whichstep


@jit(fastmath=True)
def SSTalt_normalized(passrain, sstx, ssty, trimmask, maskheight, maskwidth, top_whichrain, top_multiplier, durcheck=False, intensegrid=None, homegrid=None, Scenarios=False, storm_pos=None):
    """
    "Normalized" (dimensionless) SST: compute basin rainfall for a set of
    transposed positions of one parent storm, rescaling each cell by the ratio
    of a design-precipitation field at the target versus the home location.

    Added by Lei Yan (Feb 2025); top-N tracking added April 2025; BLF (Sept
    2026) removed the ``trimmask`` weighting of the log-field and set
    non-finite multipliers to 0.

    For each transposed position k, a cell-by-cell multiplier is computed as
    ``exp(homegrid - intensegrid[y:y+h, x:x+w])``, where both grids are the log
    of a design precipitation depth (e.g., the 10-year quantile). The
    multiplier field is applied to the storm rainfall before mask-weighting and
    summing. Positions where all rainfall is below 0.5 are set to 0.

    When ``Scenarios`` is True, the function also keeps, for every synthetic
    year and realization, the ``nperyear`` largest rainfall values and their
    multiplier fields in ``top_whichrain``/``top_multiplier``. Those arrays are
    updated in place and kept sorted in ascending order (index 0 = smallest),
    so the multipliers are available later for writing scenario files without
    storing a multiplier field for every sampled storm.

    Parameters
    ----------
    passrain : np.ndarray
        Parent storm rainfall; (ny, nx) or (nt, ny, nx) if ``durcheck``.
    sstx, ssty : np.ndarray of int, shape (npos,)
        Upper-left indices of each transposition.
    trimmask : np.ndarray, shape (maskheight, maskwidth)
    maskheight, maskwidth : int
    top_whichrain : np.ndarray, shape (nperyear, nsimulations, nrealizations)
        Running top-N rainfall values (updated in place).
    top_multiplier : np.ndarray, shape (nperyear, nsimulations, nrealizations, maskheight, maskwidth)
        Multiplier fields matching ``top_whichrain`` (updated in place).
    durcheck : bool, optional
        Apply the duration correction (max over time windows).
    intensegrid : np.ndarray, shape (ny, nx), optional
        Log design-precipitation field over the transposition domain.
    homegrid : np.ndarray, shape (maskheight, maskwidth), optional
        Log design-precipitation field over the watershed rectangle.
    Scenarios : bool, optional
        Whether to maintain the top-N arrays.
    storm_pos : tuple of np.ndarray, optional
        Output of ``np.where(whichstorms == i)``: (storm slot, synthetic year,
        realization) for each position. Needed when ``Scenarios`` is True.

    Returns
    -------
    rainsum : np.ndarray of float32, shape (npos,)
        Mask-weighted (rescaled) sums; multiply by ``timeres/60/mnorm`` for
        basin-average depth in mm.
    whichstep : np.ndarray of int32, shape (npos,)

    Notes
    -----
    Status: used by RainyDay_Py3.py (NORMALIZEDSST = "dimensionless").
    """
    rainsum = np.zeros((len(sstx)), dtype='float32')
    whichstep = np.zeros((len(sstx)), dtype='int32')
    nreals = len(rainsum)
    nsteps = passrain.shape[0]
    multiout = np.full((len(sstx), maskheight, maskwidth), np.nan, dtype='float32')

    if (intensegrid is not None) and (homegrid is not None):
        rescale = True
    else:
        rescale = False

    if durcheck == False:
        exprain = np.expand_dims(passrain, 0)
    else:
        exprain = passrain

    for k in range(nreals):
        y = int(ssty[k])
        x = int(sstx[k])

        # Shortcut: if the transposed window has essentially no rain, skip the computation
        if np.all(np.less(exprain[:, y:y + maskheight, x:x + maskwidth], 0.5)):
            rainsum[k] = 0.
            multiout[k] = -9999.
        else:
            if rescale:
                # Cell-by-cell multiplier = design depth at home / design depth at target
                # (both grids are logs, so the ratio is exp of the difference).
                # BLF 9152026: Since intensegrid is log transfomed, we don't want to multiply by trimmask weightings.
                #intensegrid_trans = intensegrid[y:y + maskheight, x:x + maskwidth] * trimmask
                intensegrid_trans = intensegrid[y:y + maskheight, x:x + maskwidth]
                multiplier = np.exp(homegrid - intensegrid_trans)
                # BLF 9152026: If we have an infinite multiplier have it be 0. 
                multiplier = np.where(np.isfinite(multiplier), multiplier, 0.0)


                # valid_mask = (trimmask != 0)
                # valid_multiplier = multiplier[valid_mask]
                # sorted_arr = np.sort(valid_multiplier)
                # n = len(sorted_arr)
                # p10 = sorted_arr[max(0, int(0.1*n)-1)]
                # p90 = sorted_arr[min(n-1, int(0.9*n))]
                # multiplier = np.clip(multiplier, p10, p90)
                multiout[k, :, :] = multiplier
            else:
                multiplier = 1.

            if durcheck == True:
                storesum = 0.
                storestep = 0
                for kk in range(0, nsteps):
                    if rescale:
                        tempsum = numba_multimask_calc_rescale(passrain[kk, y:y + maskheight, x:x + maskwidth], trimmask, multiplier)
                    else:
                        tempsum = numba_multimask_calc(passrain[kk, :], trimmask, y, x, maskheight, maskwidth) * multiplier

                    if tempsum > storesum:
                        storesum = tempsum
                        storestep = kk

                rainsum[k] = storesum
                whichstep[k] = storestep
            else:
                if rescale:
                    rainsum[k] = numba_multimask_calc_rescale(passrain[y:y + maskheight, x:x + maskwidth], trimmask, multiplier)
                else:
                    rainsum[k] = numba_multimask_calc(passrain, trimmask, y, x, maskheight, maskwidth) * multiplier

        # -----------------------------------------------------------
        # If Scenarios==True
        # Sort rainsum and extract the corresponding top n multiplier
        # -----------------------------------------------------------
        if Scenarios and (storm_pos is not None):
            # storm_pos = np.where(whichstorms==i): [0]=storm slot within year, [1]=synthetic year, [2]=realization
            y_ = storm_pos[1][k]
            z_ = storm_pos[2][k]
            val = rainsum[k]

            # compare and update top n storms
            if val > top_whichrain[0, y_, z_]:
                # update the smallest one
                top_whichrain[0, y_, z_] = val
                top_multiplier[0, y_, z_, :, :] = multiout[k,:,:]
                # re-sort
                subvals = top_whichrain[:, y_, z_].copy()
                subidx = np.argsort(subvals)
                sorted_vals = subvals[subidx]
                sorted_multi = top_multiplier[subidx, y_, z_, :, :].copy()
                top_whichrain[:, y_, z_] = sorted_vals
                top_multiplier[:, y_, z_, :, :] = sorted_multi

    return rainsum, whichstep


# =============================================================================
# added Lei 02122025: Calculate the rescaled rainfall
# =============================================================================
@jit(nopython=True, fastmath=True)
def numba_multimask_calc_rescale(passrain, trimmask, multiplier):
    """
    Sum of ``passrain * multiplier * trimmask`` over a watershed window.

    Added by Lei Yan (Feb 2025) for normalized SST.

    Parameters
    ----------
    passrain : np.ndarray, shape (maskheight, maskwidth)
        Rainfall already cropped to the transposed window.
    trimmask : np.ndarray, shape (maskheight, maskwidth)
    multiplier : np.ndarray, shape (maskheight, maskwidth)
        Cell-by-cell rescaling factors.

    Returns
    -------
    float
    """
    train = passrain * multiplier * trimmask
    rainsum = np.sum(train)
    return rainsum
# LEGACY (commented out, not used): explicit-loop variant of numba_multimask_calc_rescale. Candidate for removal.
# @jit(nopython=True, fastmath=True)
# def numba_multimask_calc_rescale(passrain, trimmask, multiplier):
#     total = 0.0
#     passrain = np.ascontiguousarray(passrain.astype(np.float32))
#     multiplier = np.ascontiguousarray(multiplier.astype(np.float32))
#     trimmask = np.ascontiguousarray(trimmask.astype(np.float32))
#     for i in prange(passrain.shape[0]):
#         for j in range(passrain.shape[1]):
#             total += passrain[i,j] * multiplier[i,j] * trimmask[i,j]
#     return total

#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def numba_multimask_calc(passrain_temp,trimmask,y,x,maskheight,maskwidth):
    """
    Mask-weighted rainfall sum for a watershed window at (y, x).

    Parameters
    ----------
    passrain_temp : np.ndarray, shape (ny, nx)
        Full-domain rainfall field.
    trimmask : np.ndarray, shape (maskheight, maskwidth)
    y, x : int
        Upper-left corner of the window.
    maskheight, maskwidth : int

    Returns
    -------
    float
        ``sum(passrain_temp[y:y+h, x:x+w] * trimmask)`` (NaNs propagate).
    """
    train=np.multiply(passrain_temp[y : y+maskheight , x : x+maskwidth],trimmask)
    rainsum=np.sum(train)       
    return rainsum


@jit(fastmath=True)
def SSTalt_singlecell(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intensemean=None,intensestd=None,intensecorr=None,homemean=None,homestd=None,durcheck=False):
    """
    Single-grid-cell (POINTAREA = "point") version of ``SSTalt``.

    Chooses the rescaling mode from the arguments given (none, deterministic,
    or stochastic; see ``SSTalt``), then delegates the per-position loop to
    ``killerloop_singlecell``.

    Parameters
    ----------
    See ``SSTalt``. ``trimmask``, ``maskheight`` and ``maskwidth`` are accepted
    for signature compatibility but not used.

    Returns
    -------
    rainsum, multiout, whichstep
        When a rescaling mode is active.
    rainsum, whichstep
        When no rescaling is used.

    Notes
    -----
    Status: used by RainyDay_Py3.py for point analyses.
    """
    rainsum=np.zeros((len(sstx)),dtype='float32')
    whichstep=np.zeros((len(sstx)),dtype='int32')
    nreals=len(rainsum)
    nsteps=passrain.shape[0]
    multiout=np.empty_like(rainsum)

    # do we do deterministic or dimensionless rescaling?
    if (intensemean is not None) and (homemean is not None):
        domean=True
    else:
        domean=False       

    # do we do stochastic rescaling?    
    if (intensestd is not None) and (intensecorr is not None) and (homestd is not None):
        rquant=np.random.random_sample(size=nreals)
        inverrf=sp.special.erfinv(2.*rquant-1.)
        doall=True
    else:
        doall=False
        #rquant=np.nan

    if durcheck==False:
        passrain=np.expand_dims(passrain,0)
       
    # deterministic or dimensionless:
    if domean and doall==False:
        rain,multi,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,intensemean=intensemean,homemean=homemean,multiout=multiout)
        return rain,multi,step
    
    # stochastic:
    elif doall:
        rain,multi,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,intensemean=intensemean,intensestd=intensestd,intensecorr=intensecorr,homemean=homemean,homestd=homestd,multiout=multiout,inverrf=inverrf)
        return rain,multi,step
    
    # no rescaling:
    else:
        rain,_,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,multiout=multiout)
        return rain,step
    


#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=False,intensemean=None,homemean=None,homestd=None,multiout=None,rquant=None,intensestd=None,intensecorr=None,inverrf=None):
    """
    Inner loop for ``SSTalt_singlecell``: rainfall at a single cell for each
    transposed position, with optional deterministic/stochastic multiplier.

    Parameters
    ----------
    passrain : np.ndarray, shape (nt, ny, nx)
        Storm rainfall (a leading axis of length 1 is added by the caller when
        ``durcheck`` is False).
    rainsum, whichstep, multiout : np.ndarray, shape (npos,)
        Output arrays (filled in place and returned).
    nreals : int
        Number of positions.
    ssty, sstx : np.ndarray of int
        Cell indices of each transposition.
    nsteps : int
        Number of time windows (used only when ``durcheck`` is True).
    durcheck : bool
    intensemean, homemean, homestd, intensestd, intensecorr : optional
        Rescaling statistics (see ``SSTalt``).
    rquant : unused
    inverrf : np.ndarray, optional
        Pre-computed ``erfinv(2u-1)`` values for the stochastic multiplier.

    Returns
    -------
    rainsum, multiout, whichstep : np.ndarray

    Notes
    -----
    When ``durcheck`` is False the multiplier is computed but not applied and
    ``multiout`` is not filled; see the review notes.
    """
    maxmultiplier=1.5  # who knows what the right number is to use here...
    for k in prange(nreals):
        y=int(ssty[k])
        x=int(sstx[k])
        
        # deterministic or dimensionless:
        if (intensemean is not None) and (homemean is not None) and (homestd is None):
            if np.less(homemean,0.001) or np.less(intensemean[y,x],0.001):
                multiplier=1.           # or maybe this should be zero     
            else:
                multiplier=np.exp(homemean-intensemean[y,x])
                if multiplier>maxmultiplier:           
                    multiplier=1.        # or maybe this should be zero
                    
        # stochastic:
        elif (intensemean is not None) and (homemean is not None) and (homestd is not None):
            if np.less(homemean,0.001) or np.less(intensemean[y,x],0.001):
                multiplier=1.          # or maybe this should be zero
            else:
                muR=homemean-intensemean[y,x]
                stdR=np.sqrt(np.power(homestd,2)+np.power(intensestd[y,x],2)-2*intensecorr[y,x]*homestd*intensestd[y,x])

                multiplier=np.exp(muR+np.sqrt(2.*np.power(stdR,2))*inverrf[k])
                if multiplier>maxmultiplier:
                    multiplier=1.        # or maybe this should be zero
        
        # no rescaling:
        else:
            multiplier=1.
            
        if durcheck==False:
            rainsum[k]=np.nansum(passrain[:,y, x])
        else:
            storesum=0.
            storestep=0
            for kk in range(nsteps):
                tempsum=passrain[kk,y,x]
                if tempsum>storesum:
                    storesum=tempsum
                    storestep=kk
            rainsum[k]=storesum*multiplier
            multiout[k]=multiplier
            whichstep[k]=storestep
            
    return rainsum,multiout,whichstep



# LEGACY (commented out, not used): killerloop (multi-cell Numba loop); never adopted. Candidate for removal.
#@jit(nopython=True,fastmath=True,parallel=True)
#def killerloop(passrain,rainsum,nreals,ssty,sstx,maskheight,maskwidth,trimmask,nsteps,durcheck):
#    for k in prange(nreals):
#        spanx=int64(sstx[k]+maskwidth)
#        spany=int64(ssty[k]+maskheight)
#        if np.all(np.less(passrain[:,ssty[k]:spany,sstx[k]:spanx],0.5)):
#            rainsum[k]=0.
#        else:
#            if durcheck==False:
#                rainsum[k]=np.nansum(np.multiply(passrain[ssty[k] : spany , sstx[k] : spanx],trimmask))
#            else:
#                storesum=float32(0.)
#                for kk in range(nsteps):
#                    tempsum=np.nansum(np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],trimmask))
#                    if tempsum>storesum:
#                        storesum=tempsum
#                rainsum[k]=storesum
#    return rainsum
    
    
                    #whichstep[k]=storestep
#return rainsum,whichstep



# this function below never worked for some unknown Numba problem-error messages indicated that it wasn't my fault!!! Some problem in tempsum
# LEGACY (commented out, not used): killerloop variant that never compiled under Numba (see note below). Candidate for removal.
#@jit(nopython=True,fastmath=True,parallel=True)
#def killerloop(passrain,rainsum,nreals,ssty,sstx,maskheight,maskwidth,masktile,nsteps,durcheck):
#    for k in prange(nreals):
#        spanx=sstx[k]+maskwidth
#        spany=ssty[k]+maskheight
#        if np.all(np.less(passrain[:,ssty[k]:spany,sstx[k]:spanx],0.5)):
#            rainsum[k]=0.
#        else:
#            if durcheck==False:
#                #tempstep=np.multiply(passrain[:,ssty[k] : spany , sstx[k] : spanx],trimmask)
#                #xnum=int64(sstx[k])
#                #ynum=int64(ssty[k])
#                #rainsum[k]=np.nansum(passrain[:,ssty[k], sstx[k]])
#                rainsum[k]=np.nansum(np.multiply(passrain[:,ssty[k] : spany , sstx[k] : spanx],masktile))
#            else:
#                storesum=float32(0.)
#                for kk in range(nsteps):
#                    #tempsum=0.
#                    #tempsum=np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],masktile[0,:,:])
#                    tempsum=np.nansum(np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],masktile[0,:,:]))
#    return rainsum


#==============================================================================
# THIS VARIANT IS SIMPLER AND UNLIKE SSTWRITE, IT ACTUALLY WORKS RELIABLY!
#==============================================================================
# LEGACY (commented out, not used): SSTwriteAlt; pre-2023 scenario writer. Candidate for removal.
#def SSTwriteAlt(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth):
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)
#    #ctr=0
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        for j in unqwhere:
#            #ctr=ctr+1
#            #print ctr
#            outrain[j,:]=np.multiply(catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],trimmask)
#    return outrain
       

#==============================================================================
# THIS VARIANT IS SAME AS ABOVE, BUT HAS A MORE INTERESTING RAINFALL PREPENDING PROCEDURE
#==============================================================================

# LEGACY (commented out, not used): SSTwriteAltPreCat; pre-2023 scenario writer with spin-up rainfall. Candidate for removal.
#def SSTwriteAltPreCat(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime):    
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)
#
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
#        for j in unqwhere:
#            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
#    

#==============================================================================
# SAME AS ABOVE, BUT HANDLES STORM ROTATION
#==============================================================================    
    
# LEGACY (commented out, not used): SSTwriteAltPreCatRotation; pre-2023 scenario writer with rotation. Candidate for removal.
#def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin,rainprop):
##def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin):
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
#
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
#        for j in unqwhere:
#            inrain=catrain[unqstm[i],:].copy()
#            
#            xctr=rlzx[j]+maskwidth/2.
#            yctr=rlzy[j]+maskheight/2.
#            xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
#            ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
#    
#            ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
#            ingridx=ingridx.flatten()
#            ingridy=ingridy.flatten()
#            outgrid=np.column_stack((ingridx,ingridy))  
#            
#            for k in range(0,inrain.shape[0]):
#                interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
#                inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
#                #inrain[k,:]=temprain
#            
#            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#
#            outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
       
@jit(fastmath=True)
def SSTspin_write_v2(catrain,rlzx,rlzy,rlzstm,trimmask,maskheight,maskwidth,precat,ptime,rainprop,rlzanglebin=None,delarray=None,spin=False,flexspin=True,samptype='uniform',cumkernel=None,rotation=False,domaintype='rectangular'):
    """
    Build output rainfall scenarios for many transposed storms, optionally
    rotating the storm and/or prepending "spin-up" rainfall from before the
    storm.

    For each unique parent storm in ``rlzstm``, and for each realization that
    uses it, the storm is (optionally) rotated about the transposition center
    using the precomputed Delaunay triangulations in ``delarray``, cropped to
    the watershed rectangle at (``rlzy``, ``rlzx``), optionally prefixed with a
    randomly chosen spin-up period from ``precat`` (same month +/- 1), and
    multiplied by ``trimmask``.

    Parameters
    ----------
    catrain : np.ndarray, shape (nstorms, nt, ny, nx)
        Storm catalog rainfall.
    rlzx, rlzy, rlzstm : np.ndarray of int, shape (nout,)
        Transposition x/y indices and parent-storm index for each output.
    trimmask : np.ndarray, shape (maskheight, maskwidth)
    maskheight, maskwidth : int
    precat : np.ndarray, shape (nstorms, nt_pre, ny, nx)
        Spin-up rainfall preceding each catalog storm.
    ptime : np.ndarray of datetime64
        Time of each catalog storm (used to match months for spin-up).
    rainprop : GriddedRainProperties
    rlzanglebin : np.ndarray of int, optional
        Rotation-angle bin for each output (1-based).
    delarray : list, optional
        Delaunay triangulations per storm and angle bin (see main script).
    spin : bool, optional
        Prepend spin-up rainfall.
    flexspin : bool, optional
        If True, spin-up rainfall is taken from a random location in the
        domain rather than from the transposition location.
    samptype : str, optional
        'uniform' or 'kernel'; controls how flexspin locations are drawn.
    cumkernel : np.ndarray, optional
        Cumulative transposition probability map (for kernel sampling).
    rotation : bool, optional
    domaintype : str, optional
        'rectangular' or 'irregular'.

    Returns
    -------
    outrain : np.ndarray of float32, shape (nout, nt_pre+nt, maskheight, maskwidth)

    Notes
    -----
    LEGACY: from the pre-August-2023 scenario writer. It calls ``numbakernel``
    with two arguments (it needs five) and uses ``np.random.random_integers``,
    which was removed from recent NumPy. Status: not currently called.
    """
    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
    nyrs=np.int16(rlzx.shape[0])
    raindur=np.int16(catrain.shape[1]+precat.shape[1])
    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
    
    for i in range(0,len(unqstm)):
        unqwhere=np.where(unqstm[i]==rlzstm)[0]
        unqmonth=ptime[unqstm[i]]
        pretimeind=np.where(np.logical_and(ptime>unqmonth-1,ptime<unqmonth+1))[0]
        
        # flexspin allows you to use spinup rainfall from anywhere in transposition domain, rather than just storm locations, but it doesn't seem to be very useful based on initial testing
        if spin==True and flexspin==True:       
            if samptype=='kernel' or domaintype=='irregular':
                rndloc=np.random.random_sample(len(unqwhere))
                shiftprex,shiftprey=numbakernel(rndloc,cumkernel)
            else:
                shiftprex=np.random.random_integers(0,np.int16(rainprop.subdimensions[1])-maskwidth-1,len(unqwhere))
                shiftprey=np.random.random_integers(0,np.int16(rainprop.subdimensions[0])-maskheight-1,len(unqwhere))
            
        ctr=0   
        for j in unqwhere:
            inrain=catrain[unqstm[i],:].copy()
                        
            # this doesn't rotate the prepended rainfall
            if rotation==True:
                xctr=rlzx[j]+maskwidth/2.
                yctr=rlzy[j]+maskheight/2.
                xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
                ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
        
                ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
                ingridx=ingridx.flatten()
                ingridy=ingridy.flatten()
                outgrid=np.column_stack((ingridx,ingridy))  
                
                for k in range(0,inrain.shape[0]):
                    interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
                    inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
                    
            if spin==True and flexspin==True:
                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(shiftprey[ctr]) : (shiftprey[ctr]+maskheight) , (shiftprex[ctr]) : (shiftprex[ctr]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
            elif spin==True and flexspin==False:
                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
            elif spin==False:
                temprain=inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]
            else:
                sys.exit("what else is there?")
            ctr=ctr+1

            outrain[j,:]=np.multiply(temprain,trimmask)
    return outrain


##==============================================================================
## SAME AS ABOVE, BUT A BIT MORE DYNAMIC IN TERMS OF SPINUP
##==============================================================================    
# LEGACY (commented out, not used): older SSTspin_write_v2 with intensity-based rescaling (never tested). Candidate for removal.
#def SSTspin_write_v2(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,rainprop,rlzanglebin=None,delarray=None,spin=False,flexspin=True,samptype='uniform',cumkernel=None,rotation=False,domaintype='rectangular',intense_data=False):
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
#    
#    if intense_data!=False:
#        sys.exit("Scenario writing for intensity-based resampling not tested!")
#        intquant=intense_data[0]
#        fullmu=intense_data[1]
#        fullstd=intense_data[2]
#        muorig=intense_data[3]
#        stdorig=intense_data[4]
#    
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-1,ptime<unqmonth+1))[0]
#        
#        if transpotype=='intensity':
#            origmu=np.multiply(murain[caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask)
#            origstd=np.multiply(stdrain[caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask)
#            #intense_dat=[intquant[],murain,stdrain,origmu,origstd]
#        
#        # flexspin allows you to use spinup rainfall from anywhere in transposition domain, rather than just storm locations, but it doesn't seem to be very useful based on initial testing
#        if spin==True and flexspin==True:       
#            if samptype=='kernel' or domaintype=='irregular':
#                rndloc=np.random.random_sample(len(unqwhere))
#                shiftprex,shiftprey=numbakernel(rndloc,cumkernel)
#            else:
#                shiftprex=np.random.random_integers(0,np.int(rainprop.subdimensions[1])-maskwidth-1,len(unqwhere))
#                shiftprey=np.random.random_integers(0,np.int(rainprop.subdimensions[0])-maskheight-1,len(unqwhere))
#            
#        ctr=0   
#        for j in unqwhere:
#            inrain=catrain[unqstm[i],:].copy()
#            
#            if intense_data!=False:
#                transmu=np.multiply(fullmu[(rlzy[i]) : (rlzy[i]+maskheight) , (rlzx[i]) : (rlzx[i]+maskwidth)],trimmask)
#                transtd=np.multiply(fullstd[(rlzy[i]) : (rlzy[i]+maskheight) , (rlzx[i]) : (rlzx[i]+maskwidth)],trimmask)
#                mu_multi=transmu/muorig
#                std_multi=np.abs(transtd-stdorig)/stdorig
#                multipliermask=norm.ppf(intquant[i],loc=mu_multi,scale=std_multi)
#                multipliermask[multipliermask<0.]=0.
#                multipliermask[np.isnan(multipliermask)]=0.
#            
#            # this doesn't rotate the prepended rainfall
#            if rotation==True:
#                xctr=rlzx[j]+maskwidth/2.
#                yctr=rlzy[j]+maskheight/2.
#                xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
#                ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
#        
#                ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
#                ingridx=ingridx.flatten()
#                ingridy=ingridy.flatten()
#                outgrid=np.column_stack((ingridx,ingridy))  
#                
#                for k in range(0,inrain.shape[0]):
#                    interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
#                    inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
#                    
#            if spin==True and flexspin==True:
#                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(shiftprey[ctr]) : (shiftprey[ctr]+maskheight) , (shiftprex[ctr]) : (shiftprex[ctr]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            elif spin==True and flexspin==False:
#                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            elif spin==False:
#                temprain=inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]
#            else:
#                sys.exit("what else is there?")
#            ctr=ctr+1
#            if intense_data!=False:
#                outrain[j,:]=np.multiply(temprain,multipliermask)
#            else:
#                outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
    
    
#==============================================================================
# LOOP FOR KERNEL BASED STORM TRANSPOSITION
# THIS FINDS THE TRANSPOSITION LOCATION FOR EACH REALIZATION IF YOU ARE USING THE KERNEL-BASED RESAMPLER
# IF I CONFIGURE THE SCRIPT SO THE USER CAN PROVIDE A CUSTOM RESAMPLING SCHEME, THIS WOULD PROBABLY WORK FOR THAT AS WELL
#==============================================================================    
# LEGACY (commented out, not used): weavekernel; relied on scipy.weave, which no longer exists. Candidate for removal.
#def weavekernel(rndloc,cumkernel):
#    nlocs=len(rndloc)
#    nrows=cumkernel.shape[0]
#    ncols=cumkernel.shape[1]
#    tempx=np.empty((len(rndloc)),dtype="int32")
#    tempy=np.empty((len(rndloc)),dtype="int32")
#    code= """
#        #include <stdio.h>
#        int i,x,y,brklp;
#        double prevprob;
#        for (i=0;i<nlocs;i++) {
#            prevprob=0.0;
#            brklp=0;
#            for (y=0; y<nrows; y++) {
#                for (x=0; x<ncols; x++) {
#                    if ( (rndloc(i)<=cumkernel(y,x)) && (rndloc(i)>prevprob) ) {
#                        tempx(i)=x;
#                        tempy(i)=y;
#                        prevprob=cumkernel(y,x);
#                        brklp=1;
#                        break;
#                    }                     
#                }
#                if (brklp==1) {
#                    break;                    
#                }                         
#            }   
#        }
#    """
#    vars=['rndloc','cumkernel','nlocs','nrows','ncols','tempx','tempy']
#    sp.weave.inline(code,vars,type_converters=converters.blitz,compiler='gcc')
#    return tempx,tempy
    
    
def pykernel(rndloc,cumkernel):
    """
    Pure-Python inverse-CDF sampling of transposition locations from a 2D
    cumulative probability map. See ``numbakernel`` for the algorithm.

    Parameters
    ----------
    rndloc : np.ndarray of float, shape (n,)
        Uniform(0, 1) random numbers.
    cumkernel : np.ndarray, shape (nrows, ncols)
        Cumulative probability map (row-major cumsum; cells outside the domain
        are set to a large value such as 100).

    Returns
    -------
    tempx, tempy : np.ndarray of int32, shape (n,)

    Notes
    -----
    Status: not currently called.
    """
    nlocs=len(rndloc)
    ncols=cumkernel.shape[1]
    tempx=np.empty((len(rndloc)),dtype="int32")
    tempy=np.empty((len(rndloc)),dtype="int32")
    flatkern=np.append(0.,cumkernel.flatten())
    
    for i in range(0,nlocs):
        x=rndloc[i]-flatkern
        x[np.less(x,0.)]=1000.
        whereind = np.argmin(x)
        y=whereind//ncols
        x=whereind-y*ncols        
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy

@jit 
def numbakernel(rndloc,cumkernel,tempx,tempy,ncols):
    """
    Inverse-CDF sampling of transposition locations from a 2D cumulative
    probability map (Numba).

    ``cumkernel`` is flattened (row-major) and a 0 is prepended, so
    ``flatkern[k]`` is the cumulative probability *before* cell k. For each
    random number r, the cell chosen is the one whose cumulative interval
    ``(flatkern[k], flatkern[k+1]]`` contains r, i.e. the largest k with
    ``flatkern[k] <= r``. That flat index is converted back to (y, x).

    Parameters
    ----------
    rndloc : np.ndarray of float, shape (n,)
        Uniform(0, 1) random numbers.
    cumkernel : np.ndarray, shape (nrows, ncols)
        Cumulative probability map.
    tempx, tempy : np.ndarray of int32, shape (n,)
        Output arrays (filled in place).
    ncols : int
        Number of columns in ``cumkernel``.

    Returns
    -------
    tempx, tempy : np.ndarray of int32
        Column and row index of each sampled location.
    """
    nlocs=len(rndloc)
    #ncols=xdim
    flatkern=np.append(0.,cumkernel.flatten())
    #x=np.zeros_like(rndloc,dtype='float64')
    for i in np.arange(0,nlocs):
        x=rndloc[i]-flatkern
        x[np.less(x,0.)]=10.
        whereind=np.argmin(x)
        y=whereind//ncols
        x=whereind-y*ncols 
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy


@jit 
def numbakernel_fast(rndloc,cumkernel,tempx,tempy,ncols):
    """
    Wrapper around ``kernelloop`` for inverse-CDF sampling of transposition
    locations; see ``numbakernel`` for the algorithm.

    Parameters
    ----------
    rndloc : np.ndarray of float, shape (n,)
    cumkernel : np.ndarray, shape (nrows, ncols)
    tempx, tempy : np.ndarray of int32, shape (n,)
        Output arrays.
    ncols : int
        Ignored; recomputed from ``cumkernel.shape[1]``.

    Returns
    -------
    tempx, tempy : np.ndarray of int32

    Notes
    -----
    Status: used by RainyDay_Py3.py for TRANSPOSITION = "nonuniform" (that path
    currently exits before reaching this call).
    """
    nlocs=int32(len(rndloc))
    ncols=int32(cumkernel.shape[1])
    flatkern=np.append(0.,cumkernel.flatten()) 
    return kernelloop(nlocs,rndloc,flatkern,ncols,tempx,tempy)

#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def kernelloop(nlocs,rndloc,flatkern,ncols,tempx,tempy):
    """
    Numba inner loop for ``numbakernel_fast``; see ``numbakernel``.

    Parameters
    ----------
    nlocs : int
    rndloc : np.ndarray of float, shape (nlocs,)
    flatkern : np.ndarray
        Flattened cumulative kernel with a leading 0.
    ncols : int
    tempx, tempy : np.ndarray of int32
        Output arrays (filled in place).

    Returns
    -------
    tempx, tempy : np.ndarray of int32
    """
    for i in prange(nlocs):
        diff=rndloc[i]-flatkern
        diff[np.less(diff,0.)]=10.
        whereind=np.argmin(diff)
        y=whereind//ncols
        x=whereind-y*ncols 
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy



#==============================================================================
# FIND THE BOUNDARY INDICIES AND COORDINATES FOR THE USER-DEFINED SUBAREA
# NOTE THAT subind ARE THE MATRIX INDICIES OF THE SUBBOX, STARTING FROM UPPER LEFT CORNER OF DOMAIN AS (0,0)
# NOTE THAT subcoord ARE THE COORDINATES OF THE OUTSIDE BORDER OF THE SUBBOX
# THEREFORE THE DISTANCE FROM THE WESTERN (SOUTHERN) BOUNDARY TO THE EASTERN (NORTHERN) BOUNDARY IS NCOLS (NROWS) +1 TIMES THE EAST-WEST (NORTH-SOUTH) RESOLUTION
#============================================================================== 


def findsubbox(inarea,variables,fname):
    """
    Find the grid subset of a NetCDF rainfall file that covers the
    transposition domain.

    Longitudes above 180 are converted to the -180..180 convention before
    selecting.

    Parameters
    ----------
    inarea : array-like, [lon_min, lon_max, lat_min, lat_max]
        Transposition domain bounds.
    variables : dict
        Names of the rainfall, latitude and longitude variables, in that order
        (the VARIABLES entry of the parameter file).
    fname : str
        Path to one input rainfall NetCDF file.

    Returns
    -------
    outextent : np.ndarray, [lon_first, lon_last, lat_first, lat_last]
        Coordinates of the first and last selected grid points.
    outdim : np.ndarray of int, [nlat, nlon]
        Size of the subset.
    lat, lon : xarray.DataArray
        Latitudes and longitudes of the subset.
    indices : np.ndarray of int, [lat_i0, lat_i1, lon_i0, lon_i1]
        Inclusive index bounds of the subset in the full file.

    Notes
    -----
    The ``slice(latmin, latmax)`` selection assumes latitude is stored in
    ascending order. Status: used by RainyDay_Py3.py.
    """
    outextent = np.empty([4])
    outdim=np.empty([2], dtype= 'int')
    infile=xr.open_dataset(fname)
    latmin,latmax,longmin,longmax = inarea[2],inarea[3],inarea[0],inarea[1]
    rain_name,lat_name,lon_name = variables.values()
    if max(infile[lon_name].values) > 180: # convert from positive degrees west to negative degrees west
        infile[lon_name] = infile[lon_name] - 360 
        # inarea[:2] += 360
        # latmin,latmax,longmin,longmax = inarea[2],inarea[3],inarea[0],inarea[1]
    outrain=infile[rain_name].sel(**{lat_name:slice(latmin,latmax)},\
                                              **{lon_name:slice(longmin,longmax)})
    outextent[2], outextent[3],outextent[0], outextent[1]=outrain[lat_name][0],outrain[lat_name][-1],\
                                outrain[lon_name][0], outrain[lon_name][-1]       
    outdim[0], outdim[1] = len(outrain[lat_name]), len(outrain[lon_name])
    lat = infile[lat_name]; lon = infile[lon_name] 
    min_latidx = np.where((lat >= latmin) & (lat <= latmax))[0][0]
    max_latidx = np.where((lat >= latmin) & (lat <= latmax))[0][-1]
    min_lonidx = np.where((lon >= longmin) & (lon <= longmax))[0][0]
    max_lonidx = np.where((lon >= longmin) & (lon <= longmax))[0][-1]
    indices = np.array([min_latidx, max_latidx, min_lonidx, max_lonidx])
    infile.close()
    return outextent, outdim, outrain[lat_name], outrain[lon_name], indices
    
    
    

#==============================================================================
# THIS RETURNS A LOGICAL GRID THAT CAN THEN BE APPLIED TO THE GLOBAL GRID TO EXTRACT
# A USER-DEFINED SUBGRID
# THIS HELPS TO KEEP ARRAY SIZES SMALL
#==============================================================================
def creategrids(rainprop):
    """
    Build boolean masks that select the user-defined subgrid from the global
    grid of the input dataset.

    Parameters
    ----------
    rainprop : GriddedRainProperties
        Uses ``dimensions`` and ``subind``.

    Returns
    -------
    outgrid : np.ndarray of bool, shape (ny_global, nx_global)
        True inside the subgrid.
    subindx, subindy : np.ndarray of bool
        1D selectors for columns and rows.

    Notes
    -----
    LEGACY. Status: not currently called.
    """
    globrangex=np.arange(0,rainprop.dimensions[1],1)
    globrangey=np.arange(0,rainprop.dimensions[0],1)
    subrangex=np.arange(rainprop.subind[0],rainprop.subind[1]+1,1)
    subrangey=np.arange(rainprop.subind[3],rainprop.subind[2]+1,1)
    subindx=np.logical_and(globrangex>=subrangex[0],globrangex<=subrangex[-1])
    subindy=np.logical_and(globrangey>=subrangey[0],globrangey<=subrangey[-1])
    gx,gy=np.meshgrid(subindx,subindy)
    outgrid=np.logical_and(gx==True,gy==True)
    return outgrid,subindx,subindy


#==============================================================================
# FUNCTION TO CREATE A MASK ACCORDING TO A USER-DEFINED POLYGON SHAPEFILE AND PROJECTION
#==============================================================================

# edited 9/14/2026 by BLF... have repurposed code from SLAM to remove NAN precip from shapefile masks. 
def rastermask(shpname,rainprop,masktype='simple',dissolve=True,ngenfile=False,precipfile=None,variables=None):            
    """
    Rasterize a polygon shapefile onto the rainfall grid to create a watershed
    or transposition-domain mask.

    Edited Sept 2026 by BLF (adapted from SLAM) so that cells where the input
    precipitation data are invalid are set to 0.

    Parameters
    ----------
    shpname : str
        Path to a polygon shapefile in geographic WGS84 coordinates.
    rainprop : GriddedRainProperties
        Uses ``subextent``, ``subdimensions`` and ``spatialres``.
    masktype : {'simple', 'fraction'}, optional
        'simple': 1 for any cell touched by the polygon, 0 otherwise.
        'fraction': approximate fraction (0-1) of each cell covered by the
        polygon, found by rasterizing on a grid 10x finer and block-averaging.
    dissolve : bool, optional
        Unused.
    ngenfile : bool, optional
        Placeholder for NextGen hydrofabric support (not implemented; exits).
    precipfile : str, optional
        Rainfall NetCDF file. If given (with ``variables``), cells whose
        rainfall is negative or non-finite at any time step are set to 0.
    variables : dict, optional
        Rainfall/latitude/longitude variable names.

    Returns
    -------
    np.ndarray of float32, shape (nlat, nlon)
        Mask in north-up orientation. The caller flips it (``np.flipud``) to
        match the south-up orientation of the rainfall arrays.

    Notes
    -----
    All polygons in the shapefile are used. Inside the function ``xdim`` holds
    the number of rows and ``ydim`` the number of columns (the names are
    swapped relative to their meaning). Status: used by RainyDay_Py3.py.
    """
    bndcoords=np.array(rainprop.subextent)
    
    xdim=rainprop.subdimensions[0]  
    ydim=rainprop.subdimensions[1] 
    
    if ngenfile:
        sys.exit("this isn't ready yet")
        project = pyproj.Transformer.from_proj(pyproj.Proj(init='epsg:4326'),pyproj.Proj(init='epsg:5070'))

    # this appears to work even if the shapefile has multiple polygons... it seems to just take the outline
    with fiona.open(shpname, "r") as shapefile:
        shapes = [shape(feature["geometry"]) for feature in shapefile]
        
        # trouble figuring out how to reproject a geojson file to WGS84
        #temp=shape(shapefile)
        #shapes=[]
        #for feature in shapefile:
        #    if ngenfile:
        #        temp=shape(shape(feature["geometry"]))
        #        t1=transform(project.transform, temp)
        #    else:
        #        shapes.append(shape(feature["geometry"]))
            
    #figure out how to make 0's for Nan is precip
    
    if masktype=='simple':
        print('creating simple mask (0s and 1s)')
        trans = from_origin(bndcoords[0], bndcoords[3], rainprop.spatialres[0], rainprop.spatialres[1])
        rastertemplate=np.ones((ydim,xdim),dtype='float32')
        
        memfile = MemoryFile()
        rastermask = memfile.open(driver='GTiff',
                                 height = rastertemplate.shape[1], width = rastertemplate.shape[0],
                                 count=1, dtype=str(rastertemplate.dtype),
                                 crs='+proj=longlat +datum=WGS84 +no_defs',
                                 transform=trans)
        rastermask.write(rastertemplate,1)
        simplemask, out_transform = mask(rastermask, shapes, crop=False,all_touched=True)
        rastertemplate=simplemask[0,:]

    elif masktype=="fraction":
        print('creating fractional mask (range from 0.0-1.0)')
        n=10
        trans = from_origin(bndcoords[0], bndcoords[3], rainprop.spatialres[0]/np.float32(n), rainprop.spatialres[1]/np.float32(n))
        rastertemplate=np.ones((ydim,xdim),dtype='float32')

        memfile = MemoryFile()
        rastermask = memfile.open(driver='GTiff',
                                 height = n*rastertemplate.shape[1], width = n*rastertemplate.shape[0],
                                 count=1, dtype=str(rastertemplate.dtype),
                                 crs='+proj=longlat +datum=WGS84 +no_defs',
                                 transform=trans)
        rastermask.write(rastertemplate,1)
        simplemask, out_transform = mask(rastermask, shapes, crop=False,all_touched=True)
        rastertemplate=simplemask[0,:]
        
        from scipy.signal import convolve2d
        
        kernel = np.ones((n, n))
        convolved = convolve2d(rastertemplate, kernel, mode='valid')
        rastertemplate=convolved[::n, ::n] / n /n 
        
    else:
        sys.exit("You entered an incorrect mask type, options are 'simple' or 'fraction'")
    #delete('temp9999.tif')   

    # Zero out basin cells where the input precip data is invalid 
    if precipfile is not None and variables is not None:
        var_name,lat_name,lon_name = variables.values()
        ds = xr.open_dataset(precipfile)
        if max(ds[lon_name].values) > 180:
            ds[lon_name] = ds[lon_name] - 360
        precip = ds[var_name].sel(**{lat_name:slice(rainprop.subextent[2],rainprop.subextent[3])},
                                   **{lon_name:slice(rainprop.subextent[0],rainprop.subextent[1])}).values
        ds.close()
        valid = np.all((precip >= 0.) & np.isfinite(precip), axis=0)
        valid = np.flipud(valid)    # data is south-up; this mask is north-up until the caller flips it
        if valid.shape != rastertemplate.shape:
            sys.exit("rastermask: precip validity grid and mask are different sizes")
        rastertemplate = np.where(valid, rastertemplate, 0.)

    return rastertemplate   



#==============================================================================
# WRITE SCENARIOS TO NETCDF ONE REALIZATION AT A TIME
#==============================================================================
def writerealization(scenarioname,rlz,nrealizations,writename,outrain,writemax,writestorm,writeperiod,writex,writey,writetimes,latrange,lonrange,whichorigstorm):
    # SAVE outrain AS NETCDF FILE
    """
    Write one realization of annual-maximum SST scenarios to a single NetCDF
    file (one storm per synthetic year).

    Parameters
    ----------
    scenarioname : str
    rlz : int
        Realization index (0-based).
    nrealizations : int
    writename : str
        Output file path.
    outrain : np.ndarray, shape (nyears, nt, nlat, nlon)
        Rainfall rates (mm/hr). NaNs are replaced with -9999 in place.
    writemax : np.ndarray, shape (nyears,)
        Basin-average storm totals (mm).
    writestorm, writeperiod, writex, writey : np.ndarray, shape (nyears,)
        Storm rank, return period, and transposition indices.
    writetimes : np.ndarray, shape (nyears, nt)
    latrange, lonrange : np.ndarray
    whichorigstorm : np.ndarray, shape (nyears,)
        Parent storm number from the catalog.

    Notes
    -----
    LEGACY: replaced in Aug 2023 by one-file-per-scenario output
    (``writescenariofile``). Status: not currently called.
    """
    dataset=Dataset(writename, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',len(latrange))
    outlons=dataset.createDimension('longitude',len(lonrange))
    time=dataset.createDimension('time',writetimes.shape[1])
    nyears=dataset.createDimension('nyears',len(writeperiod))

    # create variables
    times=dataset.createVariable('time',np.float64, ('nyears','time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude'))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude'))
    rainrate=dataset.createVariable('precrate',np.float32,('nyears','time','latitude','longitude'),zlib=True,complevel=4,least_significant_digit=1) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32,('nyears')) 
    xlocation=dataset.createVariable('xlocation',np.int16,('nyears')) 
    ylocation=dataset.createVariable('ylocation',np.int16,('nyears')) 
    returnperiod=dataset.createVariable('returnperiod',np.float32,('nyears')) 
    stormnumber=dataset.createVariable('stormnumber',np.int16,('nyears'))
    original_stormnumber=dataset.createVariable('original_stormnumber',np.int16,('nyears'))
    #stormtimes=dataset.createVariable('stormtimes',np.float64,('nyears'))          
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    rainrate.units = 'mm hr^-1'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    basinrainfall.units='mm'
    xlocation.units='dimensionless'
    ylocation.units='dimensionless'
    returnperiod.units='years'
    stormnumber.units='dimensionless'
    original_stormnumber.units='dimensionless'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    rainrate.long_name='precipitation rate'
    basinrainfall.long_name='storm total basin averaged precipitation'
    xlocation.long_name='x index of storm'
    ylocation.long_name='y index of storm'
    returnperiod.long_name='return period of storm total rainfall'
    stormnumber.long_name='storm rank, from 1 to NYEARS'
    original_stormnumber.long_name='parent storm number from storm catalog'
    
    
    
    # Global Attributes
    dataset.description = 'SST Rainfall Scenarios Realization: '+str(rlz+1)+' of '+str(nrealizations)
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'Realization '+str(rlz)+' from scenario '+scenarioname
    dataset.missing='-9999.'

    # fill the netcdf file
    latitudes[:]=latrange[::-1]
    longitudes[:]=lonrange
    outrain[np.isnan(outrain)]=-9999.
    rainrate[:]=outrain[:,:,::-1,:] 
    basinrainfall[:]=writemax
    times[:]=writetimes
    xlocation[:]=writex
    ylocation[:]=writey
    stormnumber[:]=writestorm
    returnperiod[:]=writeperiod
    original_stormnumber[:]=whichorigstorm
    #stormtimes[:]=writetimes
    
    dataset.close()
    
    
    
#==============================================================================
# WRITE SCENARIOS TO NETCDF ONE REALIZATION AT A TIME-USING THE NPERYEAR OPTION
#==============================================================================    
def writerealization_nperyear(scenarioname,writename,rlz,nperyear,nrealizations,outrain_large,outtime_large,subrangelat,subrangelon,rlz_order,nsimulations):
    # SAVE outrain AS NETCDF FILE
    #filename=writename+'_SSTrealization'+str(rlz+1)+'_Top'+str(nperyear)+'.nc'
    """
    Write one realization of scenarios with several (``nperyear``) storms per
    synthetic year to a NetCDF file.

    Parameters
    ----------
    scenarioname : str
        Used as the output path (``writename`` is not used for the path).
    writename : str
    rlz, nperyear, nrealizations, nsimulations : int
    outrain_large : np.ndarray, shape (nsimulations, nperyear, nt, nlat, nlon)
    outtime_large : np.ndarray, shape (nsimulations, nperyear, nt)
    subrangelat, subrangelon : np.ndarray
    rlz_order : np.ndarray
        Ranking of storms within each year (negative for no storm).

    Notes
    -----
    LEGACY (adapted from Guo Yu's version). Status: not currently called.
    """
    dataset=Dataset(scenarioname, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',len(subrangelat))
    outlons=dataset.createDimension('longitude',len(subrangelon))
    time=dataset.createDimension('time',outtime_large.shape[2])
    nyears=dataset.createDimension('nyears',nsimulations)
    topN=dataset.createDimension('nperyear',nperyear)

    # create variables
    times=dataset.createVariable('time',np.float64, ('nyears','nperyear','time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude'))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude'))
    rainrate=dataset.createVariable('precrate',np.float32,('nyears','nperyear','time','latitude','longitude'),zlib=True,complevel=4,least_significant_digit=1) 
    top_event=dataset.createVariable('top_event',np.int16, ('nyears'))
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    rainrate.units = 'mm hr^-1'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    top_event.units='dimensionless'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    rainrate.long_name='precipitation rate'
    top_event.long_name='largest event (storm number) of synthetic year'
    
    
    # Global Attributes
    dataset.description = 'NPERYEAR-type SST Rainfall Scenarios Realization: '+str(rlz+1)+' of '+str(nrealizations)
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'Realization '+str(rlz)+' from scenario '+scenarioname
    dataset.missing='-9999.'

    # fill the netcdf file
    latitudes[:]=subrangelat[::-1]   # need to check this!
    longitudes[:]=subrangelon
    outrain_large[np.isnan(outrain_large)]=-9999.
    rainrate[:]=outrain_large[:,:,:,::-1,:]
    times[:]=outtime_large
    n_evnet = np.nansum(rlz_order>=0,axis=0)
    n_evnet[n_evnet>=nperyear]=nperyear
    top_event[:]= n_evnet
    dataset.close()
    
#==============================================================================
# WRITE The maximized storm
#==============================================================================
def writemaximized(scenarioname,writename,outrain,writemax,write_ts,writex,writey,writetimes,latrange,lonrange):
    # SAVE outrain AS NETCDF FILE
    """
    Write a single "maximized" storm (e.g., the largest transposed storm) to a
    NetCDF file.

    Parameters
    ----------
    scenarioname : str
    writename : str
        Output path.
    outrain : np.ndarray, shape (nt, nlat, nlon)
    writemax : float
        Basin-average storm total (mm).
    write_ts : unused
    writex, writey : int
        Transposition indices.
    writetimes : np.ndarray, shape (nt,)
    latrange, lonrange : np.ndarray

    Notes
    -----
    Status: not currently called.
    """
    dataset=Dataset(writename, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',len(latrange))
    outlons=dataset.createDimension('longitude',len(lonrange))
    time=dataset.createDimension('time',len(writetimes))

    # create variables
    times=dataset.createVariable('time',np.float64, ('time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude'))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude'))
    rainrate=dataset.createVariable('precrate',np.float32,('time','latitude','longitude'),zlib=True,complevel=4,least_significant_digit=1) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32) 
    xlocation=dataset.createVariable('xlocation',np.int16) 
    ylocation=dataset.createVariable('ylocation',np.int16) 
    #stormtimes=dataset.createVariable('stormtimes',np.float64,('nyears'))          
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    rainrate.units = 'mm hr^-1'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    xlocation.units='dimensionless'
    ylocation.units='dimensionless'
    basinrainfall.units='mm'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    rainrate.long_name='precipitation rate'
    basinrainfall.long_name='storm total basin averaged precipitation'
    xlocation.long_name='x index of storm'
    ylocation.long_name='y index of storm'
    
    # Global Attributes
    dataset.description = 'SST Rainfall Maximum Storm'
    dataset.missing='-9999.'
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = "RainyDay Y'all!"
    

    
    #print dataset.description
    #print dataset.history
    
    # fill the netcdf file
    latitudes[:]=latrange[::-1]
    longitudes[:]=lonrange
    outrain[np.isnan(outrain)]=-9999.
    rainrate[:]=outrain[:,::-1,:]
    basinrainfall[:]=writemax
    times[:]=writetimes
    xlocation[:]=writex
    ylocation[:]=writey
    
    dataset.close()
        
        

#==============================================================================
# READ RAINFALL FILE FROM NETCDF (ONLY FOR RAINYDAY NETCDF-FORMATTED DAILY FILES!
#==============================================================================

# LEGACY (commented out, not used): previous readnetcdf using an 'index' argument; superseded by readnetcdf below. Candidate for removal.
# def readnetcdf(rfile,variables,index = None,dropvars=False):
#     """
#     Used to trim the dataset with defined inbounds or transposition domain

#     Parameters
#     ----------
#     rfile : Dataset file path ('.nc' file)
#         This is the path to the dataset
#     variables : TYPE
#         DESCRIPTION.
#     inbounds : TYPE, optional
#         DESCRIPTION. The default is False.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     rain_name,lat_name,lon_name = variables.values()
#     if index is not None:
#         infile = Dataset(rfile, mode='r')
#         outrain = np.array(infile.variables[rain_name][:,index[0]:index[1]+1, index[3]:index[3]+1])
#         outtime = np.array(infile.variables['time'][:], dtype='datetime64[m]')
#         infile.close()
#     else:    
#         infile = xr.open_dataset(rfile, drop_variables=dropvars) if dropvars else xr.open_dataset(rfile)  # added DBW 07282023 to avoid reading in unnecessary variables
        
#         if max(infile[lon_name].values) > 180: # convert from positive degrees west to negative degrees west
#             infile[lon_name] = infile[lon_name] - 360 
#         # if np.any(inbounds!=False):
#         #     latmin,latmax,longmin,longmax = inbounds[2],inbounds[3],inbounds[0],inbounds[1]
#         #     outrain=infile[rain_name].sel(**{lat_name:slice(latmin,latmax)},\
#         #                                               **{lon_name:slice(longmin,longmax)})
#         # else:
#         outrain=infile[rain_name]
#         outlatitude=outrain[lat_name]
#         outlongitude=outrain[lat_name] 
#         outtime=np.array(infile['time'], dtype='datetime64[m]')
#         infile.close()
#     if index is not None:
#         return outrain,outtime
        
#     else:
#         return np.array(outrain),outtime,np.array(outlatitude),np.array(outlongitude)
def find_indices(rfile,inarea,variables):
    """
    Return the inclusive index bounds of a lat/lon box within a NetCDF file.

    Parameters
    ----------
    rfile : str
        Path to a rainfall NetCDF file.
    inarea : array-like, [lon_min, lon_max, lat_min, lat_max]
    variables : dict
        Rainfall/latitude/longitude variable names.

    Returns
    -------
    list of int, [lat_i0, lat_i1, lon_i0, lon_i1]
        Used with ``readnetcdf(..., idxes=...)`` to read only the subset.

    Notes
    -----
    The dataset is not explicitly closed. Status: used by RainyDay_Py3.py.
    """
    ds = Dataset(rfile, 'r')
    rain_name, lat_name, lon_name = variables.values()

    # Extract the latitude, longitude, and variable data
    lat = ds.variables[lat_name][:]
    lon = ds.variables[lon_name][:]
    if max(ds.variables[lon_name]) > 180: # convert from positive degrees west to negative degrees west
        lon = lon - 360
    # Specify the latitude and longitude bounds for clipping
    lat_min, lat_max = inarea[2],inarea[3]  # Example latitude range
    lon_min, lon_max = inarea[0],inarea[1]  # Example longitude range

    # Find the indices corresponding to the specified bounds
    lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    return [lat_inds.min(),lat_inds.max(),lon_inds.min(), lon_inds.max()]

def readnetcdf(rfile,variables,idxes=False,dropvars=False,setup=False,calendar=False,time_units=False,):
    """
    Read rainfall and time from one input NetCDF file (typically one day).

    Two modes:

    * ``idxes`` given: read only the index subset
      ``[:, lat_i0:lat_i1+1, lon_i0:lon_i1+1]`` with netCDF4 (fast path used in
      the catalog-creation loop). Times are decoded with ``calendar``.
    * ``idxes`` not given: read the whole file with xarray (dropping
      ``dropvars``) and convert longitudes above 180 to -180..180.

    Parameters
    ----------
    rfile : str
        Path to the NetCDF file.
    variables : dict
        Rainfall, latitude and longitude variable names, in that order.
    idxes : list of int, optional
        [lat_i0, lat_i1, lon_i0, lon_i1] from ``find_indices``.
    dropvars : list of str, optional
        Variables to skip when reading with xarray.
    setup : bool, optional
        If True, also return latitude, longitude and the raw netCDF time
        variable (only valid when ``idxes`` is not given).
    calendar : str, optional
        Calendar used to decode times in the ``idxes`` path.
    time_units : optional
        Unused (kept for a commented-out Dask path).

    Returns
    -------
    outrain : array, shape (nt, nlat, nlon)
        Rainfall rate (mm/hr). A masked array in the ``idxes`` path, an xarray
        DataArray otherwise (or ndarray when ``setup``).
    outtime : np.ndarray of datetime64[m], shape (nt,)
    outlatitude, outlongitude : np.ndarray
        Only when ``setup`` is True.
    nctime : netCDF4.Variable
        Only when ``setup`` is True (used to read units and calendar).

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    # infile = xr.open_dataset(rfile, drop_variables=dropvars,chunks='auto').load() if dropvars else xr.open_dataset(rfile).load()  # added DBW 07282023 to avoid reading in unnecessary variables
    rain_name,lat_name,lon_name = variables.values()
    if np.any(idxes!=False):
        # latmin,latmax,longmin,longmax = inbounds[2],inbounds[3],inbounds[0],inbounds[1]
        # outrain=infile[rain_name].sel(**{lat_name:slice(latmin,latmax)},\
        #                                           **{lon_name:slice(longmin,longmax)})

        infile = Dataset(rfile, 'r') ;
        outrain = infile.variables[rain_name][:, idxes[0]:idxes[1]+1, idxes[2]:idxes[3]+1]
        time_var = infile.variables['time'];time_converted = num2date(time_var, units=time_var.units, calendar=calendar)
        outtime = np.array(time_converted, dtype='datetime64[m]')
        infile.close()

        # # --- HERE
        # # Open dataset with Dask-enabled lazy loading
        # # Set the scheduler to use threads (great for I/O-bound tasks like NetCDF reads)
        # dask.config.set(scheduler='threads')
        # ds = xr.open_dataset(rfile,  decode_times=False, chunks={'time': 8, 'latitude': 295, 'longitude': 590})
        #  # Subset the rainfall variable lazily
        # outrain_lazy = ds[rain_name].isel(
        #     latitude=slice(idxes[0], idxes[1]+1),
        #     longitude=slice(idxes[2], idxes[3]+1)
        # )
        
        # # Load data into memory
        # outrain = outrain_lazy.compute().values
        # # Convert time using provided calendar and units
        # time_var = ds['time']
        # time_converted = num2date(time_var.values, units=time_units, calendar=calendar)
        # outtime = np.array(time_converted, dtype='datetime64[m]')
        # ds.close()

    else:
        infile = xr.open_dataset(rfile, drop_variables=dropvars, chunks='auto').load() if dropvars else xr.open_dataset(rfile).load()
        ncfile = Dataset(rfile, 'r') ;
        nctime = ncfile.variables['time']
        if max(infile[lon_name].values) > 180: # convert from positive degrees west to negative degrees west
            infile[lon_name] = infile[lon_name] - 360
        outrain=infile[rain_name]
        outlatitude=outrain[lat_name]
        outlongitude=outrain[lon_name]
        outtime=np.array(infile['time'],dtype='datetime64[m]')
        infile.close()
    
    if setup:
        return np.array(outrain),outtime,np.array(outlatitude),np.array(outlongitude),nctime
    else:
        return  outrain,outtime
  
  
#==============================================================================
# READ RAINFALL FILE FROM NETCDF
#==============================================================================

def readcatalog(rfile) :
    """
    Read one storm file from a RainyDay storm catalog.

    Since Aug 2023 each storm is stored in its own NetCDF file, but every file
    also carries the catalog-wide arrays (storm totals, locations, and times
    for all storms), so reading any one file (usually the last) recovers the
    full catalog summary.

    Parameters
    ----------
    rfile : str
        Path to a storm catalog NetCDF file.

    Returns
    -------
    outrain : xarray.DataArray, shape (nt, nlat, nlon)
        Rainfall rate (mm/hr) for this storm; -9999 marks missing data.
    stormtime : np.ndarray of datetime64[m], shape (nt,)
        Times of this storm.
    outlatitude, outlongitude : xarray.DataArray
    outlocx, outlocy : np.ndarray of int, shape (nstorms,)
        Upper-left x/y index of the watershed rectangle at each storm's
        maximum, for all storms in the catalog.
    outmax : np.ndarray, shape (nstorms,)
        Basin-average storm total (mm) for all storms.
    outmask : xarray.DataArray, shape (nlat, nlon)
        Watershed mask (``catmask``).
    domainmask : np.ndarray, shape (nlat, nlon)
        Transposition domain mask.
    cattime : np.ndarray of datetime64[m], shape (nstorms, nt)
        Times of all storms.
    timeresolution : int
        Temporal resolution in minutes (only returned if stored in the file).

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    # infile=xr.open_dataset(rfile, engine='h5netcdf')
    infile=xr.open_dataset(rfile)
    outrain=infile['rain']
    outlatitude=infile['latitude']
    outmask=infile['gridmask']
    domainmask=np.array(infile['domainmask'])
    stormtime=np.array(infile['time'],dtype='datetime64[m]')
    outlongitude=infile['longitude']
    outlocx=np.array(infile['xlocation'])
    outlocy=np.array(infile['ylocation'])
    outmax=np.array(infile['basinrainfall'])
    cattime = np.array(infile['cattime'],dtype='datetime64[m]')

    try:
        timeresolution=np.int16(infile.timeresolution)
        resexists=True
    except:
        resexists=False
    infile.close()
    
    if resexists:
        return outrain,stormtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,cattime,timeresolution
    else:
        return outrain,stormtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,cattime

def check_time(datetime_obj):
    """
    Check whether a timestamp falls at 00:00 or 12:00.

    Used when writing the storm catalog to decide which daily input file a
    given time step belongs to. If the first time step of an input file is at
    00:00 (period-beginning convention), a time belongs to the file for its own
    date; otherwise (period-ending convention, first step e.g. 01:00), the
    00:00 step belongs to the previous day's file.

    Parameters
    ----------
    datetime_obj : numpy.datetime64
        Datetime object to check the time component of the object

    Returns
    -------
    bool
        True if the time part is '00:00(:00)' or '12:00(:00)', else False.

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    time_str = str(datetime_obj).split('T')[1][:8]  # Extract the time part
    return time_str == '00:00' or time_str == '12:00:00' or time_str == '00:00:00' or time_str =='12:00'

#==============================================================================
# WRITE RAINFALL FILE TO NETCDF
#==============================================================================


#RainyDay.writecatalog(scenarioname,catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,catmask,parameterfile,domainmask,timeresolution=rainprop.timeres)   
def writecatalog(scenarioname, catrain, catmax, catx, caty, cattime, latrange, lonrange, catalogname, gridmask,
                 parameterfile, dmask, nstorms, duration,storm_num,timeresolution=False):
    """
    Write one storm of the storm catalog to a NetCDF file.

    Each file contains the rainfall for that storm plus catalog-wide summary
    arrays (storm totals, locations, and times for all storms), the watershed
    and domain masks, the temporal resolution, and the full contents of the
    JSON parameter file in the ``description`` attribute.

    Parameters
    ----------
    scenarioname : str
    catrain : np.ndarray, shape (nt, nlat, nlon)
        Rainfall rate for this storm (NaNs are set to -9999 in place).
    catmax : np.ndarray, shape (nstorms,)
        Basin-average storm totals (mm).
    catx, caty : np.ndarray of int, shape (nstorms,)
        Upper-left indices of the watershed rectangle at each storm maximum.
    cattime : np.ndarray of datetime64, shape (nstorms, nt)
    latrange, lonrange : xarray.DataArray
    catalogname : str
        Output file path.
    gridmask : np.ndarray, shape (nlat, nlon)
        Watershed mask.
    parameterfile : str
        Path to the JSON parameter file.
    dmask : np.ndarray, shape (nlat, nlon)
        Transposition domain mask.
    nstorms : int
    duration : float
        Unused.
    storm_num : int
        0-based index of this storm (selects its row of ``cattime``).
    timeresolution : int, optional
        Temporal resolution in minutes.

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    with open(parameterfile,'r') as f:
        params = json.loads(f.read())
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes_units,longitudes_units = 'degrees_north', 'degrees_east'
    rainrate_units,basinrainfall_units = 'mm hr^-1', 'mm'
    times_units, times_calendar = 'minutes since 1970-01-01 00:00.0' , 'gregorian'

    # Variable Names
    times_name = 'time'                     ## change here
    latitudes_name,longitudes_name  = 'latitude', 'longitude'
    rainrate_name, basinrainfall_name = 'precipitation rate', 'storm total basin averaged precipitation'
    xlocation_name, ylocation_name = 'x index of storm', 'y index of storm'
    gmask_name, domainmask_name = 'mask for Aw (control volume)', 'mask for transposition domain'
    if timeresolution!=False:
        timeresolution=timeresolution
    else:
        timeresolution = "None"
    catrain[np.isnan(catrain)] = -9999.

    history, missing = 'Created ' + str(datetime.now()), '-9999.'
    source = 'RainyDay Storm Catalog for scenario ' + scenarioname + '. See description for JSON file contents.'

    data_vars = dict(
                     rain = (("time","latitude", "longitude"),catrain,{'units': rainrate_units, 'long_name': rainrate_name}),
                    # rain = (("time","latitude", "longitude"),catrain[:, ::-1, :],{'units': rainrate_units, 'long_name': rainrate_name}),
                     basinrainfall = (("storm_dim"),catmax.reshape(nstorms),{'units': basinrainfall_units, 'long_name': basinrainfall_name}),
                     xlocation = (("storm_dim"),catx.reshape(nstorms),{'units': 'dimensionless', 'long_name': xlocation_name}),
                     ylocation = (("storm_dim"),caty.reshape(nstorms),{'units': 'dimensionless', 'long_name': ylocation_name}),
                     cattime = (("storm_dim","time"), cattime),
                     gridmask= (("latitude", "longitude"), gridmask,{'units': 'dimensionless', 'long_name': gmask_name}),
                     domainmask = (("latitude", "longitude"),dmask,{'units': 'dimensionless', 'long_name': domainmask_name}),
                     # gridmask= (("latitude", "longitude"), gridmask[::-1, :],{'units': 'dimensionless', 'long_name': gmask_name}),
                     # domainmask = (("latitude", "longitude"),dmask[::-1, :],{'units': 'dimensionless', 'long_name': domainmask_name}),
                     timeresolution = ((), timeresolution) )
    coords = dict(time = ((times_name),cattime[storm_num,:]),
                  longitude = (("longitude"), lonrange.data , {'units': longitudes_units, 'long_name': longitudes_name}),
                  latitude =  (("latitude"), latrange.data, {'units': latitudes_units, 'long_name': latitudes_name}),
                  #latitude =  (("latitude"), latrange[::-1].data, {'units': latitudes_units, 'long_name': latitudes_name}),
                  )


    attrs  = dict(history =history, source =  source, missing = missing, description = str(params),  calendar = times_calendar)
    catalog = xr.Dataset(data_vars = data_vars, coords = coords, attrs = attrs)
    catalog.time.encoding['units'] = "minutes since 1970-01-01 00:00:00"

    catalog.to_netcdf(catalogname)
    catalog.close()


def writeintensityfile(scenarioname,intenserain,filename,latrange,lonrange,intensetime):
    # SAVE outrain AS NETCDF FILE
    """
    Write a gridded "storm intensity" file (top storm totals at each cell).

    Parameters
    ----------
    scenarioname : str
    intenserain : np.ndarray, shape (nstorms, nlat, nlon)
        Storm totals (mm); NaNs set to -9999 in place.
    filename : str
    latrange, lonrange : np.ndarray
    intensetime : np.ndarray, shape (nstorms, nlat, nlon)

    Notes
    -----
    LEGACY (paired with ``readintensityfile``). Status: not currently called.
    """
    dataset=Dataset(filename, 'w', format='NETCDF4')
    
    # create dimensions
    outlats=dataset.createDimension('latitude',intenserain.shape[1])
    outlons=dataset.createDimension('longitude',intenserain.shape[2])
    nstorms=dataset.createDimension('nstorms',intenserain.shape[0])

    # create variables
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude',))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude',))
    stormtotals=dataset.createVariable('stormtotals',np.float32,('nstorms','latitude','longitude',),zlib=True,complevel=4,least_significant_digit=1)
    times=dataset.createVariable('time',np.float64, ('nstorms','latitude','longitude',))

    dataset.Conventions ='CF1.8'
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Intensity File for scenario '+scenarioname
    dataset.description = 'this description should be improved :)!'
    dataset.missing='-9999.'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    stormtotals.long_name='storm total rainfall'

    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    stormtotals.units = 'mm'
    times.units = 'minutes since 1970-01-01 00:00.0'

    # fill the netcdf file
    latitudes[:]=latrange[::-1]
    longitudes[:]=lonrange
    intenserain[np.isnan(intenserain)]=-9999.
    stormtotals[:]=intenserain[:,::-1,:]
    times[:]=intensetime[:,::-1,:]
    dataset.close()
    
    
def readintensityfile(rfile,inbounds=False):
    """
    Read a gridded storm-intensity file for stochastic/deterministic rescaling.

    Parameters
    ----------
    rfile : str
    inbounds : array-like or False, optional
        Index bounds [x0, x1, y1, y0] to subset.

    Returns
    -------
    outrain, outtime, outlat, outlon : np.ndarray

    Notes
    -----
    DISABLED: the function exits immediately pending CF-convention fixes (the
    latitude orientation). Because of this, NORMALIZEDSST = "stochastic" or
    "deterministic" cannot currently run.
    """
    infile=Dataset(rfile,'r')
    sys.exit("need to make sure that all CF-related file formatting issues are solved. This main revolves around flipping the rainfall vertically, and perhaps the latitude array as well.")
    if np.any(inbounds!=False):
        outrain=np.array(infile.variables['stormtotals'][:,inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outtime=np.array(infile.variables['time'][:,inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1],dtype='datetime64[m]')
        outlat=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlon=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])
    else:
        outrain=np.array(infile.variables['stormtotals'][:])
        outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
        outlat=np.array(infile.variables['latitude'][:])
        outlon=np.array(infile.variables['longitude'][:])        
    infile.close()
    return outrain,outtime,outlat,outlon

# =============================================================================
# added Lei 02122025: Read the quantile basemap for rescaling
# =============================================================================
def read_quantilefile(amfile, duration, return_period, mask=True):
    """
    Read a gridded annual-maximum file and compute a design precipitation field
    (empirical quantile) for normalized SST.

    Added by Lei Yan (Feb 2025).

    Parameters
    ----------
    amfile : str
        NetCDF file with variable ``precrate`` (annual maxima) and dimensions
        ``duration``, ``latitude``, ``longitude``.
    duration : int
        Duration (hours) to select; must exist in the file.
    return_period : float
        Return period (years). The quantile used is ``1 - 1/return_period``.
    mask : bool, optional
        If True, crop to ``inbounds`` (see Notes).

    Returns
    -------
    design_values : np.ndarray, shape (nlat, nlon)
    lat, lon : np.ndarray

    Notes
    -----
    The ``mask=True`` branch refers to ``inbounds``, which is not defined in
    this function, so only ``mask=False`` works. Status: used by RainyDay_Py3.py.
    """
    ds = xr.open_dataset(amfile)

    # Check if the specified duration is in the dataset
    if duration not in ds['duration'].values:
        raise ValueError(f"The specified duration {duration} is not in the dataset. Available duration values are: {ds['duration'].values}")

    ds = ds.sel(duration=duration)

    # If region boundaries are specified, crop the data
    if mask:
        lon_min, lon_max, lat_min, lat_max = inbounds
        ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    precrate = ds['precrate'].values
    lat = np.array(ds['latitude'][:])
    lon = np.array(ds['longitude'][:])

    # Calculate the quantile corresponding to the empirical probability
    empirical_quantile = 1 - 1 / return_period
    #design_values = np.quantile(precrate, empirical_quantile, axis=0, method='linear')
    # np.quantile renamed the "interpolation" kwarg to "method" in NumPy 1.22.
    _np_major, _np_minor = (int(x) for x in np.__version__.split('.')[:2])
    if (_np_major, _np_minor) >= (1, 22):
        design_values = np.quantile(precrate, empirical_quantile, axis=0, method='linear')
    else:
        design_values = np.quantile(precrate, empirical_quantile, axis=0, interpolation='linear')
    
    ds.close()
    return design_values, lat, lon

def readmeanfile(rfile,inbounds=False):
    """
    Read a gridded mean storm-total file.

    Notes
    -----
    DISABLED: exits immediately pending CF-convention fixes.
    Status: not currently called.
    """
    infile=Dataset(rfile,'r')
    sys.exit("need to make sure that all CF-related file formatting issues are solved. This main revolves around flipping the rainfall vertically, and perhaps the latitude array as well.")
 
    if np.any(inbounds!=False):
        outrain=np.array(infile.variables['stormtotals'][inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outlat=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlon=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])
    else:
        outrain=np.array(infile.variables['stormtotals'][:])
        outlat=np.array(infile.variables['latitude'][:])
        outlon=np.array(infile.variables['longitude'][:])        
    infile.close()
    return outrain,outlat,outlon


def writedomain(domain,mainpath,latrange,lonrange,parameterfile):
    # SAVE outrain AS NETCDF FILE
    """
    Write a transposition domain mask to NetCDF.

    Notes
    -----
    DISABLED: exits immediately pending CF-convention fixes.
    Status: not currently called.
    """
    sys.exit("need to make sure that all CF-related file formatting issues are solved. This main revolves around flipping the rainfall vertically, and perhaps the latitude array as well.")
 
    dataset=Dataset(mainpath, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',domain.shape[0])
    outlons=dataset.createDimension('longitude',domain.shape[1])

    # create variables
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude',))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude',))
    domainmap=dataset.createVariable('domain',np.float32,('latitude','longitude',))
    
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Transposition Domain Map File'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    domainmap.units = 'dimensionless'
    
    # fill the netcdf file
    latitudes[:]=latrange
    longitudes[:]=lonrange
    domainmap[:]=domain
    
    with open(parameterfile, "r") as myfile:
        params=myfile.read()
    myfile.close
    dataset.description=params
    
    dataset.close()

# =============================================================================
# added Ashar 08162023: To extract numbers from storm files.
# =============================================================================
def extract_storm_number(file_path, catalogname):
    """
    Extracts the storm number from the filename of a storm file, using the catalog name for precise matching.

    Parameters
    ----------
    file_path : string
        File path for the storms .nc files.
    catalogname : string
        The specific prefix (catalog name) to match in the filename.

    Returns
    -------
    integer
        Returns the storm number from the path given in "file_path", or 0 if not found.
    """
    base_name = os.path.basename(file_path)
    pattern = re.escape(catalogname)+'_storm' + r'_(\d+)_\d{8}\.nc$'
    match = re.search(pattern, base_name)
    if match:
        return int(match.group(1))
    return 0
# LEGACY (commented out, not used): previous extract_storm_number filename pattern. Candidate for removal.
# def extract_storm_number(file_path, catalogname):
#     """
    

#     Parameters
#     ----------
#     file_path : string
#         File path for the storms .nc files
#     catalogname : string
#         Name of the storm catalog given in JSON file

#     Returns
#     -------
#     integer
#         returns the storm number from the path given in "file_path".

#     """
#     base_name = os.path.basename(file_path)
#     match = re.search(catalogname +r'(\d+)', base_name)
#     if match:
#         return np.int32(match.group(1))
#     return 0  

def extract_date(file_path, catalogname):
    """
    Extracts the date from the filename of a storm file, using the catalog name for precise matching.

    Parameters
    ----------
    file_path : string
        File path for the storm .nc files.
    catalogname : string
        The specific prefix (catalog name) to match in the filename.

    Returns
    -------
    string
        Returns the date of the storm in the YYYYMMDD format (string), or None if not found.
    """
    base_name = os.path.basename(file_path)
    pattern = re.escape(catalogname) +'_storm'+ r'_\d+_(\d{8})\.nc$'
    match = re.search(pattern, base_name)
    if match:
        return match.group(1)
    return None


# LEGACY (commented out, not used): previous extract_date filename pattern. Candidate for removal.
# def extract_date(file_path, pattern):
#     """
    

#     Parameters
#     ----------
#     file_path : string
#         File path for the storm catalog file
#     pattern : string
#         catalogname gievn in the JSON file

#     Returns
#     -------
#     string
#         returns the date of the storm catalog in the YYYYMMDD format(string)

#     """
#     base_name = os.path.basename(file_path)
#     match = re.search(pattern + r'\d+_(\d{8})\.nc', base_name)
#     if match:
#         return match.group(1)
#     return None
# =============================================================================
# added DBW 08152023: delete existing scenario files recursively before writing new ones
# this was provided by ChatGPT
# =============================================================================
def delete_files_in_directory(directory_path):
    """
    Recursively delete all files under a directory, leaving the (now empty)
    subdirectories in place. Used to clear old scenario files before writing
    new ones.

    Parameters
    ----------
    directory_path : str

    Notes
    -----
    Errors deleting individual files are printed, not raised.
    Status: used by RainyDay_Py3.py.
    """
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                #print(f"Deleted: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
        elif os.path.isdir(item_path):
            delete_files_in_directory(item_path)  # Recursively call the function for subdirectories


# =============================================================================
# added DBW 08142023: writing single storm scenario file using xarray
# modified by Ashar 07/12/2026: added returnperiod and original_stormnumber output variables
# =============================================================================
def writescenariofile(catrain,raintime,rainlocx,rainlocy,name_scenariofile,tstorm,tyear,trealization,maskheight,maskwidth,subrangelat,subrangelon,scenarioname,mask,origstormnumber,scenario_returnperiod):
    # the following line extracts only the transposed rainfall within the area of interest
    #transposedrain=np.multiply(catrain[:,rainlocy[0] : (rainlocy[0]+maskheight), rainlocx[0] : (rainlocx[0]+maskwidth)],mask)
    """
    Write one transposed storm scenario to its own NetCDF file.

    The parent storm's rainfall is cropped to the watershed rectangle at the
    transposition location and written with its times, the transposition
    indices, the scenario return period and the parent storm number.

    Added by DBW (Aug 2023); returnperiod and original_stormnumber added by
    Ashar (July 2026).

    Parameters
    ----------
    catrain : np.ndarray, shape (nt, nlat, nlon)
        Parent storm rainfall rate (mm/hr) over the full domain.
    raintime : np.ndarray of datetime64, shape (nt,)
    rainlocx, rainlocy : np.ndarray of int, shape (1,)
        Upper-left x/y index of the transposed watershed rectangle.
    name_scenariofile : str
        Output file path.
    tstorm, tyear, trealization : int
        Parent storm index, synthetic-year index, and realization index (used
        only in the description string).
    maskheight, maskwidth : int
    subrangelat, subrangelon : np.ndarray
        Coordinates of the watershed rectangle at its original location.
    scenarioname : str
    mask : np.ndarray
        Watershed mask. Currently unused: the multiplication by the mask is
        commented out, so the whole rectangle is written.
    origstormnumber : int
        Parent storm number from the catalog (1-based, from the filename).
    scenario_returnperiod : float
        Return period of this scenario's year rank, or -9999 for the extra
        storms when NPERYEAR > 1.

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    transposedrain=catrain[:,rainlocy[0] : (rainlocy[0]+maskheight), rainlocx[0] : (rainlocx[0]+maskwidth)]

    description_string='RainyDay storm scenario file for original storm '+str(tstorm)+', year '+str(tyear)+', realization '+str(trealization)+', created from ' + scenarioname
    latitudes_units,longitudes_units = 'degrees_north', 'degrees_east'
    rainrate_units = 'mm hr^-1'
    times_units, times_calendar = 'minutes since 1970-01-01 00:00.0' , 'gregorian'
    
    # Variable Names
    times_name = 'time'                     ## change here
    latitudes_name,longitudes_name  = 'latitude', 'longitude'
    rainrate_name= 'precipitation rate'
    xlocation_name, ylocation_name = 'x index of transposition', 'y index of transposition'
    
    history, missing = 'Created ' + str(datetime.now()), '-9999.'
    source = 'RainyDay storm scenario file created from ' + scenarioname + '. See description for JSON file contents.'
    
    data=xr.Dataset(
         {
                "rain": (["time","latitude", "longitude"], transposedrain),
                "xlocation": (["scalar_dim"], rainlocx),
                "ylocation": (["scalar_dim"], rainlocy),
                "returnperiod": (["scalar_dim"], [np.float32(scenario_returnperiod)],
                                 {"units": "years",
                                  "long_name": "return period of scenario",
                                  "comment": "valid only for the single largest storm of a given year rank; -9999. for any additional NPERYEAR>1 storms sharing that same year rank"}),
                "original_stormnumber": (["scalar_dim"], [np.int16(origstormnumber)],
                                 {"units": "dimensionless",
                                  "long_name": "parent storm number from storm catalog"})
                #"scenariotime":(["time"],raintime)
            },
            coords={
                "time": raintime,
                "latitude": subrangelat,
                "longitude": subrangelon,
                "scalar_dim": [0]
            },
            attrs={
            "history":history,
            "source" :  source,
            "missing" : missing,
            "description" : description_string,
            "calendar" : times_calendar,
            "times_units": times_units,
            "latitudes_units": latitudes_units,
            "longitudes_units": longitudes_units,
            "rainrate_units": rainrate_units,
            "rainrate_name": rainrate_name,
            "xlocation_name": xlocation_name,
            "ylocation_name": ylocation_name
            }
    )
    
    # # 
    # data_vars = dict(
    #                  rain = (("time","latitude", "longitude"),transposedrain,{'units': rainrate_units, 'long_name': rainrate_name}),
    #                 # rain = (("time","latitude", "longitude"),catrain[:, ::-1, :],{'units': rainrate_units, 'long_name': rainrate_name}),
    #                  xlocation = (("scalar_dim"),[rainlocx],{'units': 'dimensionless', 'long_name': xlocation_name}),
    #                  ylocation = (("scalar_dim"),[rainlocy],{'units': 'dimensionless', 'long_name': ylocation_name}),
    #                  time = (("time"), raintime)),
    # coords = dict(time = ((times_name),raintime),
    #                  longitude = (("longitude"), subrangelon.data , {'units': longitudes_units, 'long_name': longitudes_name}),
    #                  latitude =  (("latitude"), subrangelat.data, {'units': latitudes_units, 'long_name': latitudes_name}),
    #                  scalar_dim=(("scalar_dim"),[0])
    #                  #latitude =  (("latitude"), latrange[::-1].data, {'units': latitudes_units, 'long_name': latitudes_name}),
    #               )
    
    #attrs  = dict(history =history, source =  source, missing = missing, description = description_string,  calendar = times_calendar)
    
    #scenario = xr.Dataset(data_vars = data_vars, coords = coords, attrs = attrs)
    #scenario.time.encoding['units'] = "minutes since 1970-01-01 00:00:00"
    
    data.to_netcdf(name_scenariofile)
    data.close()


# =============================================================================
# added LY 03132025: writing single storm scenario file using normalized SST
# modified by Ashar 07/12/2026: added returnperiod and original_stormnumber output variables
# =============================================================================
def Normalized_SST_write(catrain, raintime, rainlocx, rainlocy, outmultiplier, name_scenariofile, tstorm, tyear, trealization, maskheight,maskwidth, subrangelat, subrangelon, scenarioname, mask, origstormnumber, scenario_returnperiod):
    """
    Write one normalized-SST scenario to its own NetCDF file.

    Same as ``writescenariofile`` except that the cropped rainfall is
    multiplied by the watershed mask and by the cell-by-cell rescaling
    multiplier from ``SSTalt_normalized``.

    Added by Lei Yan (Mar 2025); returnperiod and original_stormnumber added
    by Ashar (July 2026).

    Parameters
    ----------
    catrain : np.ndarray, shape (nt, nlat, nlon)
    raintime : np.ndarray of datetime64, shape (nt,)
    rainlocx, rainlocy : np.ndarray of int, shape (1,)
    outmultiplier : np.ndarray, shape (1, maskheight, maskwidth)
        Rescaling multiplier field for this scenario.
    name_scenariofile : str
    tstorm, tyear, trealization : int
    maskheight, maskwidth : int
    subrangelat, subrangelon : np.ndarray
    scenarioname : str
    mask : np.ndarray, shape (maskheight, maskwidth)
        Binary watershed mask.
    origstormnumber : int
    scenario_returnperiod : float

    Notes
    -----
    Status: used by RainyDay_Py3.py (NORMALIZEDSST = "dimensionless").
    """
    transposedrain=np.multiply(catrain[:,rainlocy[0] : (rainlocy[0]+maskheight), rainlocx[0] : (rainlocx[0]+maskwidth)],mask)
    rain_nsst = transposedrain * outmultiplier

    description_string = 'RainyDay storm scenario file for rescaled storm ' + str(tstorm) + ', year ' + str(tyear) + ', realization ' + str(trealization) + ', created from ' + scenarioname
    times_units, times_calendar = 'minutes since 1970-01-01 00:00.0', 'gregorian'

    # Variable Names
    history, missing = 'Created ' + str(datetime.now()), '-9999.'
    source = 'RainyDay storm scenario file created from ' + scenarioname + '. See description for JSON file contents.'

    data = xr.Dataset(
        {
            "rain": (["time", "latitude", "longitude"], rain_nsst),
            "xlocation": (["scalar_dim"], rainlocx),
            "ylocation": (["scalar_dim"], rainlocy),
            "returnperiod": (["scalar_dim"], [np.float32(scenario_returnperiod)],
                             {"units": "years",
                              "long_name": "return period of scenario",
                              "comment": "valid only for the single largest storm of a given year rank; -9999. for any additional NPERYEAR>1 storms sharing that same year rank"}),
            "original_stormnumber": (["scalar_dim"], [np.int16(origstormnumber)],
                             {"units": "dimensionless",
                              "long_name": "parent storm number from storm catalog"})
            # "scenariotime":(["time"],raintime)
        },
        coords={
            "time": raintime,
            "latitude": subrangelat,
            "longitude": subrangelon,
            "scalar_dim": [0]
        },
        attrs = {
            "history": history,
            "source": source,
            "missing": missing,
            "description": description_string,
            "calendar": times_calendar,
            "times_units": times_units,
            "latitudes_units": "degrees_north",
            "longitudes_units": "degrees_east",
            "rainrate_units": "mm hr^-1",
            "rainrate_name": "precipitation rate",
            "xlocation_name": "x index of transposition",
            "ylocation_name": "y index of transposition"
        }
    )

    data.to_netcdf(name_scenariofile)
    data.close()


#==============================================================================    
# http://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list 
#============================================================================== 
def subfinder(mylist, pattern):
    """
    Return the start indices of every occurrence of the sub-list ``pattern``
    in ``mylist``.

    Parameters
    ----------
    mylist : list
    pattern : list

    Returns
    -------
    list of int

    Notes
    -----
    From http://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
    Status: not currently called.
    """
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(i)
    return matches
    
    
#==============================================================================
# CREATE FILE LIST:- This will create file list and will remove the years(EXCLUDEYEARS) from the input given in .sst file. Also, it will return the number of years included in the dataset.
#==============================================================================

def try_parsing_date(text):
    """
    Parse a date string using several common formats
    ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y%m%d').

    Parameters
    ----------
    text : str

    Returns
    -------
    datetime.datetime

    Raises
    ------
    ValueError
        If none of the formats match.

    Notes
    -----
    Untested. Status: not currently called.
    """
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y%m%d'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')  ### This fucntion is not tested yet
    
    
def createfilelist(inpath, includeyears, excludemonths):
    """
    Build the sorted list of input rainfall files, keeping only the requested
    years and dropping the excluded months.

    The date of each file is read from its filename, which must contain a date
    in YYYYMMDD, YYYY-MM-DD or YYYY/MM/DD form (the first match is used).
    RainyDay expects one file per day.

    Parameters
    ----------
    inpath : str
        Glob pattern for the rainfall .nc files (RAINPATH).
    includeyears : list of int or False
        Years to include (INCLUDEYEARS). False means all years.
    excludemonths : list of int
        Months (1-12) to exclude (EXCLUDEMONTHS). Empty list for none.

    Returns
    -------
    new_list : list of str
        Files to use, sorted by name.
    nyears : int
        Number of distinct years among the kept files.

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    flist = sorted(glob.glob(inpath))
    new_list = [] ; years = set()
    for file in flist:
        base = os.path.basename(file)
        match = re.search(r'\d{4}(?:\d{2})?(?:\d{2}|\-\d{2}\-\d{2}|\/\d{2}/\d{2})', base)
        ### The block below is added for file named in formats YYYYMMDD, YYYY-MM-DD or YYYY/MM/DD and it will
        ### errors after year 1299 in format DDMMYYYY or MMDDYYYY
        try:
            file_date = datetime.strptime(match.group().replace("-","").replace("/",""),'%Y%m%d')
        except:
            sys.exit("You need to give file names in YYYYMMDD, YYYY-MM-DD or YYYY/MM/DD formats")
        file_year = file_date.year; file_month = file_date.month
        if includeyears == False:
            if file_month not in excludemonths:
                new_list.append(file); years.add(file_year)
        else:
            if file_year in includeyears and file_month not in excludemonths:
                new_list.append(file); years.add(file_year)
    nyears = len(years) ## can be made more efficient
    return new_list, nyears
    
    

#==============================================================================
# Get things set up
#==============================================================================
def rainprop_setup(infile,rainprop,variables,catalog=False):
    """
    Inspect one input rainfall file (or a storm catalog file) and derive the
    grid and time properties RainyDay needs.

    Checks performed (the program exits if any fail):

    * latitude/longitude are 1D (regular lat/lon grid);
    * x and y resolutions are equal (square cells);
    * time steps are evenly spaced;
    * for raw input, the file spans exactly one day;
    * at most one negative (missing-data) value is present.

    For raw input, it also works out which variables can be skipped
    (``droplist``) so later reads are faster.

    Parameters
    ----------
    infile : str
        Path to one rainfall file (or catalog file if ``catalog`` is True).
    rainprop : GriddedRainProperties
        Not modified; kept for the call signature.
    variables : dict
        Rainfall/latitude/longitude variable names (keys 'latname' and
        'longname' are expected for the coordinates).
    catalog : bool, optional
        True if ``infile`` is a storm catalog file.

    Returns
    -------
    When ``catalog`` is False:
        [xres, yres], [nlat, nlon], [lon_min, lon_max+xres, lat_min-yres, lat_max],
        tempres (int, minutes), nodata, droplist, calendar, time_units
    When ``catalog`` is True:
        [xres, yres], [nlat, nlon], [lon_min, lon_max, lat_min, lat_max], tempres,
        nodata, inrain, intime, inlatitude, inlongitude, catx, caty, catmax, domainmask

    Notes
    -----
    The bounding box offsets (+xres on the east, -yres on the south) reflect
    RainyDay's convention that grid coordinates mark the upper-left corner of
    each cell. The ``catalog=True`` branch unpacks nine values from
    ``readcatalog``, which returns ten or eleven, so that branch would fail.
    Status: used by RainyDay_Py3.py (``catalog=False``).
    """
    if catalog:
        inrain,intime,inlatitude,inlongitude,catx,caty,catmax,_,domainmask=readcatalog(infile)
    else:
        # configure things so that in the storm catalog creation loop, we only read in the necessary variables
        invars=copy.deepcopy(variables)
        # we don't want to drop these:
        del invars['latname']   
        del invars['longname']
        keepvars=list(invars.values())

        # open the "entire" netcdf file once in order to get the list of all variables:        
        inds=xr.open_dataset(infile)
        

        # this will only keep the variables that we need to read in. 
        droplist=find_unique_elements(inds.keys(),keepvars) # droplist will be passed to the 'drop_variables=' in xr.open_dataset within the storm catalog creation loop in RainyDay
        inds.close()
        
        inrain,intime,inlatitude,inlongitude,nctime=readnetcdf(infile,variables,dropvars=droplist,setup = True)
        time_units = nctime.units; calendar = nctime.calendar if hasattr(nctime, 'calendar') else 'gregorian';
        # inrain,intime,inlatitude,inlongitude=readnetcdf(infile,variables,dropvars=droplist)
        # if max(inlongitude) > 180:
        #     inarea = inarea + 360
    
    if len(inlatitude.shape)>1 or len(inlongitude.shape)>1:
        sys.exit("RainyDay isn't set up for netcdf files that aren't on a regular lat/lon grid!")
        #inlatitude=inlatitude[:,0]          # perhaps would be safer to have an error here...
        #inlongitude=inlongitude[0,:]        # perhaps would be safer to have an error here...
    yres=np.abs((inlatitude[1:] - inlatitude[:-1])).mean()
    xres=np.abs((inlongitude[1:] - inlongitude[:-1])).mean()
    if np.isclose(xres,yres)==False:
        sys.exit("Rainfall grid isn't square. RainyDay cannot support that.")

    unqtimes=np.unique(intime)
    if len(unqtimes)>1:
        # tdiff=unqtimes[1:]-unqtimes[0:-1]
        tdiff = np.diff(unqtimes).astype('timedelta64[m]')
        # tempres=np.min(unqtimes[1:]-unqtimes[0:-1])   # temporal resolution
        tempres = np.min(tdiff).astype('timedelta64[m]')
        # if np.any(np.not_equal(tdiff,tempres)):
        #     sys.exit("Uneven time steps. RainyDay can't handle that.")
        if not np.all(tdiff == tempres):
            sys.exit("Uneven time steps. RainyDay can't handle that.")
        
    else:
        #this is to catch daily data where you can't calculate a time resolution
        tempres=np.float32(1440.)
        tempres=tempres.astype('timedelta64[m]')      # temporal resolution in minutes-haven't checked to make sure this works right
    # print(type(tempres) , type(tdiff))
    tempres_minutes = tempres.astype('timedelta64[m]').astype(int)
    if len(intime) * tempres_minutes != 1440. and catalog==False:
        sys.exit("RainyDay requires daily input files, but has detected something different.")
    tempres=np.int32(np.float32(tempres))

    nodata=np.unique(inrain[inrain<0.])
    if len(nodata)>1:
        sys.exit("More than one missing value flag.")
    elif len(nodata)==0 and catalog==False:
        print("Warning: Missing data flag is ambiguous. RainyDay will probably handle this ok, especially if there is not missing data.")
        nodata==-999.
    elif catalog:
        nodata=-999.
    else:
        nodata=nodata[0]

    if catalog:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude),np.min(inlatitude),np.max(inlatitude)],tempres,nodata,inrain,intime,inlatitude,inlongitude,catx,caty,catmax,domainmask
    else:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude)+xres,np.min(inlatitude)-yres,np.max(inlatitude)],tempres,nodata,droplist,calendar,time_units


#==============================================================================
# READ REALIZATION
#==============================================================================

def readrealization(rfile):
    """
    Read a legacy realization NetCDF file written by ``writerealization``.

    Handles both the old ('rainrate') and newer ('precrate', stored north-up
    and flipped back here) variable names.

    Parameters
    ----------
    rfile : str

    Returns
    -------
    outrain, outtime, outlatitude, outlongitude, outlocx, outlocy, outmax,
    outreturnperiod, outstormnumber, origstormnumber, timeunits

    Notes
    -----
    LEGACY. Status: not currently called.
    """
    infile=Dataset(rfile,'r')
    if 'rainrate' in infile.variables.keys():
        oldfile=True
    else:
        oldfile=False
        
    if oldfile:
        outrain=np.array(infile.variables['rainrate'][:])
    else:
        outrain=np.array(infile.variables['precrate'][:])[:,:,::-1,:]
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    outlocx=np.array(infile.variables['xlocation'][:])
    outlocy=np.array(infile.variables['ylocation'][:])
    outmax=np.array(infile.variables['basinrainfall'][:])
    outreturnperiod=np.array(infile.variables['returnperiod'][:])
    outstormnumber=np.array(infile.variables['stormnumber'][:])
    origstormnumber=np.array(infile.variables['original_stormnumber'][:])
    #outstormtime=np.array(infile.variables['stormtimes'][:],dtype='datetime64[m]')
    timeunits=infile.variables['time'].units
    
    infile.close()
    return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outreturnperiod,outstormnumber,origstormnumber,timeunits


#==============================================================================
# READ NPERYEAR REALIZATION
#==============================================================================
def readrealization_nperyear(rfile):
    """
    Read a legacy NPERYEAR realization NetCDF file written by
    ``writerealization_nperyear``.

    Parameters
    ----------
    rfile : str

    Returns
    -------
    outrain, outtime, outlatitude, outlongitude, timeunits

    Notes
    -----
    LEGACY. Status: not currently called.
    """
    infile=Dataset(rfile,'r')
    if 'rainrate' in infile.variables.keys():
        oldfile=True
    else:
        oldfile=False
        
    if oldfile:
        outrain=np.array(infile.variables['rainrate'][:])
    else:
        outrain=np.array(infile.variables['precrate'][:])[:,:,::-1,:]
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    #outlocx=np.array(infile.variables['xlocation'][:])
    #outlocy=np.array(infile.variables['ylocation'][:])
    #outmax=np.array(infile.variables['basinrainfall'][:])
    #outreturnperiod=np.array(infile.variables['returnperiod'][:])
    #outstormnumber=np.array(infile.variables['stormnumber'][:])
    #origstormnumber=np.array(infile.variables['original_stormnumber'][:])
    #outstormtime=np.array(infile.variables['stormtimes'][:],dtype='datetime64[m]')
    timeunits=infile.variables['time'].units
    
    infile.close()
    return outrain,outtime,outlatitude,outlongitude,timeunits



#==============================================================================
# READ A PREGENERATED SST DOMAIN FILE
#==============================================================================

def readdomainfile(rfile,inbounds=False):
    """
    Read a pregenerated transposition domain mask from NetCDF (DOMAINFILE).

    Parameters
    ----------
    rfile : str
    inbounds : array-like or False, optional
        Index bounds [x0, x1, y1, y0] to subset.

    Returns
    -------
    outmask : np.ndarray
        Domain mask (1 inside, 0 outside).
    outlatitude, outlongitude : np.ndarray

    Notes
    -----
    The DOMAINFILE option in RainyDay_Py3.py currently exits before calling
    this ("capability isn't tested").
    """
    infile=Dataset(rfile,'r')
    if np.any(inbounds!=False):
        outmask=np.array(infile.variables['domain'][inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outlatitude=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlongitude=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])         
    else:
        outmask=np.array(infile.variables['domain'][:])
        outlatitude=np.array(infile.variables['latitude'][:])
        outlongitude=np.array(infile.variables['longitude'][:])
    infile.close()
    return outmask,outlatitude,outlongitude


#==============================================================================
# "Rolling sum" function to correct for short-duration biases
#==============================================================================
    
def rolling_sum(a, n):
    """
    Moving (rolling) sum over the first (time) axis, ignoring NaNs.

    Used for the duration correction: when the storm catalog is longer than
    the analysis DURATION, each output slice is the rainfall summed over one
    DURATION-long window, so the most intense window can be selected.

    Parameters
    ----------
    a : np.ndarray, shape (nt, ...)
    n : int
        Window length in time steps.

    Returns
    -------
    np.ndarray, shape (nt-n+1, ...)
        Element k is the sum of ``a[k:k+n]``.

    Notes
    -----
    Status: used by RainyDay_Py3.py.
    """
    ret = np.nancumsum(a, axis=0, dtype=float)
    ret[n:,:] = ret[n:,:] - ret[:-n,: ]
    return ret[n - 1:,: ]


#==============================================================================
# Distance between two points
#==============================================================================
    
def latlondistance(lat1,lon1,lat2,lon2):    
    #if len(lat1)>1 or len(lon1)>1:
    #    sys.exit('first 2 sets of points must be length 1');
    """
    Great-circle distance between points using the haversine formula.

    Parameters
    ----------
    lat1, lon1 : float or np.ndarray
        First point(s), in degrees.
    lat2, lon2 : float or np.ndarray
        Second point(s), in degrees.

    Returns
    -------
    float or np.ndarray
        Distance in meters (Earth radius 6,371 km).

    Notes
    -----
    Status: not currently called.
    """
    R=6371000;
    dlat=np.radians(lat2-lat1)
    dlon=np.radians(lon2-lon1)
    a=np.sin(dlat/2.)*np.sin(dlat/2.)+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.)*np.sin(dlon/2.);
    c=2.*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c
 
#==============================================================================
# rescaling functions
#==============================================================================
        
@jit(fastmath=True)
def intenseloop(intenserain,tempintense,xlen_wmask,ylen_wmask,maskheight,maskwidth,trimmask,mnorm,domainmask):
    """
    Watershed-average a stack of gridded rainfall fields at every possible
    transposition position (for stochastic/deterministic rescaling).

    For each position (y, x) inside the domain with no missing data, computes
    ``sum(intenserain[:, y:y+h, x:x+w] * trimmask) / mnorm`` for every field in
    the stack, so the rescaling statistics are at the same (watershed) scale
    as the transposed storms.

    Parameters
    ----------
    intenserain : np.ndarray, shape (nstorms, ny, nx)
        Log storm totals.
    tempintense : np.ndarray, shape (nstorms, ylen_wmask, xlen_wmask)
        Output array (filled in place).
    xlen_wmask, ylen_wmask : int
        Number of transposition positions in x and y.
    maskheight, maskwidth : int
    trimmask : np.ndarray
    mnorm : float
        Sum of ``trimmask`` (normalizing constant).
    domainmask : np.ndarray

    Returns
    -------
    tempintense : np.ndarray
        NaN where the position is outside the domain or has missing data.
    """
    for i in range(0,xlen_wmask*ylen_wmask):
        y=i//xlen_wmask
        x=i-y*xlen_wmask
        if np.equal(domainmask[y,x],1.) and  np.any(np.isnan(intenserain[:,y,x]))==False:
        # could probably get this working in nopython if I coded the multiplication explicitly, rather than using using the axis argument of nansum, which isn't numba-supported
            tempintense[:,y,x]=np.sum(np.multiply(intenserain[:,y:(y+maskheight),x:(x+maskwidth)],trimmask),axis=(1,2))/mnorm    
        else:
            tempintense[:,y,x]=np.nan
    return tempintense

@jit(nopython=True,fastmath=True)
def intense_corrloop(intenserain,intensecorr,homerain,xlen_wmask,ylen_wmask,mnorm,domainmask):   
    """
    Correlation, at every transposition position, between the storm-total
    series there and the series at the home (watershed) location.

    Parameters
    ----------
    intenserain : np.ndarray, shape (nstorms, ylen_wmask, xlen_wmask)
    intensecorr : np.ndarray, shape (ylen_wmask, xlen_wmask)
        Output array (filled in place).
    homerain : np.ndarray, shape (nstorms,)
        Series at the home location.
    xlen_wmask, ylen_wmask : int
    mnorm : float
        Unused.
    domainmask : np.ndarray

    Returns
    -------
    intensecorr : np.ndarray
        Pearson correlation; NaN outside the domain or where data are missing.
    """
    for i in range(0,xlen_wmask*ylen_wmask): 
        y=i//xlen_wmask
        x=i-y*xlen_wmask
        if np.equal(domainmask[y,x],1.) and  np.any(np.isnan(intenserain[:,y,x]))==False:
            intensecorr[y,x]=np.corrcoef(homerain,intenserain[:,y,x])[0,1]
        else:
            intensecorr[y,x]=np.nan
    return intensecorr


#==============================================================================
# read arcascii files
#==============================================================================

def read_arcascii(asciifile):
    # note: should add a detection ability for cell corners vs. centers: https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/esri-ascii-raster-format.htm
    """
    Read an ESRI ASCII grid (.asc) file.

    Parameters
    ----------
    asciifile : str

    Returns
    -------
    asciigrid : np.ndarray of float32, shape (nrows, ncols)
        Grid values with NODATA replaced by NaN.
    ncols, nrows : int
    xllcorner, yllcorner : float32
        Lower-left corner coordinates.
    cellsize : float32

    Notes
    -----
    Assumes the standard 6-line header in the usual order and corner (not
    center) registration. Status: not currently called.
    """
    temp=linecache.getline(asciifile, 1)
    temp=linecache.getline(asciifile, 2)
    xllcorner=linecache.getline(asciifile, 3)
    yllcorner=linecache.getline(asciifile, 4)
    cellsize=linecache.getline(asciifile, 5)
    nodata=linecache.getline(asciifile, 6)
    
    #ncols=np.int(ncols.split('\n')[0].split(' ')[-1])
    #nrows=np.int(nrows.split('\n')[0].split(' ')[-1])
    
    xllcorner=np.float32(xllcorner.split('\n')[0].split(' ')[-1])
    yllcorner=np.float32(yllcorner.split('\n')[0].split(' ')[-1])
    
    cellsize=np.float32(cellsize.split('\n')[0].split(' ')[-1])
    nodata=np.float32(nodata.split('\n')[0].split(' ')[-1])
    
    #asciigrid = np.loadtxt(asciifile, skiprows=6)
    asciigrid = np.array(pd.read_csv(asciifile, skiprows=6,delimiter=' ', header=None),dtype='float32')
    nrows=asciigrid.shape[0]
    ncols=asciigrid.shape[1]
    
    asciigrid[np.equal(asciigrid,nodata)]=np.nan

    return asciigrid,ncols,nrows,xllcorner,yllcorner,cellsize



#==============================================================================
# used for prepping "drop_variables" so we don't read in unnecessary variables using xarray
#==============================================================================
def find_unique_elements(list1, list2):
    """
    Return the elements of ``list1`` that are not in ``list2``.

    Used to build the ``drop_variables`` list for xarray so that only the
    rainfall variable (and coordinates) are read from each input file.

    Parameters
    ----------
    list1 : iterable
        Target list of values to be reduced according to list2 (e.g. all
        variable names in a file).
    list2 : iterable
        Values to keep (e.g. the rainfall variable name).

    Returns
    -------
    list
        The values in list1 that were not present in list2.
    """
    unique_elements_in_list1 = [x for x in list1 if x not in list2]
    #unique_elements_in_list2 = [x for x in list2 if x not in list1]
    return unique_elements_in_list1


#==============================================================================
# 
#==============================================================================
def is_monotonic(arr):
    """
    Check whether a 1D array is monotonically non-decreasing or non-increasing.

    Used to decide whether a SEASONALSAMPLING file holds a CDF (monotonic) or
    a PMF.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    bool
    """
    return np.all(np.diff(arr) >= 0) or np.all(np.diff(arr) <= 0)


#==============================================================================
# 
#==============================================================================
def day_of_year_to_datetime(year, day_of_year):
    # Create a datetime for the first day of the given year
    """
    Convert a year and day-of-year to a numpy datetime64 date.

    Parameters
    ----------
    year : int
    day_of_year : int
        1-based (1 = January 1).

    Returns
    -------
    numpy.datetime64 (day precision)
    """
    start_of_year = np.datetime64(str(year), 'Y')

    # Add the number of days to get to the desired day of the year
    result = start_of_year + np.timedelta64(day_of_year - 1, 'D')

    return result

#==============================================================================
# 
#==============================================================================
def replace_year(dt, new_year):
    # Extract the time part from the datetime
    """
    Return the same month/day/time as ``dt`` but in ``new_year``.

    Used for seasonal sampling: storm dates are mapped onto a common reference
    year (1776/1777) so they can be compared by day of year.

    Parameters
    ----------
    dt : numpy.datetime64
    new_year : int

    Returns
    -------
    numpy.datetime64

    Notes
    -----
    Implemented by adding the elapsed time since January 1 to January 1 of the
    new year, so leap-year dates after Feb 28 shift by one day when the target
    year is not a leap year (and vice versa).
    """
    time_part = dt - np.datetime64(dt, 'Y')

    # Get the current year of the datetime
    current_year = np.datetime64(dt, 'Y')

    # Calculate the difference between the current year and the new year
    year_diff = new_year - (current_year.astype(int)+1970)

    # Add the difference to the datetime
    result = np.datetime64(dt, 'Y') + np.timedelta64(year_diff, 'Y') + time_part

    return result