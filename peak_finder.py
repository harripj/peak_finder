# -*- coding: utf-8 -*-

"""
Created on Wed Aug 16 17:20:20 2017

@author: pjh523
"""

import numpy as _np
import pandas as _pd
import math as _math
from scipy import optimize as _optimize, signal as _signal, stats as _stats
import matplotlib.pyplot as _plt
from models import Gaussian as _Gaussian
import logging as _logging

def get_bin_centers(x):
    '''

    Compute bin centers from bin edges returned by histogram function.

    Parameters
    ----------
    x: array-like 1d
        Bin edges, including left and rightmost edges.

    Returns
    -------
    centers: ndarray 1d
        Bin centers.

    '''
    x = _np.asarray(x)
    return x[:-1] + _np.ediff1d(x)/2.0


def plot_gaussian_peaks(x, parameters, ax, plot_individual=False, **kwargs):
    """
    Plots multiple gaussian peaks, defined by their parameters, on axes.
    
    Parameters
    ----------
    x: 1d array-like
        x data to plot.
    parameters: array-like
        List of parameters defining the Gaussians to plot. Length must be 
        multiple of {}.
    ax: matplotlib.Axes
        Axes to plot on.
    plot_individual: bool, default is False
        If True the individual Gaussians constitutuing the overall fit
        are also plotted.
    **kwargs
        Passed to matplotlib.axes.plot.
    
    """.format(
        _Gaussian.parameters
    )
    # define default colour
    kwargs.setdefault("color", "r")
    kwargs.setdefault("label", "Fit")

    # plot individual gaussian
    if plot_individual:
        for i in range(0, len(parameters), _Gaussian.plength):
            # generate vector
            y = _Gaussian.vector(x, *parameters[i : i + _Gaussian.plength])
            # plot it
            # label="Gauss. {:d}".format(i // _Gaussian.plength)
            ax.plot(x, y, ls="dotted", color=kwargs.get("color", "k"))

    # overall fit
    fit = _Gaussian.vector(x, *parameters)
    # plot fit after individual so it stays on top
    ax.plot(x, fit, **kwargs)


def get_closest_indices(arr, values):
    """
    
    Returns the indices in array that are closest to values.

    Parameters
    ----------
    arr: array-like
        Array to search.
    values: array-like
        Values to find the closest index.

    Returns
    -------
    indices: numpy.ndarray, dtype is int

    """
    # format input
    arr = _np.asarray(arr)
    return _np.array([_np.argmin(_np.abs(arr - val)) for val in values])


def create_guess_from_peak_positions(y, x, peaks, sigma=None):
    """

    Create a 'guess' to be passed to guess parameter of guess_peaks fn just from
    the peak positions. Amplitude and width of peaks will be defined automatically.
    
    Parameters
    ----------
    y: 1d array-like
        y data.
    x: 1d array-like
        x data.
    peaks: 1d array-like
        Indices of peak locations in (x, y).
        If peaks are all integers then it is assumed they are indices.
        If not then it is assumed that peaks are in data (x) coordinates.
    sigma: None, float, or array-like, default is None
        If None then sigma is defined at the half-width-half-maximum using
        scipy.signal.peak_prominence.
        If float then sigma scales with x0.
        If array-like then sigma is defined for each peak.
        
    Returns
    -------
    guess: 1d array-like, len(guess) = 3*Gaussian.plength.
    
    """
    # format inputs
    # make sure they x and y are arrays
    x = _np.asarray(x, dtype=float)
    y = _np.asarray(y, dtype=float)

    # get peaks in index format if not already
    peaks = _np.asarray(peaks)
    # if peaks are all integers, then assume they are indices
    if not _np.issubdtype(peaks.dtype, _np.integer):
        peaks = get_closest_indices(x, peaks)

    # create output array holder
    guess = _np.empty(len(peaks) * _Gaussian.plength, dtype=float)

    # if array-like
    if isinstance(sigma, (list, tuple, _np.ndarray)):
        assert len(sigma) == len(
            peaks
        ), "sigma must be defined for each value in peak_positions."
    # if just a number
    elif isinstance(sigma, (int, float)):
        # turn to tuple to iterate over (same value at each position)
        sigma = (sigma,) * len(peaks)
    # if sigma is None then do auto sigma according to hwhm
    elif sigma is None:
        # calculate peak prominences
        prom, left, right = _signal.peak_prominences(y, peaks)
        # if any prominences are 0 remove them from guess calculation
        if _np.isclose(prom, 0).any():
            valid = _np.logical_not(_np.isclose(prom, 0))
            prom = prom[valid]
            left = left[valid]
            right = right[valid]
            peaks = peaks[valid]

        sigma = _np.empty(peaks.size, dtype=float)
        # assess each peak locally
        for i, peak in enumerate(peaks):
            # get local region
            _slice = slice(left[i], right[i])
            # get one index at half prominence
            index = _np.argmin(_np.abs(y[_slice] - (y[peak] - prom[i] / 2.0)))
            # get half-width-half-maximum, difference from found index
            # to peak position
            hwhm = abs(x[_slice][index] - x[peak])
            # fwhm = 2*sqrt(2*ln(2))*sigma
            sigma[i] = hwhm / _math.sqrt(2.0 * _math.log(2.0))
    else:
        raise TypeError("sigma should be None (auto), float, or array-like.")
    # create final guess list
    for i, peak in enumerate(peaks):
        # add (x0, A, sigma) to guess array in same order as peaks
        guess[_Gaussian.plength * i : _Gaussian.plength * (i + 1)] = (
            x[peak],
            y[peak],
            sigma[i],
        )
    # return array
    return guess


def guess_peaks(
    y,
    x=None,
    guess=None,
    sigma=None,
    x0_max=None,
    data_limits=None,
    n_peaks=None,
    threshold=None,
    fit_individually=False,
    ax=None,
    npts=128,
    plot_individual_gaussians=False,
    plot_bar=False,
    plot_peak_loc=True,
    verbose=False,
    **kwargs
):
    """

    A function to guess peaks from data, fit multiple gaussians and return the
    fit parameters. If no guess is supplied, this function guesses the peaks 
    in order of magnitude.
    
    Parameters
    ----------
    y: 1d array-like
        y values of data
    x: 1d array-like
        x values of data, same shape as y. Default is None, in this case the 
        x spacing is assumed to be 1.
    guess: 1d array-like
        The list of parameters passed to curve_fit as p0 (guess). Must be have
        length 3n, where n is int. Values correlate to the form:
            [x0_1, A_1, sigma_1, x0_2, A_2, sigma_2, ... sigma_n].
        Default is None, in which case the function guesses the peak positions.
    sigma: None, float, or array-like, default is None
        If None then sigma is defined at the half-width-half-maximum using
        scipy.signal.peak_prominence.
        If float then sigma scales with x0.
        If array-like then sigma is defined for each peak.
    x0_max: float
        The maximum peak position to fit for. All peak positions larger than
        this will not be included in the guess to the fitting function.
        Deafult is None.
    data_limits: tuple of float
        Confine the data to within x range (data_min, data_max) to aid fit.
        Default is None.
    n_peaks: int
        The maximum number of peaks to include in the fitting function. If
        defined the largest magnitude peaks found are passed to curve_fit.
        Default is None.
    threshold: float, default is None
        The minimum threshold value for the peak intensity. Used for automatic peak detection.
    fit_individually: bool, default is False
        If True the peaks are fit individually within their local vicinity. The window length is defined by the kwarg 'order' which takes a default value of 1 unless defined. This is equivalent to a window of ± order around the peak location.
    ax: mpl.Axes or None, default is None
        Axes to plot on. If None then no plotting is performed.
    plot_individual_gaussians: bool
        If True the individual found peaks are plotted on the displayed fit.
        Default is False.
    plot_bar: bool
        Plots original data as bar if True. Default is True.
    npts: int
        Number of points over which to plot fit. Default is 128.
    verbose: bool, default is False
        If True then output is printed in the console.        
    **kwargs:
        Keyword arguments for the chosen peak finding algorithm. 
        'order' is useful for scipy.signal.argrelmax, default is 1.
        Default comparator is np.greater_equal
        
    Returns
    -------
    parameters: pandas.DataFrame
        The values of the parameters for the fitted data. To pass through to 
        guess again use: parameters.values.ravel().
        
    """

    # format inputs
    y = _np.asarray(y, dtype=float)
    # if no xdata given, fit to index -> should still work
    if x is None:
        x = _np.arange(y.size, dtype=float)
    else:
        x = _np.asarray(x)
        assert len(x) == len(y), "Arrays x and y must have the same shape."

    # confine data if requested
    if data_limits is not None:
        assert (
            len(data_limits) == 2
        ), "data_limits must be array-like of the form (data_min, data_max)."
        data_min, data_max = data_limits
        mask = _np.logical_and(x >= data_min, x <= data_max)
        # mask (confine) data
        x = x[mask]
        y = y[mask]

    # define comparator for argrelmax if undefined
    kwargs.setdefault("comparator", _np.greater_equal)
    kwargs.setdefault("order", 1)

    # find peaks in y and create a guess (x0, A, sigma)
    if guess is None:
        # get extrema
        (peak_index,) = _signal.argrelextrema(y, **kwargs)

        # limit found peaks in x
        if x0_max is not None:
            # limit biggest peak position for fit (_argrelmax will find any
            # local extrema)
            peak_index = peak_index[x[peak_index] < x0_max]

        if n_peaks is not None:
            assert isinstance(
                n_peaks, int
            ), "n_peaks must be integer (# Gaussians to fit)."
            # sort by decreasing magnitude and get the biggest n_peaks
            peak_index = peak_index[_np.argsort(y[peak_index])[::-1][:n_peaks]]

            # check n_peaks detected
            if peak_index.size > n_peaks:
                if verbose:
                    print("Too many peaks found, reforcing n_peaks={}.".format(n_peaks))
                peak_index = peak_index[:n_peaks]
            elif peak_index.size < n_peaks:
                if verbose:
                    print("Only {} peaks found.".format(peak_index.size))

        # peak minimum thresholding
        if threshold is not None:
            peak_index = peak_index[y[peak_index] >= threshold]

        guess = create_guess_from_peak_positions(y, x, peak_index, sigma=sigma)
    # format guess and define peak_index from it
    else:
        # flatten guess
        guess = _np.ravel(guess)
        assert not guess.size % _Gaussian.plength, (
            "guess must be a list of 3n parameters,"
            + " eg. [x0_1, A_1, sigma_1, x0_2, A_2, sigma_2, ... sigma_n]."
        )

        # convert to numpy in any case
        guess = _np.asarray(guess, dtype=float)
        # get indices of peaks in array
        # this is not previously calculated if guess argument is defined
        peak_index = get_closest_indices(
            x, [guess[i] for i in range(0, guess.size, _Gaussian.plength)]
        )

    # now start fitting...
    if fit_individually:
        # fit each peak in guess locally (±order)
        if verbose:
            print(
                "Fitting individual peaks,"
                + " window length: ±{}.".format(kwargs["order"])
            )

        # create array to store individual fits
        params = _np.empty_like(guess, dtype=float)

        # locally fit each peak
        for i, peak in enumerate(peak_index):
            # slice min and max
            _min = peak - kwargs["order"]
            _max = peak + kwargs["order"] + 1
            # ensure slice valid, default step size
            _slice = slice(_min if _min >= 0 else None, _max if _max < x.size else None)
            # get guess
            p0 = guess[_Gaussian.plength * i : _Gaussian.plength * (i + 1)]
            # do individual fit, return guess if fails
            try:
                pop, cov = _optimize.curve_fit(
                    _Gaussian.vector, x[_slice], y[_slice], p0=p0
                )
            # if fit fails then raise error but return guess
            except RuntimeError as err:
                _logging.error(err)
                return _Gaussian.format_parameter_list(params)
            # stick the result in correct place in array
            params[_Gaussian.plength * i : _Gaussian.plength * (i + 1)] = pop
    else:
        # do full fit
        if verbose:
            print("Fitting all data.")
        # do fit, return guess if fails
        try:
            params, cov = _optimize.curve_fit(_Gaussian.vector, x, y, p0=guess)
        # if fit fails then raise error but return guess
        except RuntimeError as err:
            _logging.error(err)
            return _Gaussian.format_parameter_list(params)

    # format parameters into a pandas DataFrame
    params = _Gaussian.format_parameter_list(params)
    # sort by increasing x0
    params = params.sort_values("x0")
    if verbose:
        print("Found fit parameters:\n{}\n".format(params))

    # if plotting is required
    if ax is not None:
        assert isinstance(ax, _plt.Axes), "ax must be matplotlib.Axes."

        plot_x = _np.linspace(x.min(), x.max(), npts)
        # plot data as histogram
        if plot_bar:
            # default center aligned
            ax.bar(x, y, _np.mean(_np.ediff1d(x)), label="Data")
        # plot peak position
        if plot_peak_loc:
            for i, peak in enumerate(peak_index):
                ax.axvline(
                    x[peak], color="gray", ls="dashed", label="Peak est. #{}".format(i)
                )
        # plot Gaussians on axes
        plot_gaussian_peaks(
            plot_x,
            params.values.ravel(),
            ax,
            plot_individual=plot_individual_gaussians,
            color="r",
            label="Fit",
        )
        # axes legend
        ax.legend()

    return params


def guess_peaks_kde(values, n_peaks=None, bins=100, density=False, ax=None, npts=256, **kwargs):
    """

    Use Kernel Density Estimation to find peak locations within values and then fit the peaks.

    Parameters
    ----------
    values: array-like 1d
        Observed values.
    n_peaks: None or int
        Maximum number of peaks to consider
    bins: int
        Number of bins for histogram.
    density: bool
        If True then the probability density histogram is computed.
    ax: None or plt.Axes
        Axes to plot on if provided.
    npts: int
        Number of points to simulate KDE distribution for peak fitting.
    kwargs: passed to scipy.stats.gaussian_kde, eg. bw_method and guess_peaks.

    Returns
    -------
    params: pd.DataFrame
        Fitted peak values.
    
    """
    values = _np.asarray(values)
    # evaluate kde
    kde = _stats.gaussian_kde(values, **kwargs)

    # compute histogram from values
    vals, bins = _np.histogram(values, bins=bins, density=density)

    x = _np.linspace(values.min(), values.max(), npts)
    y = kde(x) if density else kde(x) * _np.trapz(vals, get_bin_centers(bins))

    # guess peaks in KDE
    guess = guess_peaks(y, x, n_peaks=n_peaks)

    # plot bar if requested
    if ax is not None:
        ax.bar(get_bin_centers(bins), vals, _np.ediff1d(bins))

    # use KDE peaks as guess to fit histogram data
    # second fit required as KDE normally overestimates peak width (depends on bandwidth)
    return guess_peaks(vals, get_bin_centers(bins), guess=guess.values.ravel(), ax=ax)
