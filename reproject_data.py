#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions to aid in projecting mapped data using pyproj"""
__version__ = '1.0.1'
__date__ = '2021-12-03'
__author__ = 'Oliver King'
__email__ = 'ortk1@le.ac.uk'
__url__ = 'https://github.com/ortk95/reproject-data'

import pyproj
import scipy.interpolate
import numpy as np


def transform_projection(
        data,
        proj_in, proj_out,
        x_in, y_in,
        x_out=None, y_out=None,
        out_size=100,
        interpolate_method='linear',
        ):
    """
    Reprojects mapped data from one map projection to another.

    Parameters
    ----------
    data
        2D array of data to be reprojected

    proj_in, proj_out
        Map projections of input and output projections respectively. Passed to 
        pyproj.CRS() to create transformation. Can be provided as a string or a
        cartopy projection (e.g. `proj_out=ccrs.Orthographic()`). See 
        `project_globe()` for example of defining projection strings. For full 
        list of projections and relevant parameters, see
        https://proj.org/operations/projections/index.html. 

    x_in, y_in
        Coordinates of input data in coordinate system used by proj_in. Can be 
        either 2D arrays of the same shape as data, or 1D arrays of coordinates.
        Note that both x_in and y_in must have the same dimension (i.e. both 1D
        or both 2D).

    x_out, y_out
        Coordinates of input data in coordinate system used by proj_out. Can be 
        either 2D arrays of the same shape as data, or 1D arrays of coordinates.
        Note that both x_in and y_in must have the same dimension (i.e. both 1D
        or both 2D). If x_out or y_out are None then output coordinates will be
        automatically calculated to cover the entire finite range of output 
        coordinates (use out_size to specify the size of the output grid in this
        case).

    out_size
        Grid size of output data if x_out or y_out are None. Must be an integer
        or an iterable to two integers. If out_size is an integer, the same 
        value is used for both the x and y dimensions. If out_size is an
        iterable (e.g. list, tuple etc.), the first value is the grid size in 
        the x dimension and the second value is the grid size in the y 
        dimension.

    interpolate_method
        Interpolation method used in reprojection. Passed to 
        `scipy.interpolate.griddata()`.

    Returns
    -------
    x_out, y_out
        2D coordinate arrays of reprojected data.

    data_out
        Data reprojected into proj_out with coordinates specified by x_out and
        y_out.
    """
    # Check input data and coordinates
    data = np.asarray(data)
    x_in = np.asarray(x_in)
    y_in = np.asarray(y_in)
    if len(x_in.shape) == 1:
        x_in, y_in = np.meshgrid(x_in, y_in)
    assert (data.shape == x_in.shape == y_in.shape), 'Inconsistent data/input coordinates provided'

    # Transform coordinates
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS(proj_in),
        pyproj.CRS(proj_out),
        )
    x_proj, y_proj = transformer.transform(x_in, y_in)

    # Check output coordinates
    if np.isscalar(out_size):
        out_size = (out_size, out_size)
    if x_out is None:
        valid_values = x_proj[np.isfinite(x_proj)]
        x_out = np.linspace(min(valid_values), max(valid_values), out_size[0])
    if y_out is None:
        valid_values = y_proj[np.isfinite(y_proj)]
        y_out = np.linspace(min(valid_values), max(valid_values), out_size[1])
    x_out = np.asarray(x_out)
    y_out = np.asarray(y_out)
    if len(x_out.shape) == 1:
        x_out, y_out = np.meshgrid(x_out, y_out)
    assert (x_out.shape == y_out.shape), 'Inconsistent output coordinates provided'

    # Reproject data using transformed projection
    points = np.array([(x, y) for x, y in zip(x_proj.ravel(), y_proj.ravel())])
    good_points = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    data_out = scipy.interpolate.griddata(
        points[good_points],
        data.ravel()[good_points],
        (x_out, y_out),
        method=interpolate_method)
    return x_out, y_out, data_out


def project_globe(
        data, r_eq, r_pol, km_per_px,
        distance=None,
        central_longitude=0, central_latitude=0,
        x_in=None, y_in=None,
        central_longitude_in=180,
        pixel_margin=1,
        **kw):
    """
    Projects lat/long mapped data in an orthographic/nearside perspective
    projection.

    Convenience wrapper function to call transform_projection() to simulate
    appearance of a planet from a given distance. If distance is infinite or 
    None, projection is orthographic, otherwise a nearise persepective
    projection is used.

    Parameters
    ----------
    data
        2D array of data to be reprojected. Assumes input data is in a lat/long
        grid with all latitudes and longitudes included.

    r_eq
        Equatorial radius (semimajor axis) of planet in km.

    r_pol
        Polar radius (semiminor axis) of planet in km.

    km_per_px
        Spatial resolution (in km per pixel) of output reprojected image.

    distance
        Distance of observer (in km) from planet. If distance is None or 
        infinite an orthographic projection is used (i.e. observer at infinity).

    central_longitude, central_latitude
        Coordinates (in degrees) of centre of projected data.

    x_in, y_in
        Coordinates of input data in coordinate system used by proj_in. Passed
        to `transform_projection()`. x_in should provide input longitudes and
        y_in should provide input latitudes. If None, data is assumed to cover 
        entire globe and coordinates are calculated automatically. See 
        `transform_projection()` for more details.

    central_longitude_in
        Longitude (in degrees) of centre of input data.

    pixel_margin
        Approximate pixel margin around planet in reprojected image.

    **kw
        Additional arguments passed to `transform_projection()`.

    Returns
    -------
    x_out, y_out
        2D coordinate arrays of reprojected data.

    projected
        Reprojected data.
    """
    assert r_eq >= r_pol, 'r_eq must be greater than or equal to r_pol'
    data = np.asarray(data)

    # Generate projections
    proj_in = '+proj=eqc +a={a} +b={b} +lon_0={lon_0} +to_meter={to_meter} +type=crs'.format(
        a=r_eq,
        b=r_pol,
        lon_0=central_longitude_in,
        to_meter=np.radians(1) * r_eq,
        )
    if distance is None or not np.isfinite(distance):
        # set false northing (y_0) to keep disc vertically centred
        proj_out = '+proj=ortho +a={a} +b={b} +lon_0={lon_0} +lat_0={lat_0} +y_0={y_0} +type=crs'.format(
            a=r_eq,
            b=r_pol,
            lon_0=central_longitude,
            lat_0=central_latitude,
            y_0=(r_pol - r_eq) * np.sin(np.radians(central_latitude * 2)),
            )
    else:
        proj_out = '+proj=nsper +h={h} +a={a} +b={b} +lon_0={lon_0} +lat_0={lat_0} +type=crs'.format(
            h=distance,
            a=r_eq,
            b=r_pol,
            lon_0=central_longitude,
            lat_0=central_latitude,
            )

    # Generate image with buffer along longest axis
    # create image grid which is sized by longest radius value plus pixel margin
    # ensure outer pixel includes edge of disk by usuing np.ceil()
    r_max = km_per_px * (np.ceil(r_eq / km_per_px) + pixel_margin)
    # add km_per_px/2 to exclusive upper limit so that final value is r_max
    coords_out = np.arange(-r_max, r_max + km_per_px / 2, km_per_px)

    # Reproject data
    if x_in is None:
        x_in = np.linspace(-180, 180, data.shape[1])
    if y_in is None:
        y_in = np.linspace(-90, 90, data.shape[0])
    x_out, y_out, projected = transform_projection(
        data,
        proj_in, proj_out,
        x_in, y_in,
        coords_out, coords_out,
        **kw)
    return x_out, y_out, projected


def __example():
    """
    Example of using `project_globe()` to display an image of Jupiter.
    """
    print('Running example image projection code...')
    import matplotlib.pyplot as plt
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError
    import io

    # Loading example image of Jupiter from NASA website
    print(' Loading example image...')
    url = 'https://svs.gsfc.nasa.gov/vis/a010000/a012000/a012021/Hubble_Jupiter_color_global_map_2015a_print.jpg'
    try:
        with urlopen(url) as f:
            img = np.flipud(np.asarray(
                plt.imread(io.BytesIO(f.read()), format='jpg'),
                dtype=float))
            img = img[:, :, 0] - img[:, :, 2]  # create 'interesting' image
            img = img[::4, ::4]  # slice image for speed
    except (HTTPError, URLError) as e:
        print('Error reading Jupiter image from URL. Exiting.')
        print(' ' + repr(e))
        return

    # Perform actual image reprojecetions with appropriate values for Jupiter

    print(' Projecting orthographic image...')
    x, y, img_ortho = project_globe(
        img,
        r_eq=71492,
        r_pol=66854,
        km_per_px=500,
        central_latitude=20,
        central_longitude=-90,
        )

    # Plot data
    print(' Plotting data...')
    fig, ax = plt.subplots()
    ax.contourf(x, y, img_ortho, cmap='Spectral_r', levels=100)
    ax.set_aspect('equal')

    # Add nicer formatting
    ax.set_title('Jupiter (red - blue)')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    ax.ticklabel_format(style='sci', scilimits=(-3, 3))
    plt.show()


if __name__ == '__main__':
    __example()
