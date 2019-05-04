""" Ground track plotting functions. """

# Poliastro modules
from poliastro.twobody.propagation import propagate
from poliastro.examples import soyuz_gto, churi, iss, molniya
from poliastro.plotting.util import generate_label

# Astropy modules
from astropy import units as u
from astropy import time
from astropy import coordinates as coord

# Numpy modules
import numpy as np

# Plotting modules
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class Groundtrack:
    """ Groundtrack class.

    This class holds the groundtrack plotter of
    :py:class:`~poliastro.twobody.orbit.Orbit` instances. This plots
    can be shown by making use of the :py:meth:`plot`.

    """

    def __init__(self, ax=None, num_points=150, dark=False):
        """ Plots the groundtrack of an Orbit.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            Axes for custom figures
        num_points: int, optional
            Number of pooints to use in plots, default to 150
        dark: bool, optional
            If set as True, plots the orbit in Dark mode.
        """

        self.ax = ax
        if not self.ax:
            if dark:
                with plt.style.context("dark_background"):
                    _, self.ax = plt.subplots(figsize=(20, 20))
            else:
                _, self.ax = plt.subplots(figsize=(20, 20))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.stock_img()

        self.num_points = num_points
        self._trajectories = []  # type: List[Trajectory]

    @property
    def trajectories(self):
        return self._trajectories

    def plot(self, orbit, tof, label=None, color=None, method=mean_motion):
        """ Plots the groundtrack for a given time of flight of an Orbit.

        Parameters
        ----------

        orbit: poliastro.twobody.orbit.Orbit
            Orbit to plot the porkchop
        tof: astropy.quantity.Quantity
            Time of flight for the groundtrack
        label: string
            Name or label for the groundtrack line
        color: string
            Color for the groundtrack line
        method: poliastro.twobody.propagate
            Method used for solving positions ob orbit along time

        """
        # Compute positions for tof
        time_span = time.TimeDelta(np.linspace(0, tof, num=num))
        positions = propagate(orbit, time_span, method=method)

        # Transform GCRS to ITRS
        xyz_positions = coord.CartesianRepresentation(
            x=positions.x, y=positions.y, z=positions.z
        )
        xyz_gcrs = coord.GCRS(xyz_positions, obstime=(orbit.epoch + time_span))
        xyz_itrs = xyz_gcrs.transform_to(coord.ITRS(obstime=xyz_gcrs.obstime))

        # Convert to xyz_itrs to lat and lon representation
        latlon_itrs = trajectory.represent_as(coord.SphericalRepresentation)

        # If no label is passed, generate it
        if label:
            label = generate_label(orbit, label)

        # Return lines for groundtrack
        lines = self._plot(latlon_itrs, label, color)

        return lines

    def _plot_groundtrack(self, latlon, color=None):

        # Separate latitude and longitude
        lon = latlon.lon
        lat = latlon.lat

        # Ploting the groundtrack using cartopy
        lines = self.ax.plot(
            lon.to(u.deg), lat.to(u.deg), color=color, transform=ccrs.Geodetic()
        )

        return lines

    def plot_groundtrack(self, latlon label=None, color=None):
        """ Plots a precomputed trajectory.

        Parameters
        ----------
        latlon: ~astropy.coordinates.BaseRepresentation
            Trajectory in latitude and longitude representation to plot
        label: str, optional
            Label
        color: str, optional
            Color string
        """



if __name__ == "__main__":
    groundtrack(iss.propagate(time.Time.now()), 5 * u.h)
    plt.show()
