from poliastro.plotting.misc import plot_solar_system
from poliastro.plotting.orbit.backends import PyVista3D

pyvista_backend = PyVista3D(use_stars_background=True, use_planets_textures=True)
plotter = plot_solar_system(backend=pyvista_backend)
plotter.show()
