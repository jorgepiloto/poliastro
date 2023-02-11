---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Drawing Earth satellite groundtracks

## What are orbit groundtracks?

By definition, a groundtrack is just the projection of the position left by a
satellite over its attractor. They are usually applied to Earth orbiting
spacecraft and thus, have been implemented within the `poliastro.earth.plotting`
sub-package.

Something interesting about these kind of figures is that they take the Earth's rotation into account.
Therefore, it is possible to predict over which locations in
the planet will the spacecraft be within the next hours or days.

In this notebook, we will show all the possibilities that the
`GrountrackPlotter` class offers to poliastro's users. Let us start by importing
some useful modules!

```{code-cell}
# Useful for defining quantities
from astropy import units as u

# Earth focused modules, ISS example orbit and time span generator
from poliastro.earth import EarthSatellite
from poliastro.earth.plotting import GroundtrackPlotter
from poliastro.examples import iss
from poliastro.util import time_range
```

## EarthSatellite instance and desired time span

As said before, groundtrack figures are usually related with Earth capabilities.
Because of this, the plotter requires an `EarthSatellite` object to be passed
and not a simple `Orbit` one. Main differences among them are that the first one
is a combination of `Orbit`+ `Spacecraft`, including not only orbital data but
also other one related with aerodynamic properties, for example. For more
information, please refer to official API documentation.

Let us build an `EarthSatellite` instance but without any associated
`Spacecraft` data, since for this notebook they will not be useful at all.
Furthermore, we will generate a desired time span for the next three hours since
actual ISS epoch. Notice that the `periods` parameter controls the resolution of
the time vector and thus, the amount of values:

```{code-cell}
# Build spacecraft instance
iss_spacecraft = EarthSatellite(iss, None)
t_span = time_range(
    iss.epoch - 1.5 * u.h, num_values=150, end=iss.epoch + 1.5 * u.h
)
```

## Using the plotter

Because of its interactive nature, lots of possibilities are offered by
`GroundtrackPlotter`. It can provide lots of different map projections,
customization parameters... This utility is based on `plotly.layout.geo`, please
refer to the [official
documentation](https://plotly.com/python/reference/layout/geo/) for more
information about it. Let us create a simple instance and start plotting:

```{code-cell}
# Generate an instance of the plotter, add title and show latlon grid
gp = GroundtrackPlotter()
gp.update_layout(title="International Space Station groundtrack")

# Plot previously defined EarthSatellite object
gp.plot(
    iss_spacecraft,
    t_span,
    label="ISS",
    color="red",
    marker={
        "size": 10,
        "symbol": "triangle-right",
        "line": {"width": 1, "color": "black"},
    },
)
```

It is even possible to add other kind of information rather than
`EarthSatellite`'s within previous figure. For example, we will show the actual
location of Madrid city:

```{code-cell}
# For building geo traces
import plotly.graph_objects as go

# Position in [LAT LON]
STATION = [40.416729, -3.703339] * u.deg

# Let us add a new trace in original figure
gp.add_trace(
    go.Scattergeo(
        lat=STATION[0],
        lon=STATION[-1],
        name="Madrid",
        marker={"color": "blue"},
    )
)
gp.fig.show()
```

Finally, it is also possible to make use of [several map
projections](https://plotly.com/python/reference/layout/geo/#layout-geo-projection).
Among the different possibilities, there is also one which holds 3D support:
`orthographic`. You can simply switch the previous figure by simply running:

```{code-cell}
# Switch to three dimensional representation
gp.update_geos(projection_type="orthographic")
gp.fig.show()
```
