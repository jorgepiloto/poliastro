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

# Plotting in 3D

```{code-cell}
import numpy as np

from poliastro.bodies import Earth, Sun
from poliastro.constants import J2000
from poliastro.examples import churi, iss, molniya
from poliastro.plotting import OrbitPlotter3D
from poliastro.twobody import Orbit
```

```{code-cell}
# More info: https://plotly.com/python/renderers/
import plotly.io as pio

pio.renderers.default = "plotly_mimetype+notebook_connected"
```

```{code-cell}
churi.plot(interactive=True, use_3d=True)
```

```{code-cell}
frame = OrbitPlotter3D()

frame.plot(churi)
frame.plot_body_orbit(Earth, J2000)
```

```{code-cell}
frame = OrbitPlotter3D()

frame.plot(molniya)
frame.plot(iss)
```

```{code-cell}
eros = Orbit.from_sbdb("eros")

frame = OrbitPlotter3D()

frame.plot_body_orbit(Earth, J2000)
frame.plot(eros, label="eros")
```

```{code-cell}
from astropy.time import Time
from poliastro.ephem import Ephem
from poliastro.util import time_range
```

```{code-cell}
date_launch = Time("2011-11-26 15:02", scale="utc").tdb
date_arrival = Time("2012-08-06 05:17", scale="utc").tdb

earth = Ephem.from_body(
    Earth, time_range(date_launch, end=date_arrival, num_values=50)
)
```

```{code-cell}
frame = OrbitPlotter3D()
frame.set_attractor(Sun)

frame.plot_body_orbit(Earth, J2000, label=Earth)
frame.plot_ephem(earth, label=Earth)
```

```{code-cell}
frame = OrbitPlotter3D()

frame.plot(eros, label="eros")
frame.plot_trajectory(earth.sample(), label=Earth)
```
