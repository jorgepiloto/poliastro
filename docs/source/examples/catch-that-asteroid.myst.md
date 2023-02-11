---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Catch that asteroid!

+++

First, we need to increase the timeout time to allow the download of data occur properly:

```{code-cell} ipython3
from astropy.utils.data import conf

conf.dataurl
```

```{code-cell} ipython3
conf.remote_timeout
```

```{code-cell} ipython3
conf.remote_timeout = 10000
```

Then, we do the rest of the imports:

```{code-cell} ipython3
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import solar_system_ephemeris

solar_system_ephemeris.set("jpl")

from poliastro.bodies import Sun, Earth, Moon
from poliastro.ephem import Ephem
from poliastro.frames import Planes
from poliastro.plotting import OrbitPlotter
from poliastro.plotting.misc import plot_solar_system
from poliastro.twobody import Orbit
from poliastro.util import norm, time_range

EPOCH = Time("2017-09-01 12:05:50", scale="tdb")
C_FLORENCE = "#000"
C_MOON = "#999"
```

```{code-cell} ipython3
Earth.plot(EPOCH)
```

Our first option to retrieve the orbit of the Florence asteroid is to use `Orbit.from_sbdb`, which gives us the osculating elements at a certain epoch:

```{code-cell} ipython3
florence_osc = Orbit.from_sbdb("Florence")
florence_osc
```

However, the epoch of the result is not close to the time of the close approach we are studying:

```{code-cell} ipython3
florence_osc.epoch.iso
```

Therefore, if we `propagate` this orbit to `EPOCH`, the results will be a bit different from the reality. Therefore, we need to find some other means.

Let's use the `Ephem.from_horizons` method as an alternative, sampling over a period of 6 months:

```{code-cell} ipython3
epochs = time_range(
    EPOCH - TimeDelta(3 * 30 * u.day), end=EPOCH + TimeDelta(3 * 30 * u.day)
)
```

```{code-cell} ipython3
florence = Ephem.from_horizons("Florence", epochs, plane=Planes.EARTH_ECLIPTIC)
florence
```

```{code-cell} ipython3
florence.plane
```

And now, let's compute the distance between Florence and the Earth at that epoch:

```{code-cell} ipython3
earth = Ephem.from_body(Earth, epochs, plane=Planes.EARTH_ECLIPTIC)
earth
```

```{code-cell} ipython3
min_distance = norm(florence.rv(EPOCH)[0] - earth.rv(EPOCH)[0]) - Earth.R
min_distance.to(u.km)
```

<div class="alert alert-success">This value is consistent with what ESA says! $7\,060\,160$ km</div>

```{code-cell} ipython3
abs((min_distance - 7060160 * u.km) / (7060160 * u.km)).decompose()
```

```{code-cell} ipython3
from IPython.display import HTML

HTML(
    """<blockquote class="twitter-tweet" data-lang="en"><p lang="es" dir="ltr">La <a href="https://twitter.com/esa_es">@esa_es</a> ha preparado un resumen del asteroide <a href="https://twitter.com/hashtag/Florence?src=hash">#Florence</a> 😍 <a href="https://t.co/Sk1lb7Kz0j">pic.twitter.com/Sk1lb7Kz0j</a></p>&mdash; AeroPython (@AeroPython) <a href="https://twitter.com/AeroPython/status/903197147914543105">August 31, 2017</a></blockquote>
<script src="//platform.twitter.com/widgets.js" charset="utf-8"></script>"""
)
```

And now we can plot!

```{code-cell} ipython3
:tags: [nbsphinx-thumbnail]

frame = plot_solar_system(outer=False, epoch=EPOCH)
frame.plot_ephem(florence, EPOCH, label="Florence", color=C_FLORENCE)
```

Finally, we are going to visualize the orbit of Florence with respect to the Earth. For that, we set a narrower time range, and specify that we want to retrieve the ephemerides with respect to our planet:

```{code-cell} ipython3
epochs = time_range(
    EPOCH - TimeDelta(5 * u.day), end=EPOCH + TimeDelta(5 * u.day)
)
```

```{code-cell} ipython3
florence_e = Ephem.from_horizons("Florence", epochs, attractor=Earth)
florence_e
```

We now retrieve the ephemerides of the Moon, which are given directly in GCRS:

```{code-cell} ipython3
moon = Ephem.from_body(Moon, epochs, attractor=Earth)
moon
```

```{code-cell} ipython3
plotter = OrbitPlotter()
plotter.set_attractor(Earth)
plotter.set_body_frame(Moon)
plotter.plot_ephem(moon, EPOCH, label=Moon, color=C_MOON)
```

And now, the glorious final plot:

```{code-cell} ipython3
from matplotlib import pyplot as plt

frame = OrbitPlotter()

frame.set_attractor(Earth)
frame.set_orbit_frame(Orbit.from_ephem(Earth, florence_e, EPOCH))

frame.plot_ephem(florence_e, EPOCH, label="Florence", color=C_FLORENCE)
frame.plot_ephem(moon, EPOCH, label=Moon, color=C_MOON)
```
