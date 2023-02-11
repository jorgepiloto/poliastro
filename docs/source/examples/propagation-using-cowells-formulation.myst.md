---
jupytext:
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

# Cowell's formulation

For cases where we only study the gravitational forces, solving the Kepler's equation is enough to propagate the orbit forward in time. However, when we want to take perturbations that deviate from Keplerian forces into account, we need a more complex method to solve our initial value problem: one of them is **Cowell's formulation**.

In this formulation we write the two body differential equation separating the Keplerian and the perturbation accelerations:

$$\ddot{\mathbb{r}} = -\frac{\mu}{|\mathbb{r}|^3} \mathbb{r} + \mathbb{a}_d$$

+++

<div class="alert alert-info">For an in-depth exploration of this topic, still to be integrated in poliastro, check out [this Master thesis](https://github.com/Juanlu001/pfc-uc3m).</div>

+++

<div class="alert alert-info">An earlier version of this notebook allowed for more flexibility and interactivity, but was considerably more complex. Future versions of poliastro and plotly might bring back part of that functionality, depending on user feedback. You can still download the older version [here](https://github.com/poliastro/poliastro/blob/0.8.x/docs/source/examples/Propagation%20using%20Cowell's%20formulation.ipynb).</div>

+++

## First example

Let's setup a very simple example with constant acceleration to visualize the effects on the orbit:

```{code-cell} ipython3
from astropy import time
from astropy import units as u

import numpy as np

from poliastro.bodies import Earth
from poliastro.core.propagation import func_twobody
from poliastro.examples import iss
from poliastro.plotting import OrbitPlotter
from poliastro.plotting.orbit.backends import Plotly3D
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.twobody.sampling import EpochsArray
from poliastro.util import norm
```

```{code-cell} ipython3
# More info: https://plotly.com/python/renderers/
import plotly.io as pio

pio.renderers.default = "plotly_mimetype+notebook_connected"
```

To provide an acceleration depending on an extra parameter, we can use **closures** like this one:

```{code-cell} ipython3
accel = 2e-5
```

```{code-cell} ipython3
def constant_accel_factory(accel):
    def constant_accel(t0, u, k):
        v = u[3:]
        norm_v = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        return accel * v / norm_v

    return constant_accel
```

```{code-cell} ipython3
def f(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = constant_accel_factory(accel)(t0, state, k)
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad
```

```{code-cell} ipython3
times = np.linspace(0, 10 * iss.period, 500)
times
```

```{code-cell} ipython3
ephem = iss.to_ephem(
    EpochsArray(iss.epoch + times, method=CowellPropagator(rtol=1e-11, f=f)),
)
```

And we plot the results:

```{code-cell} ipython3
frame = OrbitPlotter(backend=Plotly3D())

frame.set_attractor(Earth)
frame.plot_ephem(ephem, label="ISS")
```

## Error checking

```{code-cell} ipython3
def state_to_vector(ss):
    r, v = ss.rv()
    x, y, z = r.to_value(u.km)
    vx, vy, vz = v.to_value(u.km / u.s)
    return np.array([x, y, z, vx, vy, vz])
```

```{code-cell} ipython3
k = Earth.k.to(u.km**3 / u.s**2).value
```

```{code-cell} ipython3
rtol = 1e-13
full_periods = 2
```

```{code-cell} ipython3
u0 = state_to_vector(iss)
tf = (2 * full_periods + 1) * iss.period / 2

u0, tf
```

```{code-cell} ipython3
iss_f_kep = iss.propagate(tf)
```

```{code-cell} ipython3
iss_f_num = iss.propagate(tf, method=CowellPropagator(rtol=rtol))
```

```{code-cell} ipython3
iss_f_num.r, iss_f_kep.r
```

```{code-cell} ipython3
assert np.allclose(iss_f_num.r, iss_f_kep.r, rtol=rtol, atol=1e-08 * u.km)
assert np.allclose(
    iss_f_num.v, iss_f_kep.v, rtol=rtol, atol=1e-08 * u.km / u.s
)
```

```{code-cell} ipython3
assert np.allclose(iss_f_num.a, iss_f_kep.a, rtol=rtol, atol=1e-08 * u.km)
assert np.allclose(iss_f_num.ecc, iss_f_kep.ecc, rtol=rtol)
assert np.allclose(iss_f_num.inc, iss_f_kep.inc, rtol=rtol, atol=1e-08 * u.rad)
assert np.allclose(
    iss_f_num.raan, iss_f_kep.raan, rtol=rtol, atol=1e-08 * u.rad
)
assert np.allclose(
    iss_f_num.argp, iss_f_kep.argp, rtol=rtol, atol=1e-08 * u.rad
)
assert np.allclose(iss_f_num.nu, iss_f_kep.nu, rtol=rtol, atol=1e-08 * u.rad)
```

## Numerical validation

According to [Edelbaum, 1961], a coplanar, semimajor axis change with tangent thrust is defined by:

$$\frac{\operatorname{d}\!a}{a_0} = 2 \frac{F}{m V_0}\operatorname{d}\!t, \qquad \frac{\Delta{V}}{V_0} = \frac{1}{2} \frac{\Delta{a}}{a_0}$$

So let's create a new circular orbit and perform the necessary checks, assuming constant mass and thrust (i.e. constant acceleration):

```{code-cell} ipython3
orb = Orbit.circular(Earth, 500 << u.km)
tof = 20 * orb.period

ad = constant_accel_factory(1e-7)


def f(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = ad(t0, state, k)
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad


orb_final = orb.propagate(tof, method=CowellPropagator(f=f))
```

```{code-cell} ipython3
da_a0 = (orb_final.a - orb.a) / orb.a
da_a0
```

```{code-cell} ipython3
dv_v0 = abs(norm(orb_final.v) - norm(orb.v)) / norm(orb.v)
2 * dv_v0
```

```{code-cell} ipython3
np.allclose(da_a0, 2 * dv_v0, rtol=1e-2)
```

This means **we successfully validated the model against an extremely simple orbit transfer with an approximate analytical solution**. Notice that the final eccentricity, as originally noticed by Edelbaum, is nonzero:

```{code-cell} ipython3
orb_final.ecc
```

## References

* [Edelbaum, 1961] "Propulsion requirements for controllable satellites"
