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

# Studying Hohmann transfers

```{code-cell} ipython3
from astropy import units as u

from matplotlib import pyplot as plt
import numpy as np

from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.util import norm
```

```{code-cell} ipython3
Earth.k
```

```{code-cell} ipython3
orb_i = Orbit.circular(Earth, alt=800 * u.km)
orb_i
```

```{code-cell} ipython3
r_i = orb_i.a.to(u.km)
r_i
```

```{code-cell} ipython3
v_i_vec = orb_i.v.to(u.km / u.s)
v_i = norm(v_i_vec)
v_i
```

```{code-cell} ipython3
N = 1000
dv_a_vector = np.zeros(N) * u.km / u.s
dv_b_vector = dv_a_vector.copy()
r_f_vector = r_i * np.linspace(1, 100, num=N)
for ii, r_f in enumerate(r_f_vector):
    man = Maneuver.hohmann(orb_i, r_f)
    (_, dv_a), (_, dv_b) = man.impulses
    dv_a_vector[ii] = norm(dv_a)
    dv_b_vector[ii] = norm(dv_b)
```

```{code-cell} ipython3
:tags: [nbsphinx-thumbnail]

fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(
    (r_f_vector / r_i).value, (dv_a_vector / v_i).value, label="First impulse"
)
ax.plot(
    (r_f_vector / r_i).value, (dv_b_vector / v_i).value, label="Second impulse"
)
ax.plot(
    (r_f_vector / r_i).value,
    ((dv_a_vector + dv_b_vector) / v_i).value,
    label="Total cost",
)

ax.plot((r_f_vector / r_i).value, np.full(N, np.sqrt(2) - 1), "k--")
ax.plot((r_f_vector / r_i).value, (1 / np.sqrt(r_f_vector / r_i)).value, "k--")

ax.set_ylim(0, 0.7)
ax.set_xlabel("$R$")
ax.set_ylabel("$\Delta v_a / v_i$")

plt.legend()
fig.savefig("hohmann.png")
```
