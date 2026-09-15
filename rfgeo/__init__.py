"""rfgeo — synthetic RF emitter geolocation with multivariate (NIW) DER + CP.

Physics/data layer (PHASE 1, unchanged): crlb, waveform, channel, geometry,
dataset. Representation (PHASE 2b, unchanged): gcc (GCC-PHAT, the fixed
preprocessing that remains the main pipeline). UQ layer (upgraded): evidential
(Meinert NIW, replaces Amini NIG) and conformal (elliptical CP, replaces
circular/radial CP).
"""
from . import crlb, waveform, channel, geometry, dataset, gcc  # noqa: F401
from . import evidential, conformal                              # noqa: F401
