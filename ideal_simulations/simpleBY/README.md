Simple BY Simulations
====================

These simulations are very basic investigations of the effects of IMF
BY on coupled MHD-ring current-ionosphere simulations.

Other configurations (e.g., expanded coupling, additional models, etc.)
input files may be included here. 

Shared settings between codes:
------------------------------

- Standard Initialization: 2 hours southward -5nT, 4 hours northward IMF +5nT, then switch to idealized inputs.
- no dipole tilt, centered dipole
- F10.7: 150 for starters.

Critical Model Outputs:
-----------------------
Here is a list of critical outputs for inter-model comparisons.  The more
output the better, but these should be considered baseline:

- Full 3D output every 5 minutes or more frequently.
- Ionospheric electrodynamic data every 1 minute.
- Virtual magnemeters world wide at a 5 degree separation in lat and lon.

Model Specific Details
======================

SWMF
----
The default setup couples BATS-R-US, CIMI, and the Ridley_serial ionosphere.
The CMEE conductance model is used [Mukhophadhyay et al, 2020, Space Weather].
A customized, high resolution grid is used.