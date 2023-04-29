Final project for Astronomy 390 - Computational Astrophysics.

The goal of this project is to:

1) Create a new symplectic integration algorithm based upon the 4th-Order Yoshida Method.
2) Use this algorithm to plot the orbits of the exoplanets around Tau Ceti.


This page is organized as follows:

earth - contains initial codes that test the Yoshida method on just our home planet.

euler - a program using Euler's Method of Integration instead of the Yoshida Method, mainly to compare errors between methods.

solar_system - expand the Yoshida code to account for a multi-body system.

tau_ceti - directory containing the main simulation files for Tau Ceti.
