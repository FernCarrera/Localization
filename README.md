# Localization and Control

This script simulates a small vehicle attempting to follow a randomly generated path.
The vehicle is only aware of its position through the use of a sensor that can provide distance and heading information to known landmarks. The data is then ran through an Unscented Kalman filter using Van der Merwe sigma point generation. Currently, the filter is only tracking position and heading information, and the vehicle can receive velocity and heading commands.

The green ellipses are the covariance ellipses for the position of the vehicle.