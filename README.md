# Localization and Control

This script simulates a small vehicle attempting to follow a randomly generated path.
The vehicle is only aware of its position through the use of a sensor that can provide distance and heading information to known landmarks. The data is then ran through an Unscented Kalman filter using Van der Merwe sigma point generation. Currently, the filter is only tracking position and heading information, and the vehicle can receive velocity and heading commands.

Currently testing a kinematic Stanley controller, can view 
progress in simple_sim.py

The green ellipses are the covariance ellipses for the position of the vehicle.


![Sim_Example](/img/SS_sim_preview.png)

![Metrics_Example](/img/SS_speed_lateral.png)