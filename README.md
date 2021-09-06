# Resilience_Planning_JPGM

This repository stores python scripts for two-stage stochastic resilience planning for joint electricity and gas system for an IEEE paper (An link will be updated when the paper is published). The GIS data processing is achieved by geopandas package and optimization problem is constructed using Pyomo package.

The Impact Comparison folder compares the impact on spatially correlated failures induced by extreme weather and climate events and uncorrelated failures in traditional N-k contingency analysis. The model used is a one-stage (operation stage) deterministic optimization model for the basenetwork.

The folder of Geographically Correlated Failures stores scripts that generate geographically correlated failures within the system boundary through Monte Carlo and compute cost matrix for contingency scenarios and 30 sample expanded network structures through parallel processing. It also contains the MATLAB code for scenario reduction via ALFA. ALFA is a machine learning technique for scenario reduction based on Principla Component Analysis. The scenario reduction method is needed because the uncertainty of correlated failure locations for correlated failures and the possible combinations of failed components for makes it critical to incorporate several scenarios while including many scenarios results in the heavy computational burden because of the non-linearity of the gas system and joint operation of two subsystems.

The folder of Random Failures contains scripts that generate spatially random/uncorrelated failures throught Monte Carlo and compute the total system cost matrix. It also contains the MATLAB code for scenario reduction.

The folder of Expansion Comparison has the python scripts to solve the two-stage stochastic planning model for joint power and gas system. It also has the data 
processing and visualization script using geopandas and matplotlib to compare the expansion decision. After running the planning model, the model checks the resilience of expanded network. The stochastic planning model is a mixed integer non-linear programming problem.

The topology of joint electricity and natural gas system is shown in the figure below.
![Joint power and gas sytem with node labels_with_circles](https://user-images.githubusercontent.com/34109639/132237659-a727b1d0-0408-4433-b128-23d1cb9917b8.png)

The formulation for the two-stage stochastic planning model is in the pdf file.





