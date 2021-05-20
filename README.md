# Resilience_Planning_JPGM

This repository stored scripts for resilience planning for joint power and gas system for an IEEE paper.

The Impact Comparison folder compares the impact on spatially correlated failures induced by extreme weather and climate events and uncorrelated failures in traditional 
N-k contingency analysis. The model used is a one-stage (operation stage) deterministic optimization model for the basenetwork.

The folder of Geographically Correlated Failures stores scripts that generate geographically correlated failures through Monte Carlo and get cost matrix for 100 
contingency scenarios and 30 sample network structures throught multiprocessing. It also contains the MATLAB code for scenario reduction. 

The folder of Random Failures contains scripts that generate spatially random failures throught Monte Carlo and get the cost matrix. It also contains the MATLAB 
code for scenario reduction.

The folder of Expansion Comparison has the python scripts to solve the two-stage stochastic planning model for joint power and gas system. It also has the data 
processing and visualization code using geopandas and matplotlib to compare the expansion decision. After the planning model, the code to check the resilience of 
expanded network is also in this folder.

