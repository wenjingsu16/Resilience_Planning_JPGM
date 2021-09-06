# Resilience_Planning_JPGM

This repository stores python scripts for two-stage stochastic resilience planning for joint electricity and gas system for an IEEE paper (An link will be updated when the paper is published). The GIS data processing is achieved by geopandas package and optimization problem is constructed using Pyomo package.

The Impact Comparison folder compares the impact on spatially correlated failures induced by extreme weather and climate events and uncorrelated failures in traditional N-k contingency analysis. The model used is a one-stage (operation stage) deterministic optimization model for the basenetwork.

The folder of Geographically Correlated Failures stores scripts that generate geographically correlated failures within the system boundary through Monte Carlo and compute cost matrix for contingency scenarios and 30 sample expanded network structures through parallel processing. It also contains the MATLAB code for scenario reduction via ALFA. ALFA is a machine learning technique for scenario reduction based on Principla Component Analysis. The scenario reduction method is needed because the uncertainty of correlated failure locations for correlated failures and the possible combinations of failed components for makes it critical to incorporate several scenarios while including many scenarios results in the heavy computational burden because of the non-linearity of the gas system and joint operation of two subsystems.

The folder of Random Failures contains scripts that generate spatially random/uncorrelated failures throught Monte Carlo and compute the total system cost matrix. It also contains the MATLAB code for scenario reduction.

The folder of Expansion Comparison has the python scripts to solve the two-stage stochastic planning model for joint power and gas system. It also has the data 
processing and visualization script using geopandas and matplotlib to compare the expansion decision. After running the planning model, the model checks the resilience of expanded network. The stochastic planning model is a mixed integer non-linear programming problem.

The topology of joint electricity and natural gas system is shown in the figure below.
![Joint power and gas sytem with node labels_with_circles](https://user-images.githubusercontent.com/34109639/132237659-a727b1d0-0408-4433-b128-23d1cb9917b8.png)

The formulation for the two-stage stochastic planning model is as following. Nomenclature is first presented.

```math

\renewcommand\nomgroup[1]{%
    \item[\bfseries
    \ifstrequal{#1}{A}{Power System Sets}{%
    \ifstrequal{#1}{B}{Power System Parameters}{%
    \ifstrequal{#1}{C}{Power System Variables}{%
    \ifstrequal{#1}{D}{Natural Gas System Sets}{%
    \ifstrequal{#1}{E}{Natural Gas System Parameters}{%
    \ifstrequal{#1}{F}{Natural Gas System Variables}{%
    \ifstrequal{#1}{G}{Extreme Events and Expansion Sets}{%
    \ifstrequal{#1}{H}{Extreme Events and Expansion Parameters}{%
    \ifstrequal{#1}{I}{Extreme Events and Expansion Variables}{
    }}}}}}}}}%
]}

% Power system sets 
\nomenclature[A,01]{$N^{e}$}{Set of electric power buses (nodes)}
\nomenclature[A,02]{$\Omega$}{Set of power generators}
\nomenclature[A,03]{$\Omega^{gf}$}{Set of gas-fired power generators, $\Omega^{g} \subseteq \Omega$}
\nomenclature[A,04]{$\Omega^{ngf}$}{Set of non-gas-fired power generators, $\Omega^{ng} \subseteq \Omega$}
\nomenclature[A,05]{$\Gamma_i$}{Set of generators connected to bus $i$}
\nomenclature[A,06]{$N_i^e$}{Set of buses connected to bus $i$ by an edge}
\nomenclature[A,07]{$A^e$}{Set of power transmission lines}

% Power system Parameters
\nomenclature[B,01]{$\delta_0$}{Index of the reference bus}
\nomenclature[B,02]{$P_i^l$}{Nodal active power load at bus $i$}
\nomenclature[B,03]{$\underline{P_i^e}, \overline{P_i^e}$}{Active power generation limits of generator $i$}
\nomenclature[B,04]{$C_1^i,C_2^i$}{Cost coefficients of power generator $i$}
\nomenclature[B,05]{$H_i$}{Heat rate coefficient of gas-fired power generators}
\nomenclature[B,06]{$X_{ij}$}{Reactance of a transmission line}
\nomenclature[B,07]{$\overline{F_{ij}}$}{Thermal limit/capacity of a transmission line}
\nomenclature[B,08]{$M$}{A large penalty constant for Big M method}

% Power system variables
\nomenclature[C,01]{$p_{ij,k}$}{Active power of a transmission line in event $k$}
\nomenclature[C,02]{$p_{j,k}^e$}{Active power output of generator $j$ in event $k$}
\nomenclature[C,03]{$\theta_{i,k}$}{Phase angle at bus $i$ in event $k$}
\nomenclature[C,04]{$use_{i,k}$}{Unserved electricity demand of node $i$ in event $k$}

% Gas system sets
\nomenclature[D,01]{$N^g$}{Set of natural gas junctions (nodes)}
\nomenclature[D,02]{$A^g$}{Set of all links joining a pair of junctions}
\nomenclature[D,03]{$T_i$}{Set of gas-fired power plants connected to node $i$}
\nomenclature[D,04]{$A^p$}{Set of base pipelines, subset of $A^g$}
\nomenclature[D,05]{$A^c$}{Set of base compressors, subset of $A^g$}

% Gas system parameters
\nomenclature[E,01]{$W_a$}{Pipeline resistance (Weymouth) factor}
\nomenclature[E,02]{$\underline{PD_{ij}},\overline{PD_{ij}}$}{Pressure drop limits from junction $i$ to juntion $j$}
\nomenclature[E,03]{$\underline{\pi_i},\overline{\pi_i}$}{Squared pressure limits at junction $i$}
\nomenclature[E,04]{$\underline{\alpha_{ij}^c},\overline{\alpha_{ij}^c}$}{Compression limits squared at compressor station}
\nomenclature[E,05]{$D_i$}{Firm gas consumption at junction $i$???}
\nomenclature[E,06]{$\underline{D_i}, \overline{D_i}$}{Gas consumption limits at junction $i$}
\nomenclature[E,07]{$\underline{S_i},\overline{S_i}$}{Gas product limits at junction $i$}
\nomenclature[E,08]{$Y_i$}{Cost coefficient of gas production at junction $i$}


% Gas system variables
\nomenclature[F,01]{$\pi_{i,k}$}{Squared pressure of gas node $i$ in event $k$}
\nomenclature[F,02]{$x_{ij,k}$}{Gas flow on pipelines and compressors in event $k$}
\nomenclature[F,03]{$\lambda_{ij,k}$}{Auxiliary relaxation variable in event $k$}
\nomenclature[F,04]{$d_{i,k}$}{Gas consumption at junction $i$ in event $k$}
\nomenclature[F,05]{$s_{i,k}$}{Gas production at junction $i$ in event $k$}
\nomenclature[F,06]{$y_{ij,k}^+,y_{ij,k}^-$}{Binary flow direction for links in event $k$}



% Extreme events and expansion sets
\nomenclature[G,01]{$k \in K$}{Set of extreme weather and climate events}
\nomenclature[G,02]{$\Lambda^t$}{Set of transmission expansion candidates}
\nomenclature[G,03]{$\Lambda^p$}{Set of pipeline expansion candidates}
\nomenclature[G,04]{$\Phi_{k}^t$}{Set of transmission lines impacted in event $k$}
\nomenclature[G,05]{$\Phi_{k}^p$}{Set of pipelines impacted in event $k$ }

% Extreme events and expansion parameters
\nomenclature[H,01]{$\beta_{ij}^p$}{Expansion cost of a pipeline}
\nomenclature[H,02]{$\beta_{ij}^t$}{Expansion cost of a transmission line}
\nomenclature[H,03]{$\eta$}{Penalty cost for 1 MWh of unserved energy}
\nomenclature[H,04]{$ST_{ij,k}$}{Binary status of an existing line in event $k$}
\nomenclature[H,05]{$SP_{ij,k}$}{Binary status of existing pipelines in event $k$}


% Extreme events and expansion variables
\nomenclature[I,01]{$use_{i,k}$}{Unserved electricity demand of node $i$ in event $k$}
\nomenclature[I,02]{$usg_{i,k}$}{Unserved gas demand of junction $i$ in event $k$}
\nomenclature[I,03]{$z_{ij}^p$}{Binary expansion decision for pipeline candidates }
\nomenclature[I,04]{$z_{ij}^t$}{Binary expansion decision for transmission line candidates}
\printnomenclature[0.6in]

```



