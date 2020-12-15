# Grid_CarbonIntensity_Modelling
Abstract:
The objective of this study is to predict Grid Carbon from 2021 to 2050 for a lag duration of 30 minutes interval. The time series data is modelled using a deep neural network (DNN), Long-short term memory model, (LSTM) to improve confidence in prediction and pick up non linearity in dependence of the features. There are 6 features in the time series data namely; Bio mass, Fossil Fuel, Interconnectors, Nuclear, Wind and other renewables. Original dataset was given from 1/01/2017 00:00:00 till 04/12/2020 16:00:00, from this data from 01/01/2017 00:00:00 to 31/12/2019 23:30:00 was taken as training set and data from 01/01/2020 00:00:00 to 04/12/2020 16:00:00 was taken as test set to conceptualise and confirm the ideologies to be implemented . A target sum total of trend for each of the 6 features was to be tracked. Prediction from the time series model for 4 different scenarios where also discussed. The scenarios are namely; consumer & system transformation, leading the way and steady progression all labelled “scenario 1-4” respectively. The approach in this work accurately predicted Grid carbon scenarios for each of the scenarios for the years 2021-2050. Fossil Fuel contribution to totally energy demand fizzles out in year 2050 for the scenario consumer transformation and year 2045 for the scenario leading the way . In foresight, DNN-LSTM machines provide an accurate and robust prediction capabilities, and methods developed in this report could be implemented and considered for future energy policies.

Getting Started:
---------------------------------
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.


Prerequisites:
-------------------------------
MATLAB 2018 upwards

Methods:
-------------------------------

Datasets
-----------------------------

Running the Numerical Experiment:
Run the script Main.m  

Dependencies
----------------------------
- ksvdbox13
- ompbox10
All downloaded for your convenience

All libraries are included for your convenience.

Manuscript
-----------------------------

Extras
--------------------------------------
Extra methods are included also;
- Running supervised learning models with DNN and MLP alone (Requires the netlab and MATLAB DNN tool box)
- Running CCR/MM and MMr with DNN/DNN for the experts and gates respectively (Requires MATLAB DNN toolbox)
- Running the MMr method for using Sparse Gp experts/DNN experts/RF experts and DNN/RF gates
- Running CCR/CCR-MM and MM-MM with RandomForest Experts and RandomForest Gates. This method is fast and also gives a measure of uncerntainty

Author:
--------------------------------
Dr Clement Etienam- Research Officer-Machine Learning. Active Building Centre


Acknowledgments:
------------------------------


References:
----------------------------

[1] Luca Ambrogioni, Umut Güçlü, Marcel AJ van Gerven, and Eric Maris. The kernel mixture network: A non-parametric method for conditional density estimation 
of continuous random variables. arXiv preprint arXiv:1705.07111, 2017.

[2] Christopher M Bishop. Mixture density networks. 1994.

[3] Isobel C. Gormley and Sylvia Frühwirth-Schnatter. Mixtures of Experts Models. Chapman and Hall/CRC, 2019.

[4] R.B. Gramacy and H.K. Lee. Bayesian treed Gaussian process models with an application to computer modeling. Journal of the American Statistical Association, 103(483):1119–1130,
2008.

[5] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, Geoffrey E Hinton, et al. Adaptive
mixtures of local experts. Neural computation, 3(1):79–87, 1991.
2

[6] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm.
Neural computation, 6(2):181–214, 1994.

[7] Trung Nguyen and Edwin Bonilla. Fast allocation of gaussian process experts. In International
Conference on Machine Learning, pages 145–153, 2014.

[8] Carl E Rasmussen and Zoubin Ghahramani. Infinite mixtures of gaussian process experts. In
Advances in neural information processing systems, pages 881–888, 2002.

[9] Tommaso Rigon and Daniele Durante. Tractable bayesian density regression via logit stickbreaking
priors. arXiv preprint arXiv:1701.02969, 2017.

[10] Volker Tresp. Mixtures of gaussian processes. In Advances in neural information processing
systems, pages 654–660, 2001.

[11] Lei Xu, Michael I Jordan, and Geoffrey E Hinton. An alternative model for mixtures of experts.

[12] Rasmussen, Carl Edward and Nickisch, Hannes. Gaussian processes for machine learning (gpml) toolbox. The
Journal of Machine Learning Research, 11:3011–3015, 2010

[13] David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park, Kody J. H. Law, and
Clement Etienam. Cluster, classify, regress: A general method for learning discontinuous functions. Foundations of Data Science, 
1(2639-8001-2019-4-491):491, 2019.

[14] Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.
