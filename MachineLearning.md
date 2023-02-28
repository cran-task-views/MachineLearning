---
name: MachineLearning
topic: Machine Learning & Statistical Learning
maintainer: Torsten Hothorn
email: Torsten.Hothorn@R-project.org
version: 2022-03-07
source: https://github.com/cran-task-views/MachineLearning/
---

Several add-on packages implement ideas and methods developed at the
borderline between computer science and statistics - this field of
research is usually referred to as machine learning. The packages can be
roughly structured into the following topics:

-   *Neural Networks and Deep Learning* : Single-hidden-layer neural
    network are implemented in package
    `r pkg("nnet", priority = "core")` (shipped with base R).
    Package `r pkg("RSNNS")` offers an interface to the
    Stuttgart Neural Network Simulator (SNNS). Packages implementing
    deep learning flavours of neural networks include
    `r pkg("deepnet")` (feed-forward neural network,
    restricted Boltzmann machine, deep belief network, stacked
    autoencoders), `r pkg("RcppDL")` (denoising autoencoder,
    stacked denoising autoencoder, restricted Boltzmann machine, deep
    belief network) and `r pkg("h2o")` (feed-forward neural
    network, deep autoencoders). An interface to
    [tensorflow](http://www.tensorflow.org) is available in
    `r pkg("tensorflow")`. The `r pkg("torch")`
    package implements an interface to the [libtorch
    library](https://pytorch.org/). Prediction uncertainty can be quantified
    by the ENNreg evidential regression neural network model implemented 
    in `r pkg("evreg")`.
-   *Recursive Partitioning* : Tree-structured models for regression,
    classification and survival analysis, following the ideas in the
    CART book, are implemented in
    `r pkg("rpart", priority = "core")` (shipped with base R)
    and `r pkg("tree")`. Package
    `r pkg("rpart")` is recommended for computing CART-like
    trees. A rich toolbox of partitioning algorithms is available in
    [Weka](http://www.cs.waikato.ac.nz/~ml/weka/), package
    `r pkg("RWeka")` provides an interface to this
    implementation, including the J4.8-variant of C4.5 and M5. The
    `r pkg("Cubist")` package fits rule-based models
    (similar to trees) with linear regression models in the terminal
    leaves, instance-based corrections and boosting. The
    `r pkg("C50")` package can fit C5.0 classification
    trees, rule-based models, and boosted versions of these. 
    `r pkg("pre")` can fit rule-based models for a wider range of
    response variable types.\
    Two recursive partitioning algorithms with unbiased variable
    selection and statistical stopping criterion are implemented in
    package `r pkg("party")` and
    `r pkg("partykit")`. Function `ctree()` is based on
    non-parametric conditional inference procedures for testing
    independence between response and each input variable whereas
    `mob()` can be used to partition parametric models. Extensible tools
    for visualizing binary trees and node distributions of the response
    are available in package `r pkg("party")` and
    `r pkg("partykit")` as well. Partitioning of mixed-effects models
    (GLMMs) can be performed with package `r pkg("glmertree")`;
    partitioning of structural equation models (SEMs) can be performed 
    with package `r pkg("semtree")`.\ 
    Graphical tools for the visualization of trees are available in
    package `r pkg("maptree")`.\
    Partitioning of mixture models is performed by
    `r pkg("RPMM")`.\
    Computational infrastructure for representing trees and unified
    methods for prediction and visualization is implemented in
    `r pkg("partykit")`. This infrastructure is used by
    package `r pkg("evtree")` to implement evolutionary
    learning of globally optimal trees. Survival trees are available in
    various packages. 

    Trees for subgroup identification with respect to heterogenuous
    treatment effects  are available in packages `r pkg("partykit")`,
    `r pkg("model4you")`, `r pkg("dipm")`, `r pkg("quint")`, 
    `pkg("SIDES")`, `pkg("psica")`, and `pkg("MrSGUIDE")` (and
    probably many more).

-   *Random Forests* : The reference implementation of the random forest
    algorithm for regression and classification is available in package
    `r pkg("randomForest", priority = "core")`. Package
    `r pkg("ipred")` has bagging for regression,
    classification and survival analysis as well as bundling, a
    combination of multiple models via ensemble learning. In addition, a
    random forest variant for response variables measured at arbitrary
    scales based on conditional inference trees is implemented in
    package `r pkg("party")`.
    `r pkg("randomForestSRC")` implements a unified
    treatment of Breiman's random forests for survival, regression and
    classification problems. Quantile regression forests
    `r pkg("quantregForest")` allow to regress quantiles of
    a numeric response on exploratory variables via a random forest
    approach. For binary data, The `r pkg("varSelRF")` and
    `r pkg("Boruta")` packages focus on variable selection
    by means for random forest algorithms. In addition, packages
    `r pkg("ranger")` and `r pkg("Rborist")`
    offer R interfaces to fast C++ implementations of random forests.
    Reinforcement Learning Trees, featuring splits in variables which
    will be important down the tree, are implemented in package
    `r pkg("RLT")`. `r pkg("wsrf")` implements
    an alternative variable weighting method for variable subspace
    selection in place of the traditional random variable sampling.
    Package `r pkg("RGF")` is an interface to a Python
    implementation of a procedure called regularized greedy forests.
    Random forests for parametric models, including forests for the
    estimation of predictive distributions, are available in packages
    `r pkg("trtf")` (predictive transformation forests,
    possibly under censoring and trunction) and
    `r pkg("grf")` (an implementation of generalised random
    forests).
-   *Regularized and Shrinkage Methods* : Regression models with some
    constraint on the parameter estimates can be fitted with the
    `r pkg("lasso2")` and `r pkg("lars")`
    packages. Lasso with simultaneous updates for groups of parameters
    (groupwise lasso) is available in package
    `r pkg("grplasso")`; the `r pkg("grpreg")`
    package implements a number of other group penalization models, such
    as group MCP and group SCAD. The L1 regularization path for
    generalized linear models and Cox models can be obtained from
    functions available in package `r pkg("glmpath")`, the
    entire lasso or elastic-net regularization path (also in
    `r pkg("elasticnet")`) for linear regression, logistic
    and multinomial regression models can be obtained from package
    `r pkg("glmnet")`. The `r pkg("penalized")`
    package provides an alternative implementation of lasso (L1) and
    ridge (L2) penalized regression models (both GLM and Cox models).
    Package `r pkg("RXshrink")` can be used to identify
    and display TRACEs for a specified shrinkage path and to determine
    the appropriate extent of shrinkage. Semiparametric additive hazards
    models under lasso penalties are offered by package
    `r pkg("ahaz")`. Fisher's LDA
    projection with an optional LASSO penalty to produce sparse
    solutions is implemented in package
    `r pkg("penalizedLDA")`. The shrunken centroids
    classifier and utilities for gene expression analyses are
    implemented in package `r pkg("pamr")`. An
    implementation of multivariate adaptive regression splines is
    available in package `r pkg("earth")`. Various forms of
    penalized discriminant analysis are implemented in packages
    `r pkg("hda")` and `r pkg("sda")`. Package
    `r pkg("LiblineaR")` offers an interface to the
    LIBLINEAR library. The `r pkg("ncvreg")` package fits
    linear and logistic regression models under the the SCAD and MCP
    regression penalties using a coordinate descent algorithm. The same
    penalties are also implemented in the `r pkg("picasso")`
    package. An implementation of bundle methods for regularized risk
    minimization is available form package `r pkg("bmrm")`.
    The Lasso under non-Gaussian and heteroscedastic errors is estimated
    by `r pkg("hdm")`, inference on low-dimensional
    components of Lasso regression and of estimated treatment effects in
    a high-dimensional setting are also contained. Package
    `r pkg("SIS")` implements sure independence screening in
    generalised linear and Cox models. Elastic nets for correlated
    outcomes are available from package `r pkg("joinet")`.
    Robust penalized generalized linear models and robust support vector
    machines are fitted by package `r pkg("mpath")` using
    composite optimization by conjugation operator. The
    `r pkg("islasso")` package provides an implementation of
    lasso based on the induced smoothing idea which allows to obtain
    reliable p-values for all model parameters. Best-subset selection
    for linear, logistic, Cox and other regression models, based on a
    fast polynomial time algorithm, is available from package
    `r pkg("abess", priority = "core")`.
-   *Boosting and Gradient Descent* : Various forms of gradient boosting
    are implemented in package
    `r pkg("gbm", priority = "core")` (tree-based functional
    gradient descent boosting). Package `r pkg("lightgbm")` and `r pkg("xgboost")`
    implement tree-based boosting using efficient trees as base
    learners for several and also user-defined objective functions. The
    Hinge-loss is optimized by the boosting implementation in package
    `r pkg("bst")`. An extensible boosting framework for
    generalized linear, additive and nonparametric models is available
    in package `r pkg("mboost", priority = "core")`.
    Likelihood-based boosting for mixed models is implemented in
    `r pkg("GMMBoost")`. GAMLSS models can be fitted using
    boosting by `r pkg("gamboostLSS")`. 
-   *Support Vector Machines and Kernel Methods* : The function `svm()`
    from `r pkg("e1071", priority = "core")` offers an
    interface to the LIBSVM library and package
    `r pkg("kernlab", priority = "core")` implements a
    flexible framework for kernel learning (including SVMs, RVMs and
    other kernel learning algorithms). An interface to the SVMlight
    implementation (only for one-against-all classification) is provided
    in package `r pkg("klaR")`. 
-   *Bayesian Methods* : Bayesian Additive Regression Trees (BART),
    where the final model is defined in terms of the sum over many weak
    learners (not unlike ensemble methods), are implemented in packages
    `r pkg("BayesTree")`, `r pkg("BART")`, and
    `r pkg("bartMachine")`. Bayesian nonstationary,
    semiparametric nonlinear regression and design by treed Gaussian
    processes including Bayesian CART and treed linear models are made
    available by package `r pkg("tgp")`. Bayesian structure
    learning in undirected graphical models for multivariate continuous,
    discrete, and mixed data is implemented in package
    `r pkg("BDgraph")`; corresponding methods relying on
    spike-and-slab priors are available from package
    `r pkg("ssgraph")`. Naive Bayes classifiers are
    available in `r pkg("naivebayes")`.
-   *Optimization using Genetic Algorithms* : Package
    `r pkg("rgenoud")` offers optimization routines based on
    genetic algorithms. The package `r pkg("Rmalschains")`
    implements memetic algorithms with local search chains, which are a
    special type of evolutionary algorithms, combining a steady state
    genetic algorithm with local search for real-valued parameter
    optimization.
-   *Association Rules* : Package `r pkg("arules")` provides
    both data structures for efficient handling of sparse binary data as
    well as interfaces to implementations of Apriori and Eclat for
    mining frequent itemsets, maximal frequent itemsets, closed frequent
    itemsets and association rules. Package
    `r pkg("opusminer")` provides an interface to the OPUS
    Miner algorithm (implemented in C++) for finding the key
    associations in transaction data efficiently, in the form of
    self-sufficient itemsets, using either leverage or lift.
-   *Fuzzy Rule-based Systems* : Package `r pkg("frbs")`
    implements a host of standard methods for learning fuzzy rule-based
    systems from data for regression and classification. Package
    `r pkg("RoughSets")` provides comprehensive
    implementations of the rough set theory (RST) and the fuzzy rough
    set theory (FRST) in a single package.
-   *Model selection and validation* : Package
    `r pkg("e1071")` has function `tune()` for hyper
    parameter tuning and function `errorest()`
    (`r pkg("ipred")`) can be used for error rate
    estimation. The cost parameter C for support vector machines can be
    chosen utilizing the functionality of package
    `r pkg("svmpath")`. Data splitting for crossvalidation
    and other resampling schemes is available in the
    `r pkg("splitTools")` package. Functions for ROC
    analysis and other visualisation techniques for comparing candidate
    classifiers are available from package `r pkg("ROCR")`.
    Packages `r pkg("hdi")` and `r pkg("stabs")`
    implement stability selection for a range of models,
    `r pkg("hdi")` also offers other inference procedures in
    high-dimensional models.
-   *Causal Machine Learning* : The package
    `r pkg("DoubleML")` is an object-oriented implementation
    of the double machine learning framework in a variety of causal
    models. Building upon the `r pkg("mlr3")` ecosystem,
    estimation of causal effects can be based on an extensive collection
    of machine learning methods.
-   *Other procedures* : Evidential classifiers quantify the uncertainty
    about the class of a test pattern using a Dempster-Shafer mass
    function in package `r pkg("evclass")`. The
    `r pkg("OneR")` (One Rule) package offers a
    classification algorithm with enhancements for sophisticated
    handling of missing values and numeric data together with extensive
    diagnostic functions.
-   *Meta packages* : Package `r pkg("tidymodels")` provides
    miscellaneous functions for building predictive models, including
    parameter tuning and variable importance measures. 
    In a similar spirit, package `r pkg("mlr3")` offers high-level interfaces to
    various statistical and machine learning packages. Package
    `r pkg("SuperLearner")` implements a similar toolbox.
    The `r pkg("h2o")` package implements a general purpose
    machine learning platform that has scalable implementations of many
    popular algorithms such as random forest, GBM, GLM (with elastic net
    regularization), and deep learning (feedforward multilayer
    networks), among others. An interface to the mlpack C++ library is
    available from package `r pkg("mlpack")`.
    `r pkg("CORElearn")` implements a rather broad class of
    machine learning algorithms, such as nearest neighbors, trees,
    random forests, and several feature selection methods. Similar,
    package `r pkg("rminer")` interfaces several learning
    algorithms implemented in other packages and computes several
    performance measures.
-   *Visualisation (initially contributed by Brandon Greenwell)* The
    `stats::termplot()` function package can be used to plot the terms
    in a model whose predict method supports `type="terms"`. The
    `r pkg("effects")` package provides graphical and
    tabular effect displays for models with a linear predictor (e.g.,
    linear and generalized linear models). Friedman's partial dependence
    plots (PDPs), that are low dimensional graphical renderings of the
    prediction function, are implemented in a few packages.
    `r pkg("gbm")`, `r pkg("randomForest")` and
    `r pkg("randomForestSRC")` provide their own functions
    for displaying PDPs, but are limited to the models fit with those
    packages (the function `partialPlot` from
    `r pkg("randomForest")` is more limited since it only
    allows for one predictor at a time). Packages
    `r pkg("pdp")`, `r pkg("plotmo")`, and
    `r pkg("ICEbox")` are more general and allow for the
    creation of PDPs for a wide variety of machine learning models
    (e.g., random forests, support vector machines, etc.); both
    `r pkg("pdp")` and `r pkg("plotmo")` support
    multivariate displays (`r pkg("plotmo")` is limited to
    two predictors while `r pkg("pdp")` uses trellis
    graphics to display PDPs involving three predictors). By default,
    `r pkg("plotmo")` fixes the background variables at
    their medians (or first level for factors) which is faster than
    constructing PDPs but incorporates less information.
    `r pkg("ICEbox")` focuses on constructing individual
    conditional expectation (ICE) curves, a refinement over Friedman's
    PDPs. ICE curves, as well as centered ICE curves can also be
    constructed with the `partial()` function from the
    `r pkg("pdp")` package.
-   *XAI* : Most packages and functions from the last section "Visualization"
    belong to the field of explainable artificial intelligence (XAI). 
    The meta packages `r pkg("DALEX")` and `r pkg("iml")` offer different
    methods to interpret any model, including partial dependence,
    accumulated local effects, and permutation importance. Accumulated local 
    effects plots are also directly available in `r pkg("ALEPlot")`.
    SHAP (from *SH*apley *A*dditive ex*P*lanations) is one of the most
    frequently used techniques to interpret ML models. 
    It decomposes - in a fair way - predictions into additive contributions 
    of the predictors. For tree-based models, the very fast TreeSHAP algorithm
    exists. It is shipped directly with `r pkg("h2o")`, `r pkg("xgboost")`,
    and `r pkg("lightgbm")`. Model-agnostic implementations of SHAP
    are available in additional packages: `r pkg("fastshap")` mainly uses
    Monte-Carlo sampling to approximate SHAP values, while `r pkg("shapr")` and 
    `r pkg("kernelshap")` provide implementations of KernelSHAP. 
    SHAP values of any of these packages can be plotted by the package `r pkg("shapviz")`. 
    A port to Python's "shap" package is provided in `r pkg("shapper")`. 
    Alternative decompositions of predictions are implemented in 
    `r pkg("lime")` and `r pkg("iBreakDown")`.

### Links
-   [MLOSS: Machine Learning Open Source Software](http://www.MLOSS.org/)
