# Integrating Support Vector Machines (SVM) with Mean Variance Optimization (MVO)

A very popular approach to asset allocation is to use a prediction model to estimate some input into a portfolio 
optimization routine. For example, Butler and Kwon (2021) use a prediction model for the expected return, which is then 
used as input into an MVO model. Another approach uses prediction models to predict which assets will have a positive 
return in the next period and then uses MVO to allocate among those assets. Paiva et al. (2019), 
and Fan and Palaniswami (2001) both make use of this approach and use support vector machines to predict which assets
will exhibit positive returns. A major downfall of the aforementioned `predict then optimize' approaches is that the 
prediction model does not consider the resulting quality of the decisions. 

Interestingly, the use of a linear hyperplane to screen the assets for investment reflects the behaviours of investors 
since they often use thresholds and different criteria to remove assets from consideration. In addition to screening, 
investors generally prefer lower cardinality limits since a universe of fewer assets requires less administrative work 
and results in lower transaction costs. Another approach to mitigating transaction costs is to include regularization in
the optimization objective function. Since MVO can be viewed as a linear regression model, the effect of regularization
is to reduce the linear hypothesis class's ability to overfit the data. 

Introducing cardinality limits makes the problem of selecting  a prediction model to consider decision quality much more 
challenging (these challenges are described in detail in the paper). 

Given the considerations and challenges described above, we propose a portfolio optimization model that:
* i) is integrated with an SVM model in a risk-based manner, 
* ii) incorporates economic rationales such as cardinality
constraints and asset screening behaviours, and
* iii) has clear links to regularization and
robustness of SVM and MVO models.

## Proposed Models 

### SVM MVO

### SVM MVO 2

### PADM

# Repository Structure

# Running the Code

### 1.0 Wharton Research Data Services

### 1.1 WRDS Data Processing

### 2.1 Joint Optimization

### 2.2 ADM Comparison Notebook Timing

### 3.0 Joint Optimization 1 Factor OOT

### 3.1 Joint Optimization PADM

### 4.0 Results 

### 4.1 Financial Results

## Google Colab and Gurobi

### References