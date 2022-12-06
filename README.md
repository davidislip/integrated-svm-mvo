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

Please see the paper for details on the model formulations. 

# Running the Code
For all notebooks (except *2.2 ADM Comparison Notebook Timing*), please use Anaconda with the environment.yml file

### 1.0 GetFundamentals
Since, the Wharton Research Data Services (WRDS) data can't be shared publicly, a script to download the raw data from 
WRDS is provided. To use the script, one has to sign up for WRDS at their website https://wrds-www.wharton.upenn.edu/. 
Signing up typically requires that your institution/company has a partnership with them.

After signing up, the script 1.0 GetFundamentals.sas can be run in the WRDS SAS Studio interface to extract the raw 
fundamentals data. Save the output as WRDS_out.csv in your local cache folder and the next notebook will use it. 

### 1.1 WRDS Data Processing
This notebook does the following tasks:
* pulls the price data via the yfinance api
* cleans the price data and stores the cleaned data in 
* estimates the covariance and mean for each month and stores it in Forecasts.pkl
* cleans the WRDS data
* exports the cleaned WRDS data for the format used by the SVM MVO model 

### 2.1 Joint Optimization - 2 Factor Single Month
This notebook focuses on a single optimization problem. It solves the joint SVM-MVO problem using Gurobi, generates the efficient 
frontier, and shows how the optimal hyperplane can change depending on the risk aversion.

### 2.2 ADM Comparison Notebook Timing
This notebook (originally coded in Google Colab), tests the alternating method proposed in the paper against gurobi. 

### 3.0 Joint Optimization 1 Factor OOT
This notebook does the rebalancing experiments for the *Exact-Vol* model. All solves are done to optimality by Gurobi.  

### 3.1 Joint Optimization PADM
This notebook performs the rebalancing experiments for the *PADM-Fundamentals* model. All solutions are obtained via the 
proposed alternating method using vector penalty parameters.

### 4.0 Results 
This notebook produces the figures and relevant information for the timing and performance comparison between the PADM 
and the branch and bound solutions.

### 4.1 Financial Results
This notebook produces the figures and relevant information for the monthly rebalancing experiments. 

## Google Colab and Gurobi
Question: Why are some notebooks meant to be run in Colab and why are some meant to be run in standard gurobi? 

Answer: Originally, all the notebooks were going to be written in Colab. The idea of just being able to open a colab notebook, 
filling in your gurobi license details and then being good to go was extremely appealing. Furthermore, the tests take a long 
time to run and Gurobi Pro+ has the option to execute in the background ... hence freeing up resources on my local machine. 

At the start of writing the code, the gurobi Web Licensing Service (WLS) worked really well allowing for larger models to 
be run on the colab instances. However, a couple months in, the WLS stopped working... please see below for a Gurobi ticket for this issue

https://support.gurobi.com/hc/en-us/community/posts/4411084834833-License-issue-on-colab

But, at that time due to Covid, gurobi was not requiring that local academic licenses be activated while connected to a University network. 
The work around was to use bash commands in the notebook to try and activate the gurobi license. This worked up until this September when 
Gurobi required that the academic license be activated while connected to a University network. It was not until that latest version of Gurobi that 
they announced that the WLS should work again. 

https://support.gurobi.com/hc/en-us/community/posts/4411084834833-License-issue-on-colab

As such, testing is ongoing to make sure that we can switch out Gurobi 9 with the newest Gurobi for the timing experiments.

### References

Butler A, Kwon R (2021) Integrating prediction in mean-variance portfolio optimization. Available at SSRN
3788875 .

Paiva FD, Cardoso RTN, Hanaoka GP, Duarte WM (2019) Decision-making for financial trading: A fusion
approach of machine learning and portfolio selection. Expert Systems with Applications 115:635–655.

Fan A, Palaniswami M (2001) Stock selection using support vector machines. IJCNN’01. International Joint
Conference on Neural Networks. Proceedings (Cat. No. 01CH37222), volume 3, 1793–1798 (IEEE).

Wharton Research Data Services W (2021) Fundamentals quarterly compustat - capital iq from standard
poor’s. URL https://wrds-www.wharton.upenn.edu/.

Aroussi R (2021) yfinance. URL https://pypi.org/project/yfinance/.