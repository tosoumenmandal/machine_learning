
# coding: utf-8

# # How to select the best models
# 
# ## Predictions
# 
# Before going into detail about model selection, we will first make predictions with multiple linear regression. Remember that if we want to make predictions, the go-to package in Python is *scikit-learn*. Since we often use the *statsmodels* package, I will show you how to make predictions using the *statsmodels* and the *scikit-learn* packages. We will use the box office example with advertising, number of screens, and rating (Rated-R) as variables.

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Make the data
#y are box office revenues, x1: advertising, x2: number of theaters, dummy: rated R  
d = {"y":[23,12,36,27,45,70,55,8,62,28],
     "x1":[29,49,89,110,210,190,153,20,122,41],
     "x2": [2.036,2.919,1.707,1.505,2.232,2.910,2.795,1.46,3.288,1.838],
     "dummy": [1,1,0,1,1,0,0,1,0,0]
    }
data = pd.DataFrame(data = d)


# In[20]:


import statsmodels.api as sm

#Remember that the number of theaters was not signficant
#Build OLS model
data1 = sm.add_constant(data)
lm_stats = sm.OLS(data1["y"],data1[["const","x1", "x2", "dummy"]]).fit()
print(lm_stats.summary())


# In[21]:


#Now let's make a prediction to show that it does not matter whether a variable is signifcant or not
#const:1, x1: 200, x2: 2, dummy: 1 
lm_stats.predict([1,200,2,1])
 


# In[22]:


#This is exacty (small difference are due to rounding the coefficients to 4 digits)
1*12.4045 + 200*0.1880 + 2*6.1509 + 1*(-17.6097)


# In[23]:


#Now let's make a prediction with statsmodels
np.random.seed(40)   
        
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

#Make train/test spit
train, test = train_test_split(data1, test_size=0.30)

#Model and predict
lm_stats = sm.OLS(train["y"],train[["const","x1", "x2", "dummy"]]).fit()
pred_stats = lm_stats.predict(test[["const","x1", "x2", "dummy"]])

print("RMSE statsmodels: "+str(np.sqrt(mse(test[["y"]],pred_stats))))


# In[24]:


#Now let's make a prediction with sklearn
np.random.seed(40)   
        
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression  

#Make train/test spit
train, test = train_test_split(data1, test_size=0.30)

#Model and predict
lm_sk = LinearRegression().fit(train[["x1", "x2", "dummy"]], train["y"])
pred_sk = lm_sk.predict(test[["x1", "x2", "dummy"]])

print("RMSE sklearn: "+str(np.sqrt(mse(test[["y"]],pred_sk))))
#As expected the results are the same for sklearn as for 


# ## Model selection with approximation measures
# 
# We will now select the best model based on approximation measures: AIC, BIC, and adjusted R-squared. To select the best model, you must calculate all possible combinations of the predictors and see which one has the lowest AIC/BIC or the highest R-squared. Now, let's select the best two predictor models using AIC/BIC and adjusted R-squared.

# In[25]:


#The AIC, BIC, and adjusted R-squared are built-in for in the statsmodels packake, so we will aslo focus on this one
#We will only use the train set, since these measures will approximate the test set error

#Always begin with a null model: model with only an intercept
lm_null = sm.OLS(train["y"],train[["const"]]).fit()

#One predictor models
lm_x1 = sm.OLS(train["y"],train[["const","x1"]]).fit()
lm_x2 = sm.OLS(train["y"],train[["const","x2"]]).fit()                              
lm_dummy = sm.OLS(train["y"],train[["const","dummy"]]).fit() 

#Two predictor models
lm_x1x2 = sm.OLS(train["y"],train[["const","x1", "x2"]]).fit()
lm_x1dummy = sm.OLS(train["y"],train[["const","x1", "dummy"]]).fit()                               
lm_x2dummy = sm.OLS(train["y"],train[["const","x2", "dummy"]]).fit() 

#Extract all measures
results = pd.DataFrame(columns = ["Model","AIC", "BIC", "AdjR"])
results.Model = ["lm_null", "lm_x1", "lm_x2", "lm_dummy", "lm_x1x2","lm_x1xdummy", 
                            "lm_x2xdummy"]
results.AIC = [lm_null.aic, lm_x1.aic, lm_x2.aic, lm_dummy.aic, lm_x1x2.aic,lm_x1dummy.aic, lm_x2dummy.aic]
results.BIC = [lm_null.bic, lm_x1.bic, lm_x2.bic, lm_dummy.bic, lm_x1x2.bic,lm_x1dummy.bic, lm_x2dummy.bic]
results.AdjR = [lm_null.rsquared_adj, lm_x1.rsquared_adj, lm_x2.rsquared_adj,
        lm_dummy.rsquared_adj, lm_x1x2.rsquared_adj,lm_x1dummy.rsquared_adj, 
        lm_x2dummy.rsquared_adj]
results


# In[26]:


#Let's look at the minimimum value for AIC and BIC
results[["AIC", "BIC"]].apply(np.min)
#This is the 5th model, so the model including advertising (x1) and rated-R (dummy)


# In[27]:


#Check whether the results are the same for the adj R
results[["AdjR"]].apply(np.max)
#This is also the 5th model


# ## Model selection with cross-validation
# 
# We know that the approximation measures only approximate the test set error. Now we will calculate the true test set error and see whether we get the same results. They are different scales, so might impact the different models in another way. So, in order to make the correct decision upon the best mode, it is best practice to first normalise the data. 

# In[28]:


#Before we will model cross-validation: we will make a pipeline to calculate the rmse
#This will shorten the code and make it easier to model
#We will use scikit-learn to fit the model here. Since we want to test a null model, we will not fit an intercept
#Instead we will always add const as an intercept just as in statsmodels

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import Normalizer
from math import sqrt
    
    
def calculate_rmse(X, y, nFold):    
    #Make sure that your estimator is from sklearn and not from statsmodels!
    predictions = cross_val_predict(LinearRegression(fit_intercept=False), X, y, cv=nFold)
    
    rmse = sqrt(mse(y, predictions))
    
    return rmse


#Now calculate the RMSE of all the different models with 3 fold cross-validation
#Note that we do not add the const here
#Null model
rmse_null = calculate_rmse(data1[["const"]],data1["y"],nFold=3)

#One predictor models
rmse_x1 = calculate_rmse(data1[["const", "x1"]],data1["y"],nFold=3)
rmse_x2 = calculate_rmse(data1[["const", "x2"]],data1["y"],nFold=3)                              
rmse_dummy = calculate_rmse(data1[["const", "dummy"]],data1["y"],nFold=3)

#Two predictor models
rmse_x1x2 = calculate_rmse(data1[["const", "x1", "x2"]],data1["y"],nFold=3)
rmse_x1dummy = calculate_rmse(data1[["const", "x1", "dummy"]],data1["y"],nFold=3)                               
rmse_x2dummy = calculate_rmse(data1[["const", "x2","dummy"]],data1["y"],nFold=3)

#Look at the best model
rmse_null, rmse_x1, rmse_x2, rmse_dummy, rmse_x1x2, rmse_x1dummy, rmse_x2dummy, rmse_x2dummy
#Again you can see


# In this case, cross-validation and approximation measures give the same result. However, the results could have differed. In general, cross-validation is the best approach since you estimate the test set error directly. As you also might notice, this technique might become unmanageable when you have a lot of predictors. There are other model selection techniques (e.g. ridge, lasso, or elastic net regression) that are much more efficient. 
# 
# Also, if you have some proprietary knowledge (e.g. from talking with managers and experts in the field or by doing a literature review) about what variables to include in your model, you can narrow down the number of models to compare. However, one big downside of cross-validation is that when confronted with big dataset, it can take some time to get the performance measures. So in that case, approximation measures might be a good alternative. 

# In[ ]:




