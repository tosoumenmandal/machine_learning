
# coding: utf-8

# # Activity 2: Build a multiple linear regression model
# 
# You are now tasked with building a multiple linear regression model with two variables. Let's look at the box office sales example again. The two predictors are advertising spending and Facebook likes. 
# 
# Part 1: Build two simple linear regression models and interpret the output. 
# 
# Part 2: Make a multiple linear regression model and, again, interpret the coefficients.
# 
# ### Data exploration
# 
# Examine the data and make scatter plots between the predictors and the response. 

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Make the data
#y are box offce revenues, x1: advertising,x2: number of theaters  
#All variables are expressed in 000s
#Linear regression accepts Pandas DataFrame 
d = {"y":[23,12,36,27,45,70,55,8,62,28],
     "x1":[29,49,89,110,210,190,153,20,122,41],
     "x2": [2.536,2.919,1.707,2.005,2.232,2.910,2.795,1.8,18.88,1.838]
    }
data = pd.DataFrame(data = d)
data

plt.scatter(data[['x1']], data[['y']], color='orange')
plt.scatter(data[['x2']], data[['y']], color='green')
plt.show()


# ### Part 1: Simple linear regression
# 
# Build two simple linear regression models for advertising and number of theatres.

# ### Part 2: Multiple linear regression
# 
# Build one multiple linear regression model with two predictors: advertising and number of theatres.

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_data_initiate_model_single(data, x, y, size):
    train, test= train_test_split(data, test_size=size, random_state=42)
    train= sm.add_constant(train)
    test= sm.add_constant(test)
    
    Linreg= sm.OLS(train[[y]], train[['const', x]]).fit()
    y_pred= Linreg.predict(test[['const', x]])
    
    return Linreg, y_pred, train, test


linreg1, y_pred1, train1, test1= split_data_initiate_model_single(data, 'x1', 'y', 0.30)

print("\n***Predictor X1***")
print("MAE: "+ str(mean_absolute_error(test1[['y']], y_pred1)))
print("RMSE: "+ str(np.sqrt(mean_squared_error(test1[['y']], y_pred1))))
print("R-squared test: "+ str(r2_score(test1[['y']], y_pred1)))
print("R-squared train: "+ str(linreg1.rsquared))

#Plot Results
plt.scatter(test1[['x1']], test1[['y']], color='orange')
plt.plot(test1[['x1']], y_pred1, color='orange', linewidth=2)
plt.show()

linreg2, y_pred2, train2, test2= split_data_initiate_model_single(data, 'x2', 'y', 0.30)

print("\n***Predictor X2***")
print("MAE: "+ str(mean_absolute_error(test2[['y']], y_pred2)))
print("RMSE: "+ str(np.sqrt(mean_squared_error(test2[['y']], y_pred2))))
print("R-squared test: "+ str(r2_score(test2[['y']], y_pred2)))
print("R-squared train: "+ str(linreg2.rsquared))

#Plot Results
plt.scatter(test2[['x2']], test2[['y']], color='green')
plt.plot(test2[['x2']], y_pred2, color='green', linewidth=2)
plt.show()


# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_data_initiate_model_multiple(data, x, y, size):
    train, test= train_test_split(data, test_size=size, random_state=42)
    train= sm.add_constant(train)
    test= sm.add_constant(test)
    
    Linreg= sm.OLS(train[[y]], train[[x[0], x[1]]]).fit()
    y_pred= Linreg.predict(test[[x[0], x[1]]])
    
    return Linreg, y_pred, train, test


linreg, y_pred, train, test= split_data_initiate_model_multiple(data, ('x1','x2'), 'y', 0.30)

print("\n***Predictor X1***")
print("MAE: "+ str(mean_absolute_error(test[['y']], y_pred)))
print("RMSE: "+ str(np.sqrt(mean_squared_error(test[['y']], y_pred))))
print("R-squared test: "+ str(r2_score(test[['y']], y_pred)))
print("R-squared train: "+ str(linreg1.rsquared))


# In[ ]:




