import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

# 1. Refer to this Kaggle competition:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/
#
# 2. Follow the data preprocessing steps from:
# https://www.kaggle.com/code/apapiu/regularized-linear-models
# Then run a ridge regression using alpha = 0.1. Make a submission of this prediction,
# what is the RSME you got?
# (hint: remember to exponentiate 'np.expm1(ypred) your prdictions)
#
# 3. Compare a ridge regression and lasso regression model. Optimize tha alphas using
# cross validation. What is the best score you can get from a single ridge regression
# model and from a single lasso model?
#
# 4. Plot the l0 norm (number of nonzeros) of the coefficients that lasso produces as
# you vary the strength of regularization parameter alpha.
#
# 5. Add the outputs of your models as features and train a ridge regression on all the 
# features plus the model outputs (this is called Ensembling and Stacking).
#
# 6. Install XGBoost (Gradient Boosting) and train a gradient boosting regression. What
# score can you get just from a single XGB? (you will need to optimize over its parameters).
#
# 7. Try to get a more accurate model. Try feature engineering and stacking many models.
# No non-Python tools are allowed.


train = pd.read_csv('lab2/house-prices-advanced-regression-techniques/train.csv')    # Read in training dataset
test = pd.read_csv('lab2/house-prices-advanced-regression-techniques/test.csv')      # Read in test dataset

train.head()    # Print all rows of the training dataset

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))



# PART 2 (Data Preprocessing)

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# Transform the skewed numeric features of training set by taking log([feature]+1) to make the features more normal
#   and create dummy variables for the categorical features (price and log(price+1)):
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
# prices.hist()

# Log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

# Log transform the skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # Compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

# Fill NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

# Create matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# Using scikit-learn, use regularized linear regression modules (lasso and ridge regularization). First,
#   define a function that returns cross-validation RSME so we can evaluate our models and pick the best tuning par:
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# Run a ridge regression using alpha = 0.1:
model_ridge = Ridge()
cv_ridgeOne = rmse_cv(Ridge(0.1)).mean()
print(cv_ridgeOne)
# RSME for alpha = 0.1 is 0.1377499



# PART 3 (RIDGE AND LASSO REGRESSION MODELS)

# Main tuning parameter for the ridge model is alpha - a regularization parameter that measures how flexible our
#   our model is. Higher the regularization, less prone the model will be to overfit. But, it will also lose
#   flexibility and might not capture all of the signal of the data:
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
print(cv_ridge.min())   # Minimum RSME (best score) is 0.1273123

# Next, try lasso regression model:
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print(rmse_cv(model_lasso).min())   # Minimum RSME is 0.1033097



# PART 4

non_zero_coefs = []
alphaLasso = [1, 0.1, 0.001, 0.0005]
for alpha in alphaLasso:
    mdlLasso = Lasso(alpha=alpha)
    mdlLasso.fit(X_train, y)
    coef = pd.Series(mdlLasso.coef_, index = X_train.columns)
    num_non_zeros = np.count_nonzero(coef)
    non_zero_coefs.append(num_non_zeros)
plt.semilogx(alphaLasso, non_zero_coefs, marker='o')
plt.xlabel('alpha')
plt.ylabel('L0')
plt.show()



# PART 5

lassoPredictions = np.empty((X_train.shape[0], len(alphas)))
ridgePredictions = np.empty((X_train.shape[0], len(alphas)))
for i, alpha in enumerate(alphas):
    lassoPredictions[:, i] = mdlLasso.fit(X_train,y).predict(X_train)
    ridge_model = Ridge(alpha=alpha)
    ridgePredictions[:, i] = ridge_model.fit(X_train, y).predict(X_train)

X_stacked = np.column_stack((X_train, lassoPredictions))
X_stacked = np.column_stack((X_stacked, ridgePredictions))

def rmse_cvStack(model):
    rmse= np.sqrt(-cross_val_score(model, X_stacked, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

ridge_stacked = Ridge(alpha = 0.5)
score_ridgeStacked = rmse_cvStack(ridge_stacked).min()
print(score_ridgeStacked)



# PART 6

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

preds = 0.7*lasso_preds + 0.3*xgb_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("lab2/ridge_sol.csv", index = False)



# PART 7

selector = SelectFromModel(model_xgb, threshold=0.1)
selector.fit(X_train, y)
X_train_selected = selector.transform(X_train)

model_xgb.fit(X_train, y)

xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

preds = 0.7*lasso_preds + 0.3*xgb_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("lab2/ridge_sol7.csv", index = False)