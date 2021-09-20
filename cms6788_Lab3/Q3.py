
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import sklearn.model_selection
from scipy.stats import skew
from scipy.stats.stats import pearsonr
def Q3_P3():
    # PREPROCESSING

    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    train.head()
    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price": train["SalePrice"], "log(price + 1)": np.log1p(train["SalePrice"])})
    # prices.hist()

    # log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)

    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    # creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    # Models
    from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoLarsCV
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error

    #UNDERFITTING MODEL
    model_lasso = Lasso(1000)
    model_lasso.fit(X_train, y)
    l1_pred = model_lasso.predict(X_test)
    l1_pred_train = model_lasso.predict(X_train)
    rms = np.sqrt(mean_squared_error(y, l1_pred_train))
    print("Lasso Training Score:", rms)
    solution = pd.DataFrame({"id": test.Id, "SalePrice": np.expm1(l1_pred)})
    solution.to_csv("Lasso_Regression_Underfit.csv", index=False)

    #OVERFITTING MODEL
    model_ridge = Ridge(.1)
    model_ridge.fit(X_train, y)
    l2_pred = model_ridge.predict(X_test)
    l2_pred_train = model_ridge.predict(X_train)

    X_train = all_data[:train.shape[0]]
    X_train.insert(0, "l2_out", l2_pred_train)
    X_test = all_data[train.shape[0]:]
    X_test.insert(0, "l2_out", l2_pred)
    y = train.SalePrice

    Ridge_w_outcome = Ridge(10)
    Ridge_w_outcome.fit(X_train, y)
    ridge_w_outcome_predictions = Ridge_w_outcome.predict(X_test)
    ridge_w_outcome_predictions_train = Ridge_w_outcome.predict(X_train)
    rms = np.sqrt(mean_squared_error(y, ridge_w_outcome_predictions_train))
    print("Ridge w Outcome Training Score:", rms)
    solution = pd.DataFrame({"id": test.Id, "SalePrice": np.expm1(ridge_w_outcome_predictions)})
    solution.to_csv("Lasso_Regression_Overfit.csv", index=False)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Q3_P3()