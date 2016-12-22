
# coding: utf-8

# In[281]:

import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pickle


# In[282]:

dataset = pd.read_csv('C:\Users\Vignesh\Downloads\household_test.csv')
orig_dataset = dataset


# In[283]:

#Data pre-processing
#Converting the variable type to category
#Making the years a subtracted value from the current year
#Encoding alphabetical categories into numerical categories
le = preprocessing.LabelEncoder()
dataset.MSSubClass = pd.Series(le.fit_transform(dataset.MSSubClass),dtype = "category")
dataset.MSZoning = pd.Series(le.fit_transform(dataset.MSZoning),dtype = "category")
dataset.Street = pd.Series(le.fit_transform(dataset.Street),dtype = "category")
dataset.LotShape = pd.Series(le.fit_transform(dataset.LotShape),dtype = "category")
dataset.LandContour = pd.Series(le.fit_transform(dataset.LandContour),dtype = "category")
dataset.Utilities = pd.Series(le.fit_transform(dataset.Utilities),dtype = "category")
dataset.LotConfig = pd.Series(le.fit_transform(dataset.LotConfig),dtype = "category")
dataset.LandSlope = pd.Series(le.fit_transform(dataset.LandSlope),dtype = "category")
dataset.Neighborhood = pd.Series(le.fit_transform(dataset.Neighborhood),dtype = "category")
dataset.Condition1 = pd.Series(le.fit_transform(dataset.Condition1),dtype = "category")
dataset.Condition2 = pd.Series(le.fit_transform(dataset.Condition2),dtype = "category")
dataset.BldgType = pd.Series(le.fit_transform(dataset.BldgType),dtype = "category")
dataset.HouseStyle = pd.Series(le.fit_transform(dataset.HouseStyle),dtype = "category")
dataset.OverallQual = pd.Series(dataset.OverallQual,dtype = "category")
dataset.OverallCond = pd.Series(dataset.OverallCond,dtype = "category")
dataset.YearBuilt = datetime.now().year - dataset.YearBuilt
dataset.YearRemodAdd = datetime.now().year - dataset.YearRemodAdd
dataset.RoofStyle = pd.Series(le.fit_transform(dataset.RoofStyle),dtype = "category")
dataset.RoofMatl = pd.Series(le.fit_transform(dataset.RoofMatl),dtype = "category")
dataset.Exterior1st = pd.Series(le.fit_transform(dataset.Exterior1st),dtype = "category")
dataset.Exterior2nd = pd.Series(le.fit_transform(dataset.Exterior2nd),dtype = "category")
dataset.MasVnrType = pd.Series(le.fit_transform(dataset.MasVnrType),dtype = "category")
dataset.ExterQual = pd.Series(le.fit_transform(dataset.ExterQual),dtype = "category")
dataset.ExterCond = pd.Series(le.fit_transform(dataset.ExterCond),dtype = "category")
dataset.Foundation = pd.Series(le.fit_transform(dataset.Foundation),dtype = "category")
dataset.BsmtQual = pd.Series(le.fit_transform(dataset.BsmtQual),dtype = "category")
dataset.BsmtCond = pd.Series(le.fit_transform(dataset.BsmtCond),dtype = "category")
dataset.BsmtExposure = pd.Series(le.fit_transform(dataset.BsmtExposure),dtype = "category")
dataset.BsmtFinType1 = pd.Series(le.fit_transform(dataset.BsmtFinType1), dtype = "category")
dataset.BsmtFinType2 = pd.Series(le.fit_transform(dataset.BsmtFinType2), dtype = "category")
dataset.Heating = pd.Series(le.fit_transform(dataset.Heating), dtype = "category")
dataset.HeatingQC = pd.Series(le.fit_transform(dataset.HeatingQC), dtype = "category")
dataset.CentralAir = pd.Series(le.fit_transform(dataset.CentralAir), dtype = "category")
dataset.Electrical = pd.Series(le.fit_transform(dataset.Electrical), dtype = "category")
dataset.KitchenQual = pd.Series(le.fit_transform(dataset.KitchenQual), dtype = "category")
dataset.BsmtFullBath = dataset.BsmtFullBath.fillna(0)
dataset.BsmtHalfBath = dataset.BsmtHalfBath.fillna(0)
dataset.BsmtFullBath = pd.Series(dataset.BsmtFullBath,dtype = "category")
dataset.BsmtHalfBath = pd.Series(dataset.BsmtHalfBath,dtype = "category")
dataset.FullBath = pd.Series(dataset.FullBath,dtype = "category")
dataset.HalfBath = pd.Series(dataset.HalfBath,dtype = "category")
dataset.BedroomAbvGr = pd.Series(dataset.BedroomAbvGr,dtype = "category")
dataset.KitchenAbvGr = pd.Series(dataset.KitchenAbvGr,dtype = "category")
dataset.TotRmsAbvGrd = pd.Series(dataset.TotRmsAbvGrd,dtype = "category")
dataset.Functional = pd.Series(le.fit_transform(dataset.Functional), dtype = "category")
dataset.Fireplaces = pd.Series(dataset.Fireplaces,dtype = "category")
dataset.GarageCars = dataset.GarageCars.fillna(0)
dataset.GarageCars = pd.Series(dataset.GarageCars,dtype = "category")
dataset.FireplaceQu = pd.Series(le.fit_transform(dataset.FireplaceQu), dtype = "category")
dataset.GarageType = pd.Series(le.fit_transform(dataset.GarageType), dtype = "category")
dataset.GarageYrBlt = datetime.now().year - dataset.GarageYrBlt
dataset.GarageFinish = pd.Series(le.fit_transform(dataset.GarageFinish), dtype = "category")
dataset.GarageQual = pd.Series(le.fit_transform(dataset.GarageQual), dtype = "category")
dataset.GarageCond = pd.Series(le.fit_transform(dataset.GarageCond), dtype = "category")
dataset.PavedDrive = pd.Series(le.fit_transform(dataset.PavedDrive), dtype = "category")
dataset.PoolQC = pd.Series(le.fit_transform(dataset.PoolQC), dtype = "category")
dataset.Fence = pd.Series(le.fit_transform(dataset.Fence), dtype = "category")
dataset.MiscFeature = pd.Series(le.fit_transform(dataset.MiscFeature), dtype = "category")
dataset.MoSold = pd.Series(dataset.MoSold,dtype = "category")
dataset.YrSold = datetime.now().year - dataset.YrSold
dataset.SaleType = pd.Series(le.fit_transform(dataset.SaleType), dtype = "category")
dataset.SaleCondition = pd.Series(le.fit_transform(dataset.SaleCondition), dtype = "category")
dataset.Alley = pd.Series(le.fit_transform(dataset.Alley), dtype = "category")
dataset.LotFrontage = dataset.LotFrontage.fillna(0)
dataset.MasVnrArea = dataset.MasVnrArea.fillna(0)
dataset.GarageYrBlt = dataset.GarageYrBlt.fillna(dataset.mean()['GarageYrBlt'])
dataset.GarageArea = dataset.GarageArea.fillna(0)
dataset.TotalBsmtSF = dataset.TotalBsmtSF.fillna(0)
dataset.BsmtUnfSF = dataset.BsmtUnfSF.fillna(0)
dataset.BsmtFinSF2 = dataset.BsmtFinSF2.fillna(0)
dataset.BsmtFinSF1 = dataset.BsmtFinSF1.fillna(0)


# In[284]:

dataset = dataset.drop('Id',1)
X = dataset
#print X


# In[285]:

print dataset


# In[270]:

#Feature Selection - RFE....Looking to build model with 25 features
filename = 'C:\Users\Vignesh\Documents\\rfemodel.sav'
rfemodel = pickle.load(open(filename, 'rb'))
X_reduced = rfemodel.transform(X)
X_reduced_new = pd.DataFrame(X_reduced)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
#print X_reduced_new


# In[286]:

#X_train, X_test, Y_train, Y_test = train_test_split(X_reduced_new, Y, test_size = 0.4, random_state = 0)
filename = 'C:\Users\Vignesh\Documents\\gbrmodel_withoutreduction.sav'
clf_model = pickle.load(open(filename, 'rb'))


# In[287]:

#pred_Y_train = clf.predict(X_train)
pred_Y_test = clf_model.predict(X)


# In[288]:

Saleprice = pd.DataFrame({'Saleprice':pred_Y_test})
final_dataset = orig_dataset.join(Saleprice)


# In[290]:

final_dataset.to_csv(path_or_buf = 'C:\Users\Vignesh\Downloads\\house_value_gb_pred.csv')


# In[ ]:



