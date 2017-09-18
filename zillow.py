from utils import *
import os
import gc

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error

from sklearn_pandas import DataFrameMapper

import xgboost as xgb

import keras
from keras import initializers
from keras.layers import Input, Embedding, Dense, Flatten, Dropout
from keras.models import Model
from keras.layers.merge import concatenate, multiply, add, average, maximum, dot
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise

import scipy

CATEGORY_DEFAULT = ''
CATEGORY_DTYPE = 'str'
CONTINUOUS_DEFAULT = 0
CONTINUOUS_DTYPE = 'float32'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class DataPreprocessor:

    def __init__(self, data):
        self.data = data
        
        self.trxns = None
        self.props = None
        self.train = None
        
        self.lm = linear_model.Ridge(alpha = .5)
        self.trend = None
        self.error = None
        self.preds = None
        
        self.categories = []

    def transactions(self):
        if self.trxns is None:
            self.trxns = from_pickle('zillow/trxns.pkl')

        if self.trxns is None:
            self.preprocess_transactions()
            to_pickle(self.trxns, 'zillow/trxns.pkl')

        return self.trxns

    def properties(self):
        if self.props is None:
            self.props = from_pickle('zillow/props.pkl')

        if self.props is None:
            self.clean_properties()
            to_pickle(self.props, 'zillow/props.pkl')

        return self.props

    def training(self, ycol='logerror', month=9, dropcols=set()):
        if self.train is None:
            self.train = from_pickle('zillow/train.pkl')
            # self.trend = from_pickle('zillow/trend.pkl')
            # self.error_month = from_pickle('zillow/error_month.pkl')
            # self.error_fips_month = from_pickle('zillow/error_fips_month.pkl')

        if self.train is None:
            trn = self.transactions()
            prp = self.properties()
            mrg = trn.merge(prp, how='left', on='parcelid')
            self.data.clear()
            del trn, prp; gc.collect()
            
            # self.trend = self.forecast_logerror(mrg)
            # to_pickle(self.trend, 'zillow/trend.pkl')
            # to_pickle(self.error_month, 'zillow/error_month.pkl')
            # to_pickle(self.error_fips_month, 'zillow/error_fips_month.pkl')
            
            # mrg = mrg.merge(self.trend, how='left', on=['fips', 'transaction_month'])

            self.train = mrg
            to_pickle(self.train, 'zillow/train.pkl')
            gc.collect()

        return self.train

    def prediction(self, month):
        if self.preds is None:
            self.train = None
            gc.collect()
            
            self.preds = pd.DataFrame()
            self.preds['parcelid'] = self.data.submission()['ParcelId']
            self.preds = self.preds.merge(self.properties(), how='left', on='parcelid')
            self.preds.insert(0, 'transaction_month', month)
            
            self.props = None
            gc.collect()
        else:
            self.preds['transaction_month'] = month
            # dropcols = [ c for c in self.preds.columns if c.startswith('logerror') ] + ['transaction_month']
            # self.preds.drop(dropcols, inplace=True)
        # self.preds.insert(0, 'transaction_month', month)
        # self.preds = preds.merge(self.error_trend(), how='left', on=['fips', 'transaction_month']).fillna(0)
        gc.collect()
        return self.preds

    def error_month(self):
        if self.error_month is None:
            self.tranining()
        return self.error_month

    def error_fips_month(self):
        if self.error_fips_month is None:
            self.tranining()
        return self.error_fips_month
    
    def error_trend(self):
        if self.trend is None:
            self.training()
        return self.trend

#------------------------------------------------------------------------------

    def split_census(self, df):
        c = 'rawcensustractandblock'
        df[c] = df[c].astype('str')
        df['fips'] = df[c].str.slice(0, 4).replace('nan', '')
        df['tract'] = df[c].str.slice(4, 11)
        df['block'] = df[c].str.slice(11)
        self.categories.append('fips')
        self.categories.append('tract')
        self.categories.append('block')
        return df

    # utility function for splitting the date into components
    def split_date(self, df):
        txd = 'transactiondate'
        df[txd] = pd.to_datetime(df[txd])
        # df[datecol+'_year'] = df[datecol].dt.year
        df['transaction_month'] = df[txd].dt.month
        df['transaction_month'] = df['transaction_month']
        # df[datecol+'_week'] = df[datecol].dt.week
        # df[datecol+'_day'] = df[datecol].dt.day
        # df[datecol+'_dayofweek'] = df[datecol].dt.dayofweek
        df.drop(txd, axis=1, inplace=True)
        return df

#------------------------------------------------------------------------------

    def forecast_logerror(self, df):
        month = self.preprocess_logerror(df, ['transaction_month'], '_month').reset_index()
        self.error_month = month
        # month = month.append(self.forecast_month(month))
        month = self.forecast_month(month)
        month = month.reset_index(drop=True)

        fips_month = self.preprocess_logerror(df, ['fips', 'transaction_month'], '_fips_month').reset_index()
        self.error_fips_month = fips_month
        # fips_month = fips_month.append(self.forecast_fips_month(fips_month))
        fips_month = self.forecast_fips_month(fips_month)
        fips_month = fips_month.reset_index(drop=True)

        return fips_month.merge(month, how='left', on='transaction_month')

    def preprocess_logerror(self, df, groupby, sfx):
        l = 'logerror'
        la = 'logerror_abs'
        lp = 'logerror_percent'
        grouped = df.loc[:, groupby+[l, la, lp]].groupby(groupby)
        ave = grouped.mean().rename(columns={l: l+sfx+'_ave', la: la+sfx+'_ave', lp: lp+sfx+'_ave'})
        # med = grouped.median().rename(columns={l: l+sfx+'_med', la: la+sfx+'_med'})
        std = grouped.std().rename(columns={l: l+sfx+'_std', la: la+sfx+'_std', lp: lp+sfx+'_std'})
        var = grouped.var().rename(columns={l: l+sfx+'_var', la: la+sfx+'_var', lp: lp+sfx+'_var'})
        combined = ave #.merge(med, left_index=True, right_index=True)
        combined = combined.merge(std, left_index=True, right_index=True)
        combined = combined.merge(var, left_index=True, right_index=True)
        return combined

    def forecast(self, df_train, df_preds, xcol, ycol):
        # print(df_train.loc[df_train[xcol]<11, [xcol]])
        # self.lm.fit(df_train.loc[:, [xcol]], df_train.loc[:, [ycol]])

        # only use months ealier than october
        # self.lm.fit(df_train.loc[df_train[xcol]<10, [xcol]], df_train.loc[df_train[xcol]<10, [ycol]])
        # df_preds[ycol] = self.lm.predict(df_preds.loc[:, [xcol]])

        l = 0.25
        x = df_train[df_train[xcol]<10][xcol]
        y = df_train[df_train[xcol]<10][ycol]
        # x = df_train[xcol]
        # y = df_train[ycol]
        coefs = np.polyfit(np.exp(-l*x), y, 1)
        # coefs = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y, p0=(coefs[0], l))
        # print(coefs)
        # df_preds[ycol] = pd.Series([ coefs[0][0]*np.exp(coefs[0][1]*x) for x in df_preds[xcol] ])
        df_preds[ycol] = pd.Series([ coefs[0]*np.exp(-l*x)+coefs[1] for x in df_preds[xcol] ])
        
        return df_preds

    def forecast_month(self, df_train):
        cols = list(df_train.columns)
        cols.remove('transaction_month')
        df_preds = pd.DataFrame([ m for m in range(1, 25) ], columns=['transaction_month'])
        for c in cols:
            df_preds = self.forecast(df_train, df_preds, 'transaction_month', c)
        return df_preds

    def forecast_fips_month(self, df_train):
        cols = list(df_train.columns)
        cols.remove('transaction_month')
        cols.remove('fips')
        df_preds = pd.DataFrame()
        for f in df_train['fips'].unique():
            df_fips = pd.DataFrame([ m for m in range(1, 25) ], columns=['transaction_month'])
            df_fips['fips'] = f
            for c in cols:
                df_fips = self.forecast(df_train[df_train['fips'] == f], df_fips, 'transaction_month', c)
            df_preds = df_preds.append(df_fips)
        return df_preds

#------------------------------------------------------------------------------

    def preprocess_transactions(self):
        df = self.data.transactions()

        c = 'logerror'
        df['logerror_percent'] = (np.exp(df[c])-1)*100
        df['logerror_abs'] = df[c].abs()
        df = self.split_date(df)
        
        # e_min = df[c].quantile(.01)
        # e_max = df[c].quantile(.99)
        # self.trxns = df[(df.logerror >= e_min) & (df.logerror <= e_max)].reset_index(drop=True)
        
        self.trxns = df

    # from Nikunj
    # Creating Additional Features
    def add_features(self, df_train):
        #error in calculation of the finished living area of home
        # c = 'N-LivingAreaError'
        # df_train[c] = (df_train['calculatedfinishedsquarefeet'].astype(float) / (df_train['finishedsquarefeet12'].astype(float) + 0.001))
        # df_train[c] = df_train[c].replace(np.inf, df_train[c].mean(), inplace=True)
        # df_train[c] = df_train[c].fillna(df_train[c].mean(), inplace=True)

        #proportion of living area
        df_train['N-LivingAreaProp'] = df_train['calculatedfinishedsquarefeet']/df_train['lotsizesquarefeet']
        # df_train['N-LivingAreaProp2'] = (df_train['finishedsquarefeet12']/df_train['finishedsquarefeet15'])
        # df_train['N-LivingAreaProp2'] = df_train['N-LivingAreaProp2'].fillna(df_train['N-LivingAreaProp2'].mean(), inplace=True)
        # df_train['N-LivingAreaProp2'] = df_train['N-LivingAreaProp2'].replace(np.inf, df_train['N-LivingAreaProp2'].mean(), inplace=True)

        #Amout of extra space
        df_train['N-ExtraSpace'] = df_train['lotsizesquarefeet'] - df_train['calculatedfinishedsquarefeet'] 
        df_train['N-ExtraSpace-2'] = df_train['finishedsquarefeet15'] - df_train['finishedsquarefeet12'] 

        #Total number of rooms
        df_train['N-TotalRooms'] = df_train['bathroomcnt'] + df_train['bedroomcnt']

        #Average room size
        # df_train['N-AvRoomSize'] = df_train['calculatedfinishedsquarefeet'] / df_train['roomcnt'] 
        # df_train['N-AvRoomSize'] = df_train['N-AvRoomSize'].fillna(df_train['N-AvRoomSize'].mean(), inplace=True)
        # df_train['N-AvRoomSize'] = df_train['N-AvRoomSize'].replace(np.inf, df_train['N-AvRoomSize'].mean(), inplace=True)

        # Number of Extra rooms
        df_train['N-ExtraRooms'] = df_train['roomcnt'] - df_train['N-TotalRooms'] 

        #Ratio of the built structure value to land area
        # df_train['N-ValueProp'] = df_train['structuretaxvaluedollarcnt']/df_train['landtaxvaluedollarcnt']

        #Does property have a garage, pool or hot tub and AC?
        df_train['N-GarPoolAC'] = ((df_train['garagecarcnt']>0) & (df_train['pooltypeid10'].astype(int)>0) &\
                                   (df_train['airconditioningtypeid'].astype(int)!=5))*1 

        df_train["N-location"] = df_train["latitude"] + df_train["longitude"]
        df_train["N-location-2"] = df_train["latitude"] * df_train["longitude"]
        df_train["N-location-2round"] = df_train["N-location-2"].round(-4)

        df_train["N-latitude-round"] = df_train["latitude"].round(-4)
        df_train["N-longitude-round"] = df_train["longitude"].round(-4)
        
        #Ratio of tax of property over parcel
        # df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt'] / df_train['taxamount']
        # df_train['N-ValueRatio'] = df_train['N-ValueRatio'].replace(np.inf, df_train['N-ValueRatio'].mean(), inplace=True)

        #TotalTaxScore
        df_train['N-TaxScore'] = df_train['taxvaluedollarcnt'] * df_train['taxamount']

        #polnomials of tax delinquency year
        df_train["N-taxdelinquencyyear-2"] = df_train["taxdelinquencyyear"] ** 2
        df_train["N-taxdelinquencyyear-3"] = df_train["taxdelinquencyyear"] ** 3

        #Length of time since unpaid taxes
        df_train['N-life'] = 2018 - df_train['taxdelinquencyyear']
        
        #Number of properties in the zip
        zip_count = df_train['regionidzip'].value_counts().to_dict()
        df_train['N-zip_count'] = df_train['regionidzip'].map(zip_count)

        #Number of properties in the city
        city_count = df_train['regionidcity'].value_counts().to_dict()
        df_train['N-city_count'] = df_train['regionidcity'].map(city_count)

        #Number of properties in the city
        region_count = df_train['regionidcounty'].value_counts().to_dict()
        df_train['N-county_count'] = df_train['regionidcounty'].map(region_count)

        #Indicator whether it has AC or not
        df_train['N-ACInd'] = (df_train['airconditioningtypeid'].astype(int)!=5)*1

        #Indicator whether it has Heating or not 
        df_train['N-HeatInd'] = (df_train['heatingorsystemtypeid'].astype(int)!=13)*1

        #There's 25 different property uses - let's compress them down to 4 categories
        c = 'N-PropType'
        df_train[c] = df_train.propertylandusetypeid.astype(float).replace(
            {31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 248 : "Mixed", 260 : "Home", 261 : "Home", 
             262 : "Home", 263 : "Home", 264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home", 269 : "Not Built", 
             270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 
             291 : "Not Built" })
        self.categories.append(c)
        
        #polnomials of the variable
        df_train["N-structuretaxvaluedollarcnt-2"] = df_train["structuretaxvaluedollarcnt"] ** 2
        # df_train["N-structuretaxvaluedollarcnt-3"] = df_train["structuretaxvaluedollarcnt"] ** 3

        #Average structuretaxvaluedollarcnt by city
        group = df_train.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
        df_train['N-Avg-structuretaxvaluedollarcnt'] = df_train['regionidcity'].map(group)

        #Deviation away from average
        df_train['N-Dev-structuretaxvaluedollarcnt'] = abs((df_train['structuretaxvaluedollarcnt']-\
                                                            df_train['N-Avg-structuretaxvaluedollarcnt']))/\
                                                            df_train['N-Avg-structuretaxvaluedollarcnt']
        return df_train
        
    def clean_properties(self):
        p = self.data.properties()
        cln = pd.DataFrame()
        
        c = 'parcelid'
        cln[c] = p[c]
        
        c = 'airconditioningtypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'architecturalstyletypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'basementsqft'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'bathroomcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('float32')
        c = 'bedroomcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('int')        
        c = 'buildingclasstypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'buildingqualitytypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'calculatedbathnbr'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'threequarterbathnbr'
        cln[c] = p[c].fillna(0).astype('int')
        c = 'fullbathcnt'
        cln[c] = p[c].fillna(0).astype('int')
        
        c = 'decktypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'finishedfloor1squarefeet'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'calculatedfinishedsquarefeet'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'finishedsquarefeet12'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'finishedsquarefeet13'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'finishedsquarefeet15'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'finishedsquarefeet50'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'finishedsquarefeet6'
        cln[c] = p[c].fillna(0).astype('float32')        
        
        # c = 'fips'
        # cln[c] = p[c].fillna(0).astype('int').astype('str')
        
        c = 'fireplacecnt'
        cln[c] = p[c].fillna(0).astype('int')        
        c = 'garagecarcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('int')        
        c = 'garagetotalsqft'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'hashottuborspa'
        cln[c] = p[c] == True
        c = 'heatingorsystemtypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'latitude'
        cln[c] = p[c].fillna(p[c].mean()).astype('float32')
        c = 'longitude'
        cln[c] = p[c].fillna(p[c].mean()).astype('float32')
        
        c = 'lotsizesquarefeet'
        cln[c] = p[c].fillna(p[c].median()).astype('float32')
        
        c = 'poolcnt'
        cln[c] = p[c].fillna(0).astype('int')
        c = 'poolsizesum'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'pooltypeid10'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        c = 'pooltypeid2'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        c = 'pooltypeid7'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'propertycountylandusecode'
        cln[c] = p[c].fillna(p[c].mode()[0])
        self.categories.append(c)
        c = 'propertylandusetypeid'
        cln[c] = p[c].fillna(p[c].mode()[0])
        self.categories.append(c)
        c = 'propertyzoningdesc'
        cln[c] = p[c].fillna(p[c].mode()[0])
        self.categories.append(c)

        # c = 'rawcensustractandblock'
        # cln[c] = p[c].astype('str').replace('nan', None)
        
        c = 'regionidcity'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        c = 'regionidcounty'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        c = 'regionidneighborhood'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        c = 'regionidzip'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        
        c = 'roomcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('int')
        c = 'storytypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        c = 'typeconstructiontypeid'
        cln[c] = p[c].fillna(0).astype('int')
        self.categories.append(c)
        c = 'unitcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('int')
        
        c = 'yardbuildingsqft17'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'yardbuildingsqft26'
        cln[c] = p[c].fillna(0).astype('float32')
        
        c = 'yearbuilt'
        cln[c] = p[c].fillna(2016).astype('int')
        
        c = 'numberofstories'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('int')
        c = 'fireplaceflag'
        cln[c] = p[c] == True
        
        c = 'structuretaxvaluedollarcnt'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'taxvaluedollarcnt'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'landtaxvaluedollarcnt'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'taxamount'
        cln[c] = p[c].fillna(0).astype('float32')
        
        c = 'assessmentyear'
        cln[c] = p[c].fillna(2017).astype('int')
        cln['assessmentyear_age'] = 2017 - cln[c]
        cln[c].astype('str')
        
        c = 'taxdelinquencyflag'
        cln[c] = p[c] == 'Y'
        
        c = 'taxdelinquencyyear'
        cln[c] = p[c]
        cln.loc[p[c] < 70, c] = cln[p[c] < 70][c] + 2000
        cln.loc[p[c] >= 70, c] = cln[p[c] >= 70][c] + 1900
        cln[c] = cln[c].fillna(2016).astype('int')
        cln['taxdelinquencyyear_age'] = 2017 - cln[c]
        self.categories.append(c)

        c = 'censustractandblock'
        cln[c] = p[c].fillna(0)
        self.categories.append(c)

        # FEATURE ENGINEERING        
        c = 'rawcensustractandblock'
        cln[c] = p[c]
        cln = self.split_census(cln)
        self.categories.append(c)
        
        c = 'yearbuilt'
        cln['yearbuilt_adjusted'] = cln[c] - 1801
        cln['yearbuilt_age'] = 2017 - cln[c]
        self.categories.append(c)
        
        c = 'longitude'
        m = cln[c].mean()
        cln['longitude_adjusted'] = (cln[c] - m).fillna(0)
        
        c = 'latitude'
        m = cln[c].mean()
        cln['latitude_adjusted'] = (cln[c] - m).fillna(0)        
        
        def distance(x, y):
            return np.sqrt(x**2 + y**2).sum()
        
        cln['distance'] = distance(cln['longitude_adjusted'], cln['latitude_adjusted'])
        
        loc_fips_mean = cln[['longitude', 'latitude', 'fips']].groupby(['fips']).mean().reset_index()
        loc_fips_mean.columns = ['fips', 'longitude_fips_mean', 'latitude_fips_mean']
        
        cln = cln.merge(loc_fips_mean, how='left', on='fips').fillna(0)
        cln['longitude_adjusted_fips'] = (cln['longitude'] - cln['longitude_fips_mean']).fillna(0)
        cln['latitude_adjusted_fips'] = (cln['latitude'] - cln['latitude_fips_mean']).fillna(0)
        cln['distance_fips'] = distance(cln['longitude_adjusted_fips'], cln['latitude_adjusted_fips'])
        cln = cln.drop(['longitude_fips_mean', 'latitude_fips_mean'], axis=1)
        
        self.props = self.add_features(cln)
        return cln       


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# For pulling in input data
class DataLoader:

    def __init__(self):
        self.preprocessed = DataPreprocessor(self)
        
        self.clear()

        self.file_train = 'train_2016_v2'
        self.file_props = 'properties_2016'
        self.file_subm = 'sample_submission'

    def load(self, f):
        csv = os.path.join('zillow', f + '.csv')
        return pd.read_csv(csv)

    def properties(self):
        if self.props is None:
            self.props = from_pickle('zillow/properties_2016.pkl')
            
        if self.props is None:
            p = self.load(self.file_props)
            for c in p.columns:
                if p[c].dtype == np.float64:
                    p[c] = p[c].astype(np.float32)
            to_pickle(p, 'zillow/properties_2016.pkl')
            self.props = p
            del p; gc.collect()
            
        return self.props

    def transactions(self):
        if self.train is None:
            self.train = self.load(self.file_train)
        return self.train

    def submission(self):
        if self.subm is None:
            self.subm = self.load(self.file_subm)
        return self.subm
    
    def clear(self):
        self.props = None
        self.train = None
        self.subm = None
        gc.collect()
        