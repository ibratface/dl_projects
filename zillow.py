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
            self.preds = pd.DataFrame()
            self.preds['parcelid'] = self.data.submission()['ParcelId']
            self.preds = self.preds.merge(self.properties(), how='left', on='parcelid')
            self.preds.insert(0, 'transaction_month', month)
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
        df['fips'] = df[c].str.slice(0, 4).replace('nan', 0).astype(CATEGORY_DTYPE)
        df['tract'] = df[c].str.slice(4, 11).replace('', 0).astype(CATEGORY_DTYPE)
        df['block'] = df[c].str.slice(11).replace('', 0).astype(CATEGORY_DTYPE)
        df[c] = df[c].astype(CATEGORY_DTYPE)
        # df.loc[:, ('fips', 'tract', 'block')].head().transpose()
        return df

    # utility function for splitting the date into components
    def split_date(self, df):
        txd = 'transactiondate'
        df[txd] = pd.to_datetime(df[txd])
        # df[datecol+'_year'] = df[datecol].dt.year
        df['transaction_month'] = df[txd].dt.month
        df['transaction_month'] = df['transaction_month'].astype('str')
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

    def clean_properties(self):
        p = self.data.properties()
        cln = pd.DataFrame()
        
        c = 'parcelid'
        cln[c] = p[c]
        
        c = 'airconditioningtypeid'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'architecturalstyletypeid'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'basementsqft'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'bathroomcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('float32')
        c = 'bedroomcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('int')        
        c = 'buildingclasstypeid'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'buildingqualitytypeid'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        
        c = 'calculatedbathnbr'
        cln[c] = p[c].fillna(0).astype('float32')
        c = 'threequarterbathnbr'
        cln[c] = p[c].fillna(0).astype('int')
        c = 'fullbathcnt'
        cln[c] = p[c].fillna(0).astype('int')
        
        c = 'decktypeid'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        
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
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        
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
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'pooltypeid2'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'pooltypeid7'
        
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'propertycountylandusecode'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('str')
        c = 'propertylandusetypeid'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('str')
        c = 'propertyzoningdesc'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('str')

        # c = 'rawcensustractandblock'
        # cln[c] = p[c].astype('str').replace('nan', None)
        
        c = 'regionidcity'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'regionidcounty'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'regionidneighborhood'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'regionidzip'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        
        c = 'roomcnt'
        cln[c] = p[c].fillna(p[c].mode()[0]).astype('int')
        c = 'storytypeid'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
        c = 'typeconstructiontypeid'
        cln[c] = p[c].fillna(0).astype('int').astype('str')
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
        cln[c].astype('str')

        c = 'censustractandblock'
        cln[c] = p[c].astype('str').replace('nan', '0')

        # FEATURE ENGINEERING        
        c = 'rawcensustractandblock'
        cln[c] = p[c]
        cln = self.split_census(cln)
        
        c = 'yearbuilt'
        cln['yearbuilt_adjusted'] = cln[c] - 1801
        cln['yearbuilt_age'] = 2017 - cln[c]
        cln[c] = cln[c].astype('str')
        
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
        
        self.props = cln
        return cln       


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# For pulling in input data
class DataLoader:

    def __init__(self):
        self.preprocessed = DataPreprocessor(self)
        
        self.clear()

        self.file_train = 'train_2016_v2'
        self.file_train_pre = 'train.pre'
        self.file_props = 'properties_2016'
        self.file_props_pre = 'props.pre'
        self.file_subm = 'sample_submission'

    def load(self, f):
        csv = os.path.join('zillow', f + '.csv')
        return pd.read_csv(csv)

    def properties(self):
        if self.props is None:
            self.props = self.load(self.file_props)
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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class DataTransformer:
    
    def __init__(self):
        self.train = None
        self.mappers = None

    def fit(self, df, dropcols=set()):
        if self.mappers == None:
            self.mappers = from_pickle('zillow/mappers.pkl')
            if self.mappers:
                self.cat_map_fit = self.mappers[0][0]
                self.con_map_fit = self.mappers[1][0]
                self.add_map_fit = self.mappers[2][0]

        if self.mappers == None:
            rem_cols = set(df.columns) - set(['logerror', 'logerror_abs', 'parcelid']) - dropcols
            add_cols = set([ c for c in rem_cols if c.startswith('logerror') ] + ['transaction_month'])
            rem_cols = rem_cols - add_cols
            cat_cols = set([ c for c in rem_cols if df[c].dtype.name == 'object' ])
            con_cols = rem_cols - cat_cols

            add_maps = [([c], MinMaxScaler()) for c in add_cols]
            cat_maps = [(c, LabelEncoder()) for c in cat_cols]
            con_maps = [([c], MinMaxScaler()) for c in con_cols]

            add_mapper = DataFrameMapper(add_maps)
            self.add_map_fit = add_mapper.fit(self.data.error_trend())

            cat_mapper = DataFrameMapper(cat_maps)
            self.cat_map_fit = cat_mapper.fit(self.data.properties())

            con_mapper = DataFrameMapper(con_maps)
            self.con_map_fit = con_mapper.fit(self.data.training())

            self.mappers = [(self.cat_map_fit, 'int64'), (self.con_map_fit, 'float32'), (self.add_map_fit, 'float32')]
            to_pickle(self.mappers, 'zillow/mappers.pkl')

            gc.collect()

        return self.mappers

    def transform(self, df):
        mapped = []
        for m, dtype in self.mappers:
            mapped.append(m.transform(df).astype(dtype))
        return np.concatenate(mapped, axis=1)    
    
    def fit_transform(self, data, dropcols=set()):
        mappers = self.fit(data.properties(), dropcols)
        x_train, x_valid, x_test, y_train, y_valid, y_test = data.training()
        x_train = self.transform(x_train, mappers)
        x_valid = self.transform(x_valid, mappers)
        x_test = self.transform(x_test, mappers)
        gc.collect()
        return x_train, x_valid, x_test, y_train, y_valid, y_test    

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class XGB:
    
    def __init__(self, data=DataLoader()):
        self.data = data
        self.clf = None
    
    def remove_outliers(self, X, y, ycol):
        # return X, y
        t = self.data.preprocessed.transactions()
        l_min = t[ycol].quantile(.05)
        l_max = t[ycol].quantile(.95)
        f = (y >= l_min) & (y <= l_max)
        return X[f], y[f]

    def adapt(self, df, dropcols=set()):
        mustdropcols = set([
                    'propertyzoningdesc', 
                    'propertycountylandusecode', 
                    'censustractandblock',
                    'rawcensustractandblock',
                    'parcelid',
                   ])

        mustdropcols = mustdropcols | set(dropcols)
        df = df.drop(mustdropcols, axis=1)

        for c in df.columns:
            if df[c].dtype.name == 'object':
                df[c] = df[c].replace('',0).astype(np.float32)

        return df

    def train(self, ycol='logerror', params=None, dropcols=set(), verbose_eval=False):
        if params is None:
            params = {
                'eta': 0.005,
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'max_depth': 3,
                'silent': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.8,
                'min_child_weight': 4,
                'lambda': 2,
            }        
        
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.data.preprocessed.training(ycol=ycol)
        
        # combine train and valid
        x_train = pd.concat([x_train, x_valid])
        y_train = pd.concat([y_train, y_valid])
        x_valid = x_test
        y_valid = y_test
        
        x_train, y_train = self.remove_outliers(x_train, y_train, ycol)
        # x_valid, y_valid = self.remove_outliers(x_valid, y_valid, ycol)

        d_train = xgb.DMatrix(self.adapt(x_train, dropcols), label=y_train, silent=True)
        d_valid = xgb.DMatrix(self.adapt(x_valid, dropcols), label=y_valid, silent=True)
        # d_test = xgb.DMatrix(self.adapt(x_test), label=y_test, silent=True)

        del x_train, x_valid, x_test; gc.collect()

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        evals_result = {}
        self.clf = xgb.train(params, d_train, 10000, watchlist, evals_result=evals_result, 
                             early_stopping_rounds=200, verbose_eval=verbose_eval)
        del d_train, d_valid; gc.collect()

        print (self.clf.attributes()['best_msg'])
        return self.clf, evals_result


    def predict(self, dropcols=set()):
        # build the test set
        subm = pd.DataFrame()
        months = [10, 11, 12, 22, 23, 24]
        dates = ['201610', '201611', '201612', '201710', '201711', '201712']
    #     months = [10]
    #     dates = ['201610']
        for month, date in zip(months, dates):
            print('Predicting...', date)

            merged = self.data.preprocessed.prediction(month)
            subm['ParcelId'] = merged['parcelid']
            merged = self.adapt(merged, dropcols)

            dm_test = xgb.DMatrix(merged)
            del merged; gc.collect()
            
            subm[date] = self.clf.predict(dm_test)
            del dm_test; gc.collect()

        subm.to_csv('zillow/submission.xgb.csv.gz', index=False, float_format='%.4f', compression='gzip')
        return subm


    def importance(self):
        imp = self.clf.get_fscore()
        imp = sorted(imp.items(), key=lambda x: x[1])

        df = pd.DataFrame(imp, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance');
        return df


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class NeuralNet:

    def __init__(self, data, dropcols=set()):
        self.data = data
        self.dropcols = dropcols

        self.x_train = None
        self.x_valid = None
        self.x_test = None

        self.y_train = None
        self.y_valid = None
        self.y_test = None

        self.model = None
        self.mappers = None

    def feature_split(self, df):
        return np.split(df, df.shape[1], axis=1)

    def get_model(self):
        if self.model == None:
            embsz = 32

            def categorical_input(fname, fclasses):
                vocsz = len(fclasses)
            #     print(vocsz)
                inp = Input((1,), dtype='int64', name=fname+'_inp')
                emb_init = keras.initializers.RandomUniform(minval=-0.06/embsz, maxval=0.06/embsz)
                out = Embedding(vocsz, embsz, input_length=1, embeddings_initializer=emb_init)(inp)
                out = Flatten(name=fname+'_flt')(out)
                # if fname == 'parcelid':
                #    out = Dropout(0.9)(out)
                out = Dense(1, name=fname+'_den', activation='relu', use_bias=False, kernel_initializer='ones')(out)
                return inp, out

            def continuous_input(fname):
                inp = Input((1,), dtype='float32', name=fname+'_inp')
                out = Dense(1, name=fname+'_den', activation='relu', use_bias=False, kernel_initializer='ones')(inp)
                return inp, out

            def dense_stack(inp, layers, units, dropout):
                den = Dropout(dropout)(inp)
                for i in range(layers):
                    den = Dense(units, activation='relu', kernel_initializer='random_uniform')(den)
                return den
            
            def dense_weave(inp, threads, layers, units, dropout):
                return [ dense_stack(inp, layers, units, dropout) for t in range(threads) ]
            
            self.fit()
            cat_in = [ categorical_input(f[0], f[1].classes_) for f in self.cat_map_fit.features ]
            con_in = [ continuous_input(f[0][0]) for f in self.con_map_fit.features ] +\
            [ continuous_input(f[0][0]) for f in self.add_map_fit.features ]

            # err_ave_in = Input((1,), dtype='float32', name='err_ave_inp')
            # err_dev_in = Input((1,), dtype='float32', name='err_dev_inp')

            den = concatenate([ o for _, o in cat_in ] + [ o for _, o in con_in ])
            # den = Dropout(0.02)(den)
            # den = Dense(1024, activation='relu', kernel_initializer='random_uniform')(den)
            # den = Dense(1024, activation='relu', kernel_initializer='random_uniform')(den)
            den = concatenate(dense_stack(den, 16, 2, 1) + 
                              dense_stack(den, 8, 2, 2) + 
                              dense_stack(den, 4, 2, 4) + 
                              dense_stack(den, 2, 2, 8))
            den = Dense(1, activation='linear')(den)
            # den = Dense(1, activation='linear')(den)
            # out = multiply([den, err_dev_in])
            # out = add([out, err_ave_in])

            model = Model(inputs=[ i for i, _ in cat_in ] + [ i for i, _ in con_in ], outputs=[den])
            opt = keras.optimizers.Adam(lr=0.001, decay=0.0)
            model.summary()
            model.compile(loss='mean_absolute_error', optimizer=opt)

            self.model = model
            gc.collect()

        return self.model

    def train(self, epochs=5, callbacks=None, verbose=1):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.training()
        x = np.concatenate([x_valid, x_test])
        y = np.concatenate([y_valid, y_test])
        hist = self.get_model().fit(self.feature_split(x_train), y_train, batch_size=256, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=(self.feature_split(x), y), shuffle=True)
        preds = self.get_model().predict(self.feature_split(x_test))
        print (preds)
        mae = mean_absolute_error(y_test, preds)
        return mae

    def test(self):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.training()
        x = np.concatenate([x_valid, x_test])
        y = np.concatenate([y_valid, y_test])
        preds = self.get_model().predict(self.feature_split(x))
        mae = mean_absolute_error(y, preds)
        return mae

    def predict(self, verbose=1):
        # build the test set
        subm = pd.DataFrame()
        subm['ParcelId'] = self.data.properties()['parcelid']
        months = [10, 11, 12, 22, 23, 24]
        dates = ['201610', '201611', '201612', '201710', '201711', '201712']
    #     months = [10]
    #     dates = ['201610']
        for month, date in zip(months, dates):
            print('\nPredicting...', date)

            x = self.data.prediction(month)
            # print (x.columns)
            # print (x.isnull().any())
            # e = x[['logerror_month_ave', 'logerror_month_std']]
            # e['logerror_month_std'] = e['logerror_month_std'].apply(np.sqrt) * 3
            x = self.transform(x, self.mappers)
            # x = np.concatenate([x, e], axis=1)
            x = self.feature_split(x)
            
            subm[date] = self.model.predict(x, verbose=verbose)
            subm.to_csv('zillow/{}.csv.gz'.format(date), index=False, float_format='%.4f', compression='gzip')
            del x; gc.collect()

        subm.to_csv('zillow/submission_nn.csv.gz', index=False, float_format='%.4f', compression='gzip')
        return subm
