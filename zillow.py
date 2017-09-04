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
            gc.collect()

        return self.trxns

    def properties(self):
        if self.props is None:
            self.props = from_pickle('zillow/props.pkl')

        if self.props is None:
            self.preprocess_properties()
            to_pickle(self.props, 'zillow/props.pkl')
            gc.collect()
        return self.props

    def training(self, month=9, dropcols=set()):
        if self.train is None:
            self.train = from_pickle('zillow/train.pkl')
            self.trend = from_pickle('zillow/trend.pkl')
            self.error_month = from_pickle('zillow/error_month.pkl')
            self.error_fips_month = from_pickle('zillow/error_fips_month.pkl')

        if self.train is None:
            trn = self.transactions()
            prp = self.properties()
            mrg = trn.merge(prp, how='left', on='parcelid')

            self.trend = self.forecast_logerror(mrg)
            to_pickle(self.trend, 'zillow/trend.pkl')
            to_pickle(self.error_month, 'zillow/error_month.pkl')
            to_pickle(self.error_fips_month, 'zillow/error_fips_month.pkl')
            
            mrg = mrg.merge(self.trend, how='left', on=['fips', 'transaction_month'])

            self.train = mrg
            to_pickle(self.train, 'zillow/train.pkl')
            gc.collect()

        x = self.train.drop(['logerror', 'logerror_abs'], axis=1).drop(dropcols, axis=1)
        y = self.train['logerror']
        c = x.transaction_month
        return x[c < month], x[c == month], x[c > month], y[c < month], y[c == month], y[c > month]        

    def prediction(self, month):
        if self.preds is None:
            self.preds = pd.DataFrame()
            self.preds['parcelid'] = self.data.submission()['ParcelId']
            self.preds = self.preds.merge(self.properties(), how='left', on='parcelid')
        preds = self.preds.copy()
        preds.insert(0, 'transaction_month', month)
        preds = preds.merge(self.error_trend(), how='left', on=['fips', 'transaction_month']).fillna(0)
        gc.collect()
        return preds

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
        df['fips'] = df[c].str.slice(0, 4).replace('nan', CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)
        df['tract'] = df[c].str.slice(4, 11).replace('', CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)
        df['block'] = df[c].str.slice(11).replace('', CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)
        df['tractandblock'] = df['tract'] + df['block']
        df[c] = df[c].astype(CATEGORY_DTYPE)
        df.loc[:, ('fips', 'tract', 'block')].head().transpose()
        return df

    # utility function for splitting the date into components
    def split_date(self, df):
        txd = 'transactiondate'
        df[txd] = pd.to_datetime(df[txd])
    #     df[datecol+'_year'] = df[datecol].dt.year
        df['transaction_month'] = df[txd].dt.month
    #     df[datecol+'_week'] = df[datecol].dt.week
    #     df[datecol+'_day'] = df[datecol].dt.day
    #     df[datecol+'_dayofweek'] = df[datecol].dt.dayofweek
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
        grouped = df.loc[:, groupby+[l, la]].groupby(groupby)
        ave = grouped.mean().rename(columns={l: l+sfx+'_ave', la: la+sfx+'_ave'})
        # med = grouped.median().rename(columns={l: l+sfx+'_med', la: la+sfx+'_med'})
        std = grouped.std().rename(columns={l: l+sfx+'_std', la: la+sfx+'_std'})
        var = grouped.var().rename(columns={l: l+sfx+'_var', la: la+sfx+'_var'})
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

        l = 0.5
        x = df_train.loc[df_train[xcol]<10][xcol]
        y = df_train[df_train[xcol]<10][ycol]
        coefs = np.polyfit(np.exp(-l*x), y, 1)
        coefs = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y, p0=(coefs[0], l))
        # print(coefs)
        df_preds[ycol] = pd.Series([ coefs[0][0]*np.exp(coefs[0][1]*x) for x in df_preds[xcol] ])
        
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

        df['logerror_abs'] = df['logerror'].abs()
        df = self.split_date(df)

        self.trxns = df

    def preprocess_properties(self):
        df = pd.DataFrame()
        props = self.data.properties()

        c = 'rawcensustractandblock'
        df[c] = props[c]
        df = self.split_census(df)

        # c = 'bathnbedcnt'
        # df[c] = props['bathroomcnt'] + props['bedroomcnt']
        # df[c] = df[c].fillna(df[c].mean())

        c = 'yearbuilt'
        df[c] = props[c].fillna(2016)
        df['year'] = df[c] - 1801
        df['age'] = 2017 - df[c]
        df[c] = df[c].astype('str')        
        
        c = 'hashottuborspa'
        df[c] = props[c] == True
        df[c] = df[c].astype('bool')

        c = 'taxdelinquencyflag'
        df[c] = props[c].astype('str')
        df[c] = df[c] == 'Y'
        df[c] = df[c].astype('bool')

        c = 'propertycountylandusecode'
        df[c] = props[c].fillna(CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)

        c = 'propertyzoningdesc'
        df[c] = props[c].fillna(CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)

        c = 'longitude'
        df[c] = props[c].fillna(0)
        m = df[c].mean()
        df['lon_adj'] = (df[c] - m).fillna(0)
        
        c = 'latitude'
        df[c] = props[c].fillna(0)
        m = df[c].mean()
        df['lat_adj'] = (df[c] - m).fillna(0)
        
        def distance(x, y):
            return np.sqrt(x**2 + y**2).sum()
        
        df['distance'] = distance(df['lon_adj'], df['lat_adj'])
        
        loc_fips_mean = props[['longitude', 'latitude', 'fips']].groupby(['fips']).mean().reset_index()
        loc_fips_mean.columns = ['fips', 'lon_fips_mean', 'lat_fips_mean']
        
        df = df.merge(loc_fips_mean, how='left', on='fips').fillna(0)
        df['lon_adj_fips'] = (df['longitude'] - df['lon_fips_mean']).fillna(0)
        df['lat_adj_fips'] = (df['latitude'] - df['lat_fips_mean']).fillna(0)
        df['distance_fips'] = distance(df['lon_adj_fips'], df['lat_adj_fips'])
        df = df.drop(['lon_fips_mean', 'lat_fips_mean'], axis=1)
        
        for c in props.columns:
            if c in df.columns:
                continue

            if 'typeid' in c or 'regionid' in c:
                df[c] = props[c].fillna(CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)
            elif props[c].dtype == np.object:
                df[c] = props[c] == True
                df[c] = df[c].astype('bool')
            else:
                df[c] = props[c]

            if df[c].dtype == np.float64:
                if (props[c] == CONTINUOUS_DEFAULT).any():
                    df[c] = props[c].fillna(props[c].mean())
                else:
                    df[c] = props[c].fillna(CONTINUOUS_DEFAULT)
                df[c] = df[c].astype(CONTINUOUS_DTYPE)

        self.props = df
        gc.collect()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# For pulling in input data
class DataLoader:

    def __init__(self):
        self.preprocessed = DataPreprocessor(self)
        
        self.props = None
        self.train = None
        self.subm = None

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
    
    def remove_outliers(self, X, y):
        f = np.abs(y-y.mean())<=(3*y.std())
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

    def train(self, params=None, dropcols=set(), verbose_eval=False):
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
        
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.data.preprocessed.training()
        
        # combine train and valid
        x_train = pd.concat([x_train, x_valid])
        y_train = pd.concat([y_train, y_valid])
        x_valid = x_test
        y_valid = y_test
        
        x_train, y_train = self.remove_outliers(x_train, y_train)

        d_train = xgb.DMatrix(self.adapt(x_train, dropcols), label=y_train, silent=True)
        d_valid = xgb.DMatrix(self.adapt(x_valid, dropcols), label=y_valid, silent=True)
        # d_test = xgb.DMatrix(self.adapt(x_test), label=y_test, silent=True)

        del x_train, x_valid, x_test; gc.collect()

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        evals_result = {}
        self.clf = xgb.train(params, d_train, 10000, watchlist, evals_result=evals_result, 
                             early_stopping_rounds=100, verbose_eval=verbose_eval)
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

            merged = zillow_data.preprocessed.prediction(month)
            subm['ParcelId'] = merged['parcelid']
            merged = adapt(merged, dropcols)

            dm_test = xgb.DMatrix(merged)
            del merged; gc.collect()
            
            subm[date] = clf.predict(dm_test)
            del dm_test; gc.collect()

        subm.to_csv('zillow/submission.csv.gz', index=False, float_format='%.4f', compression='gzip')
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

            def dense_stack(inp, dropout, units, layers):
                # den = Dropout(dropout)(inp)
                # den = BatchNormalization()(inp)
                # den = GaussianNoise(1e-5)(inp)
                den = inp
                for i in range(layers):
                    den = Dense(units, activation='relu', kernel_initializer='random_uniform')(den)
                return den
            
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
            # den = Dense(1024, activation='relu', kernel_initializer='random_uniform')(den)
            den = concatenate([ dense_stack(den, 0.2, 256, 2) for l in range(51) ])
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
