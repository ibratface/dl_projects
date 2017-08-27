from utils import *
import os
from sklearn import linear_model

CATEGORY_DEFAULT = ''
CATEGORY_DTYPE = 'str'
CONTINUOUS_DEFAULT = 0
CONTINUOUS_DTYPE = 'float32'

# For pulling in input data
class DataLoader:
    def __init__(self):
        self.props = pd.DataFrame()
        self.props_pre = pd.DataFrame()
        self.train = pd.DataFrame()
        self.train_pre = pd.DataFrame()
        self.train_mrg = pd.DataFrame()
        self.subm = pd.DataFrame()

        self.lm = linear_model.Ridge(alpha = .5)

        self.file_train = 'train_2016_v2'
        self.file_train_pre = 'train.pre'
        self.file_props = 'properties_2016'
        self.file_props_pre = 'props.pre'
        self.file_subm = 'sample_submission'

    def load_file(self, f, cache=True):
        pkl = os.path.join('zillow', f + '.pkl')
        csv = os.path.join('zillow', f + '.csv')
        if os.path.exists(pkl):
            df = pd.read_pickle(pkl)
        elif os.path.exists(csv):
            df = pd.read_csv(csv)
            if cache: df.to_pickle(pkl)
        else:
            df = pd.DataFrame()
        return df

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

    def preprocess_logerror(self, df, groupby, sfx_ave, sfx_std):
        l = 'logerror'
        la = 'logerror_abs'
        grouped = df.loc[:, groupby+[l, la]].groupby(groupby)
        ave = grouped.mean().rename(columns={l: l+sfx_ave, la: la+sfx_ave})
        std = grouped.var().rename(columns={l: l+sfx_std, la: la+sfx_std})
        return ave.merge(std, left_index=True, right_index=True)

    def forecast(self, df_train, df_preds, xcol, ycol):
        # print(df_train.loc[df_train[xcol]<11, [xcol]])
        self.lm.fit(df_train.loc[:, [xcol]], df_train.loc[:, [ycol]])
        df_preds[ycol] = self.lm.predict(df_preds.loc[:, [xcol]])
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

    def preprocess_training(self):
        trn = self.get_training()

        trn['logerror_abs'] = trn['logerror'].abs()
        trn = self.split_date(trn)

        self.logerror_month = self.preprocess_logerror(trn, ['transaction_month'], '_month_ave', '_month_std').reset_index()
        # self.logerror_month = self.logerror_month.append(self.forecast_month(self.logerror_month))
        self.logerror_month = self.forecast_month(self.logerror_month)
        self.logerror_month = self.logerror_month.reset_index(drop=True)

        self.train_pre = trn

    def preprocess_properties(self):
        pre = pd.DataFrame()
        c = 'rawcensustractandblock'

        pre[c] = self.props[c]
        pre = self.split_census(pre)

        c = 'calculatedbathnbr'
        pre[c] = self.props[c].fillna(self.props['bathroomcnt']+self.props['bedroomcnt'])
        pre[c] = pre[c].fillna(CONTINUOUS_DEFAULT)

        c = 'yearbuilt'
        pre[c] = self.props[c].fillna(2016)

        c = 'hashottuborspa'
        pre[c] = self.props[c] == True
        pre[c] = pre[c].astype('bool')

        c = 'taxdelinquencyflag'
        pre[c] = self.props[c].astype('str')
        pre[c] = pre[c] == 'Y'
        pre[c] = pre[c].astype('bool')

        c = 'propertycountylandusecode'
        pre[c] = self.props[c].fillna(CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)

        c = 'propertyzoningdesc'
        pre[c] = self.props[c].fillna(CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)

        for c in self.props.columns:
            if c in pre.columns:
                continue

            if 'typeid' in c or 'regionid' in c:
                pre[c] = self.props[c].fillna(CATEGORY_DEFAULT).astype(CATEGORY_DTYPE)
            elif self.props[c].dtype == np.object:
                pre[c] = self.props[c] == True
                pre[c] = pre[c].astype('bool')
            else:
                pre[c] = self.props[c]

            if pre[c].dtype == np.float64:
                if (self.props[c] == CONTINUOUS_DEFAULT).any():
                    pre[c] = self.props[c].fillna(self.props[c].mean())
                else:
                    pre[c] = self.props[c].fillna(CONTINUOUS_DEFAULT)
                pre[c] = pre[c].astype(CONTINUOUS_DTYPE)

        self.props_pre = pre

    def get_properties(self):
        if self.props.empty:
            self.props = self.load_file(self.file_props)
        return self.props

    def get_properties_preprocessed(self):
        if self.props_pre.empty:
            self.props_pre = self.load_file(self.file_props_pre)

        if self.props_pre.empty:
            self.get_properties()
            self.preprocess_properties()

        return self.props_pre

    def get_training(self):
        if self.train.empty:
            self.train = self.load_file(self.file_train)
        return self.train

    def get_training_preprocessed(self):
        if self.train_pre.empty:
            self.train_pre = self.load_file(self.file_train_pre)

        if self.train_pre.empty:
            self.get_training()
            self.preprocess_training()

        return self.train_pre

    def get_training_merged(self):
        if self.train_mrg.empty:
            trn = self.get_training_preprocessed()
            prp = self.get_properties_preprocessed()
            mrg = trn.merge(prp, how='left', on='parcelid')

            self.logerror_fips_month = self.preprocess_logerror(mrg, ['fips', 'transaction_month'], '_fips_month_ave', '_fips_month_std').reset_index()
            # self.logerror_fips_month = self.logerror_fips_month.append(self.forecast_fips_month(self.logerror_fips_month))
            self.logerror_fips_month = self.forecast_fips_month(self.logerror_fips_month)
            self.logerror_fips_month = self.logerror_fips_month.reset_index(drop=True)

            mrg = mrg.merge(self.logerror_month, how='left', on=['transaction_month'])
            mrg = mrg.merge(self.logerror_fips_month, how='left', on=['fips', 'transaction_month'])

            self.train_mrg = mrg
        return self.train_mrg

    def get_submission(self):
        if self.subm.empty:
            self.subm = self.load_file(self.file_subm)
        return self.subm

    def get_prediction(self, month):
        preds = self.get_properties_preprocessed().copy()
        preds.insert(0, 'transaction_month', month)
        preds = preds.merge(self.logerror_month, how='left', on=['transaction_month'])
        preds = preds.merge(self.logerror_fips_month, how='left', on=['fips', 'transaction_month'])
        return preds
