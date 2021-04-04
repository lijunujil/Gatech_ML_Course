import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# def bank_data(n_data=None):
#     data_df = pd.read_csv("bank-additional.csv", sep=";")
#
#     # convert them to binary
#     data_df['y'] = data_df['y'].map({'no': 0, 'yes': 1}).astype('uint8')
#     data_df["default"] = data_df["default"].map({'no': 1, 'unknown': 0, 'yes': 0}).astype('uint8')
#     data_df["housing"] = data_df["housing"].map({'yes': 1, 'unknown': 0, 'no': 0}).astype('uint8')
#     data_df["loan"] = data_df["loan"].map({'yes': 1, 'unknown': 0, 'no': 0}).astype('uint8')
#     data_df["contact"] = data_df["contact"].map({'cellular': 1, 'telephone': 0}).astype('uint8')
#     data_df["pdays"] = data_df["pdays"].replace(999, 0).astype('uint8')
#     data_df["previous"] = data_df["previous"].apply(lambda x: 1 if x > 0 else 0).astype('uint8')
#     data_df["poutcome"] = data_df["poutcome"].map({'nonexistent': 0, 'failure': 0, 'success': 1}).astype('uint8')
#
#     # normalized to start from 0
#     data_df['cons.price.idx'] = (data_df['cons.price.idx'] * 10).astype('uint8')
#     data_df['cons.price.idx'] = data_df['cons.price.idx'] - data_df['cons.price.idx'].min()
#
#     data_df['cons.conf.idx'] = data_df['cons.conf.idx'] * -1
#     data_df['cons.conf.idx'] = data_df['cons.conf.idx'] - data_df['cons.conf.idx'].min()
#
#     # log transformation
#     data_df['nr.employed'] = np.log2(data_df['nr.employed']).astype('uint8')
#
#     data_df["euribor3m"] = data_df["euribor3m"].astype('uint8')
#     data_df["campaign"] = data_df["campaign"].astype('uint8')
#     data_df["pdays"] = data_df["pdays"].astype('uint8')
#
#     data_df = pd.concat([data_df, pd.get_dummies(data_df["job"], prefix="job")], axis=1)
#     data_df = pd.concat([data_df, pd.get_dummies(data_df["education"], prefix="education")], axis=1)
#     data_df = pd.concat([data_df, pd.get_dummies(data_df["marital"], prefix="marital")], axis=1)
#     data_df = pd.concat([data_df, pd.get_dummies(data_df["month"], prefix="month")], axis=1)
#     data_df = pd.concat([data_df, pd.get_dummies(data_df["day_of_week"], prefix="day_of_week")], axis=1)
#     data_df['age_group'] = pd.cut(data_df['age'], bins=[0, 14, 24, 64, float('inf')], labels=[1, 2, 3, 4], include_lowest=True).astype('uint8')
#     data_df['duration_group'] = pd.cut(data_df['duration'], bins=[0, 120, 240, 360, 480, float('inf')], labels=[1, 2, 3, 4, 5], include_lowest=True).astype('uint8')
#
#     data_df.drop(['job', "education", 'marital', 'month', 'day_of_week', 'age', 'duration'], axis=1, inplace=True)
#
#     if n_data is not None:
#         data_df = data_df[:n_data]
#     # print(data_df.dtypes)
#     train, target = data_df[data_df.columns.difference(['y'])], data_df['y']
#     return train, target

def bank_data_original():
    data_df = pd.read_csv("bank-additional.csv", sep=";")

    data_df['y'] = data_df['y'].map({'yes': 1, 'no': 0})

    transform_list = data_df.select_dtypes(exclude=np.number).columns.tolist()
    data_df[transform_list] = data_df[transform_list].apply(LabelEncoder().fit_transform)

    return data_df


def bank_data(n_data=None):
    data_df = pd.read_csv("bank-additional.csv", sep=";")

    data_df['y'] = data_df['y'].map({'yes': 1, 'no': 0})
    data_df = pd.get_dummies(data_df, drop_first=True)

    if n_data is not None:
        data_df = data_df[:n_data]
    # print(data_df.dtypes)
    train = data_df.drop(['y'], axis=1)
    target = data_df['y']

    scaler = StandardScaler()
    transformed_train = pd.DataFrame(scaler.fit_transform(train))
    transformed_train.columns = train.columns

    return transformed_train, target


def digits_data():
    df_train = pd.read_csv("optdigits_train.csv", header=None)
    df_test = pd.read_csv("optdigits_test.csv", header=None)
    df_combined = df_train.append(df_test)
    train, target = df_combined.iloc[:, :-1], df_combined.iloc[:, -1]

    # scaler = StandardScaler()
    # transformed_train = pd.DataFrame(scaler.fit_transform(train))
    # transformed_train.columns = train.columns

    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # train, target = digits.data, digits.target

    return train, target
