---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="xg5m6EgrhS4t" -->
# RecSys RetailRocket
<!-- #endregion -->

<!-- #region id="fH3nvW9aCR6Q" -->
## Setup
<!-- #endregion -->

```python id="C8xVE9Qea_Jj"
# !pip install -q -U kaggle
# !pip install --upgrade --force-reinstall --no-deps kaggle
# !mkdir ~/.kaggle
# !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# # !kaggle datasets list
```

```python id="XcAYGpO6cmCE"
!kaggle datasets download -d retailrocket/ecommerce-dataset
!mkdir -p ./data && unzip ecommerce-dataset.zip
!mv ./*.csv ./data && rm ecommerce-dataset.zip
```

```python id="HxrmQPXZgKaB"
import os
import re
import time
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

import bz2
import csv
import json
import operator

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```

<!-- #region id="3et4hlvIMnGu" -->
## Data Loading
<!-- #endregion -->

```python id="RZgv-uQyMBj0"
events_df = pd.read_csv('./data/events.csv')
category_tree_df = pd.read_csv('./data/category_tree.csv')
item_properties_1_df = pd.read_csv('./data/item_properties_part1.csv')
item_properties_2_df = pd.read_csv('./data/item_properties_part2.csv')
```

```python id="Nb6JTWWkcrZv"
item_prop_df = pd.concat([item_properties_1_df, item_properties_2_df])
item_prop_df.reset_index(drop=True, inplace=True)
del item_properties_1_df
del item_properties_2_df
```

```python id="kewR0WW1HoPI" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1611216398306, "user_tz": -330, "elapsed": 1498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="645140b1-0920-424e-97ae-a204c259251a"
events_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="kWMyH56HNVnC" executionInfo={"status": "ok", "timestamp": 1611216400383, "user_tz": -330, "elapsed": 1692, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c9bd9d2f-fdd3-4455-8f44-5272968c63b2"
item_prop_df.head()
```

<!-- #region id="VYBJSHOEOLC9" -->
- Property is the Item's attributes such as category id and availability while the rest are hashed for confidentiality purposes

- Value is the item's property value e.g. availability is 1 if there is stock and 0 otherwise

- Note: Values that start with "n" indicate that the value preceeding it is a number e.g. n277.200 is equal to `277.2`
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yK69-KAAOX87" executionInfo={"status": "ok", "timestamp": 1611216653504, "user_tz": -330, "elapsed": 1573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e75d8eb3-44de-4646-8b71-8658f793f9bd"
category_tree_df.head()
```

<!-- #region id="Pp0jvUKXA4GE" -->
## EDA
<!-- #endregion -->

<!-- #region id="SHJe-s8p6mXl" -->
Q: what are the items under category id `1016`?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="M-rO7aQSOh9t" executionInfo={"status": "ok", "timestamp": 1610435781867, "user_tz": -330, "elapsed": 3938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="139f4ca4-6d63-4f63-d3cf-d053e6fd936c"
item_prop_df.loc[(item_prop_df.property == 'categoryid') & (item_prop_df.value == '1016')].sort_values('timestamp').head()
```

<!-- #region id="6AKenTUXDqhK" -->
Q: What is the parent category of `1016`?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SdfJDVpwDyFY" executionInfo={"status": "ok", "timestamp": 1611234517954, "user_tz": -330, "elapsed": 2112, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c464f4bf-57ff-4d6d-e61e-72ca8351ca76"
category_tree_df[category_tree_df.categoryid==1016]
```

<!-- #region id="Z_4-hu5pEGDu" -->
Q: What are items under category `213`?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1l8SbTWvD_Zc" executionInfo={"status": "ok", "timestamp": 1611234585996, "user_tz": -330, "elapsed": 6128, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="503e9b23-1185-46f4-8240-53acb0de14e5"
item_prop_df.loc[(item_prop_df.property == 'categoryid') & (item_prop_df.value == '213')].sort_values('timestamp').head()
```

<!-- #region id="SG8zLy9LEn0p" -->
visitors who bought something, assuming that there were no repeat users with different visitor IDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GirnbnFQPnnf" executionInfo={"status": "ok", "timestamp": 1610433500322, "user_tz": -330, "elapsed": 1135, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="721b62cf-2272-484c-df8b-68ee7fbd957e"
customer_purchased = events_df[events_df.transactionid.notnull()].visitorid.unique()
all_customers = events_df.visitorid.unique()
customer_browsed = [x for x in all_customers if x not in customer_purchased]
print("%d out of %d"%(len(all_customers)-len(customer_browsed), len(all_customers)))
```

<!-- #region id="fQVOxoOlR1YP" -->
Snapshot of a random session with visitor id 102019
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 328} id="DrGTwmU_Rtw6" executionInfo={"status": "ok", "timestamp": 1610432898423, "user_tz": -330, "elapsed": 1395, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="726a5542-de68-433c-cbab-3eb1aa03222f"
events_df[events_df.visitorid == 102019].sort_values('timestamp')
```

```python colab={"base_uri": "https://localhost:8080/"} id="D3iV5U_FGRP8" executionInfo={"status": "ok", "timestamp": 1611218631523, "user_tz": -330, "elapsed": 1200, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="442555a6-5cb4-453f-d45d-e0c217eae1a9"
def _todatetime(dt):
  return datetime.datetime.fromtimestamp(int(dt/1000)).strftime('%Y-%m-%d %H:%M:%S')

print('Range of transaction dates = ', _todatetime(events_df['timestamp'].min()), 'to', _todatetime(events_df['timestamp'].max()))
```

<!-- #region id="dBx7rbofBRZu" -->
## Preprocessing
<!-- #endregion -->

```python id="hJ26GxG9EObt"
def preprocess_events(df):

  # convert unix time to pandas datetime
  df['date'] = pd.to_datetime(df['timestamp'], unit='ms', origin='unix')
  
  # label the events
  # events.event.replace(to_replace=dict(view=1, addtocart=2, transaction=3), inplace=True)

  # convert event to categorical
  df['event_type'] = df['event'].astype('category')

  # # drop the transcationid and timestamp columns
  # df.drop(['transactionid', 'timestamp'], axis=1, inplace=True)

  # # label encode
  # le_users = LabelEncoder()
  # le_items = LabelEncoder()
  # events['visitorid'] = le_users.fit_transform(events['visitorid'])
  # events['itemid'] = le_items.fit_transform(events['itemid'])
  
  # return train, valid, test
  return df
```

```python colab={"base_uri": "https://localhost:8080/"} id="i6B_f-huEjlj" executionInfo={"status": "ok", "timestamp": 1611219049016, "user_tz": -330, "elapsed": 1716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d7a382a4-ceb4-4a44-a804-33a67cc7b0fa"
events_processed = preprocess_events(events_df)
events_processed.head()
```

```python id="bw2lm-vll8Bc"
dfx = events_df.sample(frac=0.01)
```

```python id="41ZS2fxEdkeb"
def sessionize(events_df: pd.DataFrame):

  session_duration = datetime.timedelta(minutes=30)
  gpby_visitorid = events_df.groupby('visitorid')

  session_list = []
  for a_visitorid in gpby_visitorid.groups:

    visitor_df = events_df.loc[gpby_visitorid.groups[a_visitorid], :].sort_values('date')
    if not visitor_df.empty:
        visitor_df.sort_values('date', inplace=True)

        # Initialise first session
        startdate = visitor_df.iloc[0, :]['date']
        visitorid = a_visitorid
        items_dict = dict([ (i, []) for i in events_df['event_type'].cat.categories ])
        for index, row in visitor_df.iterrows():

            # Check if current event date is within session duration
            if row['date'] - startdate <= session_duration:
            # Add itemid to the list according to event type (i.e. view, addtocart or transaction)
                items_dict[row['event']].append(row['itemid'])
                enddate = row['date']
            else:
                # Complete current session
                session_list.append([visitorid, startdate, enddate] + [ value for key, value in items_dict.items() ])
                # Start a new session
                startdate = row['date']
                items_dict = dict([ (i, []) for i in events_df['event_type'].cat.categories ])
                # Add current itemid
                items_dict[row['event']].append(row['itemid'])

        # If dict if not empty, add item data as last session.
        incomplete_session = False
        for key, value in items_dict.items():
            if value:
                incomplete_session = True
                break
        if incomplete_session:
            session_list.append([visitorid, startdate, enddate] + [value for key, value in items_dict.items()])

  return session_list
```

```python colab={"base_uri": "https://localhost:8080/"} id="78f7h51riXNa" executionInfo={"status": "ok", "timestamp": 1611219197499, "user_tz": -330, "elapsed": 71296, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="985652e7-1650-4af5-e4fb-a3a10c66e315"
session_list = sessionize(dfx)
sessions_df = pd.DataFrame(session_list, columns=['visitorid', 'startdate', 'enddate', 'addtocart', 'transaction', 'view'])
sessions_df.head()
```

```python id="DY8lnv5qMM35"
class BaseDataset(object):
    def __init__(self, input_path, output_path):
        super(BaseDataset, self).__init__()

        self.dataset_name = ''
        self.input_path = input_path
        self.output_path = output_path
        self.check_output_path()

        # input file
        self.inter_file = os.path.join(self.input_path, 'inters.dat')
        self.item_file = os.path.join(self.input_path, 'items.dat')
        self.user_file = os.path.join(self.input_path, 'users.dat')
        self.sep = '\t'

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # selected feature fields
        self.inter_fields = {}
        self.item_fields = {}
        self.user_fields = {}

    def check_output_path(self):
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

    def get_output_files(self):
        output_inter_file = os.path.join(self.output_path, self.dataset_name + '.inter')
        output_item_file = os.path.join(self.output_path, self.dataset_name + '.item')
        output_user_file = os.path.join(self.output_path, self.dataset_name + '.user')
        return output_inter_file, output_item_file, output_user_file

    def load_inter_data(self) -> pd.DataFrame():
        raise NotImplementedError

    def load_item_data(self) -> pd.DataFrame():
        raise NotImplementedError

    def load_user_data(self) -> pd.DataFrame():
        raise NotImplementedError

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_user(self):
        try:
            input_user_data = self.load_user_data()
            self.convert(input_user_data, self.user_fields, self.output_user_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to user file\n')

    @staticmethod
    def convert(input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[column] = input_data.iloc[:, column]
        with open(output_file, 'w') as fp:
            fp.write('\t'.join([selected_fields[column] for column in output_data.columns]) + '\n')
            for i in tqdm(range(output_data.shape[0])):
                fp.write('\t'.join([str(output_data.iloc[i, j])
                                    for j in range(output_data.shape[1])]) + '\n')

    def parse_json(self, data_path):
        with open(data_path, 'rb') as g:
            for l in g:
                yield eval(l)

    def getDF(self, data_path):
        i = 0
        df = {}
        for d in self.parse_json(data_path):
            df[i] = d
            i += 1
        data = pd.DataFrame.from_dict(df, orient='index')
        
        return data
```

```python id="_mqXrXmnMIlT"
class RETAILROCKETDataset(BaseDataset):
    def __init__(self, input_path, output_path, interaction_type, duplicate_removal):
        super(RETAILROCKETDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'retailrocket'
        self.interaction_type = interaction_type
        assert self.interaction_type in ['view', 'addtocart',
                                         'transaction'], 'interaction_type must be in [view, addtocart, transaction]'
        self.duplicate_removal = duplicate_removal

        # input file
        self.inter_file = os.path.join(self.input_path, 'events.csv')
        self.item_file1 = os.path.join(self.input_path, 'item_properties_part1.csv')
        self.item_file2 = os.path.join(self.input_path, 'item_properties_part2.csv')
        self.sep = ','

        # output file
        if self.interaction_type == 'view':
            self.output_inter_file = os.path.join(self.output_path, 'retailrocket-view.inter')
        elif self.interaction_type == 'addtocart':
            self.output_inter_file = os.path.join(self.output_path, 'retailrocket-addtocart.inter')
        elif self.interaction_type == 'transaction':
            self.output_inter_file = os.path.join(self.output_path, 'retailrocket-transaction.inter')
        self.output_item_file = os.path.join(self.output_path, 'retailrocket.item')

        # selected feature fields
        if self.duplicate_removal:
            if self.interaction_type == 'view':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'count:float'}
            elif self.interaction_type == 'addtocart':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'count:float'}
            elif self.interaction_type == 'transaction':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'count:float'}
        else:
            if self.interaction_type == 'view':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token'}
            elif self.interaction_type == 'addtocart':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token'}
            elif self.interaction_type == 'transaction':
                self.inter_fields = {0: 'timestamp:float',
                                     1: 'visitor_id:token',
                                     2: 'item_id:token',
                                     3: 'transaction_id:token'}
        self.item_fields = {0: 'item_timestamp:float',
                            1: 'item_id:token',
                            2: 'property:token',
                            3: 'value:token_seq'}

    def convert_inter(self):
        if self.duplicate_removal:
            fin = open(self.inter_file, "r")
            fout = open(self.output_inter_file, "w")

            lines_count = 0
            for _ in fin:
                lines_count += 1
            fin.seek(0, 0)

            fout.write('\t'.join([self.inter_fields[column] for column in self.inter_fields.keys()]) + '\n')
            dic = {}

            for i in tqdm(range(lines_count)):
                if i == 0:
                    fin.readline()
                    continue
                line = fin.readline()
                line_list = line.split(',')
                key = (line_list[1], line_list[3])
                if line_list[2] == self.interaction_type:
                    if key not in dic:
                        dic[key] = (line_list[0], 1)
                    else:
                        if line_list[0] > dic[key][0]:
                            dic[key] = (line_list[0], dic[key][1] + 1)
                        else:
                            dic[key] = (dic[key][0], dic[key][1] + 1)

            for key in dic.keys():
                fout.write(dic[key][0] + '\t' + key[0] + '\t' + key[1] + '\t' + str(dic[key][1]) + '\n')

            fin.close()
            fout.close()
        else:
            fin = open(self.inter_file, "r")
            fout = open(self.output_inter_file, "w")

            lines_count = 0
            for _ in fin:
                lines_count += 1
            fin.seek(0, 0)

            fout.write('\t'.join([self.inter_fields[column] for column in self.inter_fields.keys()]) + '\n')

            for i in tqdm(range(lines_count)):
                if i == 0:
                    fin.readline()
                    continue
                line = fin.readline()
                line_list = line.split(',')
                if line_list[2] == self.interaction_type:
                    if self.interaction_type != 'transaction':
                        del line_list[4]
                    else:
                        line_list[4] = line_list[4].strip()
                    del line_list[2]
                    fout.write('\t'.join([str(line_list[i]) for i in range(len(line_list))]) + '\n')

            fin.close()
            fout.close()

    def convert_item(self):
        fin1 = open(self.item_file1, "r")
        fin2 = open(self.item_file2, "r")
        fout = open(self.output_item_file, "w")

        lines_count1 = 0
        for _ in fin1:
            lines_count1 += 1
        fin1.seek(0, 0)

        lines_count2 = 0
        for _ in fin2:
            lines_count2 += 1
        fin2.seek(0, 0)

        fout.write('\t'.join([self.item_fields[column] for column in self.item_fields.keys()]) + '\n')

        for i in tqdm(range(lines_count1)):
            if i == 0:
                line = fin1.readline()
                continue
            line = fin1.readline()
            line_list = line.split(',')
            fout.write('\t'.join([str(line_list[i]) for i in range(len(line_list))]))

        for i in tqdm(range(lines_count2)):
            if i == 0:
                line = fin2.readline()
                continue
            line = fin2.readline()
            line_list = line.split(',')
            fout.write('\t'.join([str(line_list[i]) for i in range(len(line_list))]))

        fin1.close()
        fin2.close()
        fout.close()
```

```python colab={"base_uri": "https://localhost:8080/"} id="HJOZgfHpNIHt" executionInfo={"status": "ok", "timestamp": 1611237234343, "user_tz": -330, "elapsed": 50991, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d325b981-696d-4f76-aaf8-8a295fdb419d"
# data_object = RETAILROCKETDataset('./data', '.', 'view', True)
# data_object.convert_inter()
# data_object.convert_item()
```

<!-- #region id="D_19B88xJufx" -->
## Feature Engineering
<!-- #endregion -->

<!-- #region id="iFBpIeaZJyr_" -->
Page Time
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="StuYb5igkAHR" executionInfo={"status": "ok", "timestamp": 1611220625580, "user_tz": -330, "elapsed": 2550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45de3b2c-9d93-4a81-bc90-2826c56e14aa"
sessions_df['pages'] = sessions_df['view'].apply(lambda x: len(x))
pages_more_than1 = sessions_df['pages'] > 1
pages_less_than1 = pages_more_than1.apply(lambda x: not x)
sessions_df.loc[pages_more_than1, 'pagetime'] = (sessions_df.loc[pages_more_than1, 'enddate'] - sessions_df.loc[pages_more_than1, 'startdate']) /\
                                                (sessions_df.loc[pages_more_than1, 'pages'] - 1)
sessions_df.loc[pages_less_than1, 'pagetime'] = pd.Timedelta(0)
sessions_df.head(10)
```

<!-- #region id="SEElLub9sQjg" -->
The rule of thumb on creating a simple yet effective recommender system is to downsample the data without losing quality. It means, you can take only maybe 50 latest transactions for each user and you still get the quality you want because behavior changes over-time.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="viVKSGKAsOex" executionInfo={"status": "ok", "timestamp": 1610439976574, "user_tz": -330, "elapsed": 2076, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4bbfd66e-613d-407e-a21b-6940c872159d"
trans = events_df[events_df['event'] == 'transaction']
trans2 = trans.groupby(['visitorid']).head(50)
trans2
```

```python id="bYPEhXJTtBQl"
visitors = trans['visitorid'].unique()
items = trans['itemid'].unique()

trans2['visitors'] = trans2['visitorid'].apply(lambda x : np.argwhere(visitors == x)[0][0])
trans2['items'] = trans2['itemid'].apply(lambda x : np.argwhere(items == x)[0][0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="FxW7xi0YtYgt" executionInfo={"status": "ok", "timestamp": 1610440089282, "user_tz": -330, "elapsed": 1419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a3861d50-ab77-4107-cf96-da7eaca5cbe9"
trans2.head()
```

<!-- #region id="8HmRqRC0tuFt" -->
Create the user-item matrix
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sMIQd7DBtg7i" executionInfo={"status": "ok", "timestamp": 1610440152639, "user_tz": -330, "elapsed": 11930, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70845225-35c8-4360-d3ed-646543dc9081"
from scipy.sparse import csr_matrix

occurences = csr_matrix((visitors.shape[0], items.shape[0]), dtype='int8')

def set_occurences(visitor, item):
    occurences[visitor, item] += 1

trans2.apply(lambda row: set_occurences(row['visitors'], row['items']), axis=1)

occurences
```

<!-- #region id="dAfq2N60uBrf" -->
Co-occurrence is a better occurrence

Let’s construct an item-item matrix where each element means how many times both items bought together by a user. Call it the co-occurrence matrix.
<!-- #endregion -->

```python id="-eM9fETruJz3"
cooc = occurences.transpose().dot(occurences)
cooc.setdiag(0)
```

```python id="lJ7Nql5kEK8H"
  # split into train, test and valid
  train, test = train_test_split(events, train_size=0.9)
  train, valid = train_test_split(train, train_size=0.9)
  print('Train:{}, Valid:{}, Test:{}'.format(train.shape,
                                            valid.shape,
                                            test.shape))
```

<!-- #region id="C1jTXQCRQJKO" -->
https://nbviewer.jupyter.org/github/tkokkeng/EB5202-RetailRocket/blob/master/retailrocket-features.ipynb
<!-- #endregion -->

<!-- #region id="Q1dfT0E_QHB8" -->
![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAwQAAACoCAYAAABezSnqAAAgAElEQVR4Ae2dTW7cuNOH+4YGDOQu/03gF4Hv4Wwyx0h2mTvMIptZzAm81guSKrKqRFJSf0qtZ4CBWy1+VD31Y5Gl7tingf8gAAEIQAACEIAABCAAgcMSOB3WcxyHAAQgAAEIQAACEIAABAYKAkQAAQhAAAIQgAAEIACBAxOgIDhw8HEdAhCAAAQgAAEIQAACFARoAAIQgAAEIAABCEAAAgcmQEFw4ODjOgQgAAEIQAACEIAABCgI0AAEIAABCEAAAhCAAAQOTICC4MDBx3UIQAACEIAABCAAAQhQEKABCEAAAhCAAAQgAAEIHJjAgwuC/4YfX96H33sIwL8/htdvo6Xh9ek0vP713x4sx0YIQOCZCfz9PpxOp+H972d2Et9uS0DvxeH1aTh9+TFsZoczGvf2jden03A6vQ4//vXXZ5LTe/6ZQ+hu//31Oto3DIM/Q4z+hXV8CucMf60HuvJrY9eVx758uN/DuzC5ZLArx/ISU7bc96EFwe9vYQHvoSC4kii3rARsgwAEIACBQxLY/F5sCgIXovFwHQ/S4Za/ds2XXV5/z+8dvBP/UMwk6/z1MpvPa9Wz67wRr9nrGnG4xhjX9Gm7Yz2sIEgiDAWBq+q/vA/v4elErPSHYdpuhCkJ4q/0tD6Mo5/Y23626EiLTebW9/SThbFSH9x74amJr+6HYTBj6icrM3ZuVxpYBgEIbIVAyi8lV5lNXHKMfEIwXqfcKn3cpiht5FNPP8Zix8f8mPN2yd0yhMmN+gGQHNziE8D3yZNAk8N1TtX9nvKTkfsyNZz1E/aRedbaX+mTqLTX/k6fIsT9WzQWIj7qLL5/2adW2q73b/pTMOETPsGw852+/N/wv3HuqH/Rjeg93isH71w8jH3SGaKy54uYV/4s2n8d3r/VPyHQfgab/xfbyfkkMBR/5Wzk1pis3ZGRFEZl7umnPeWes2ulf7G5nIe+vcdvThjucn7K+UHY27jps5uJybdpXjCfnoznRG1HnD+fB68RS8mdP6zmlaa0/Tae4m85y6ZPcscxoz59fMc+anzz4NzHO2hX8vg58VN9HlYQBBuSKCWZSOAKwCQMua8BDuUjtRGEGUsEGr/SI8Ecv+4TYEqSkI1FjyH3BHrcZN0YZnzvh2srQdVz6E1RBYOXEIAABKoEYh6R3CgbyPiVDp2rTG6y7XSOzJvWmO/SteTaqgWNN33eruRpyakm30q/cU6XJ2XTN5tnzKFtnxoG7vBtYSPxvjVTv4dZxqIVf1iOsanF1GhKfFgZBjOu2CcFhrUvHyDHPbZ1nbSk+6bX6X2ZQ+x1+/hK82PzUdOW2zi+Wad+bn892pwPvxU9uEOhXuu5SBM+PbvO8VNiJet8dnzHdmxv4iNnpPFePvCOc5m2cd5bxnK019s0+ptY67hKHnVxkuLo9D68x2/HiNZ8fMunXEY7jq8wkfUpOj4nhNJnewWBOC0WDqWyMpWnEZ200UFJFbau3NSQ9ol+XCgpePX2TsBmMbt72d7Rlp6d2iBeQwACEGgSUPnJ5J/ycCRsCGlzkI1GrvWhO91Lm5g8aRw3JDkwNG2o3Rj7qrztbQi9ynzj0yw5ROQ5bR61B5qxf5xDNs/pU8+adft8795MJT5ykLHz+3ja2Ki4eV2O1+ccVPyctkC09rUKAHtgEt8qihAtmk8PlF+VLkvespxkLdozipw3Utuybu2187dxxiicp7Zrnl27ljjm2/i4y8FXr1eVH2wsw2DKv8lY1pfkx71jaW0QvUnsNFtBk94bP+Wp+B7Psjn3Kf/HAfyYxm9TQIUOzj4x4oyf2y4IpDqsCctB8QCT6MeA5I+PZFHKRywK5ESImqZqF97WbfXrsYuxZcZOPQuvIQABCLQIxJz25cfwW//jxNBY5Rif99LH5/4Q8iN+Pef1r/TzffzapWxwrfnr7/c3s5QLl+RbnWPHMcevcchXAPLH5rIvyH294daN3Nm792dqD4l2frOf5eJODmUqbj4uY3zO0ZWfU2vcHCBDZMc9WAoAf21901IYbTdfT5ZDufJLd1nx2s9rfHLnhtRW5pYCTa5tPIIJZiy1/qN5bmzfvmvXCv9y08l82l79OvUwtse3VJvJWDYO3vZsgxyKbxJLa4PoS3Rt/JE1UDuzal/zpz3hTeX/6FDys5xfTR738Rbfc4FRqKx9temCwAbfQXNQTFA0BQlQDIAbw4B0QddjmHYlASVBTPsZW5baaebjAgIQgIAjEHPJ6/DqfwOMyjEm97juZeN5HV51PjRPRiedZt7wOVUfVvw9lSvHjT8f4lyOtbm/ZcI4nnrg02q5r/c9t9sztbzt/F5Ttu00pnJQuoR5mlO+ImSL3qLj8StzXkvu2tuf7VLrJrxn2ym/cod1LxInOdS78d3B17e11zYeE1udH7Unxto3O7aza52LqbXzxcZnarst7qI35bdaudh5X7QfxlTHwLa7NJauv/NXz5XYSrE89T219Z8ctNoV7fR89YxM25UXmy4INGgRUf7+f08A7p4Okn6dgzNWVvqerQL7gjD93MYmdsvHecanlcGiOQQgcGQCYx7yB2Cd78xm1dpowoaUNq2cA89+yj7Okf9x72jjOJ7OjXmumG+ln7NDnnJpn0xOdbnYtHsWbQgbORDfmqk8ka4fZPyepWNqDyNWb77fqui4g2GaU3jYeWSvzsWl6yv3pVDJ9pu1ImtLDmFOZ6uMHxuP2kzzSkzH8c3cwl/m9tfSV/y3evBnjDB79jGa4nzp2XWOn8Lbfcfe+G3yS92edEYSX0ctjrb62N43ls5eFzutc/1a4pLPrMLp2+/8y3KajMwcTu/CZGSa5hRtnBPA0uehBUEGFjcT57QW8vgEKz4Zc6JrHbQFknzcLO0kOchHMPZpm4hxrOBkc8oLbNxITbASTElYcVzVT3yU+Y1gShx4BQEIQGCWQMoz5eAQO4wbhOQYyTkpx8khbxxabUq6b95wZy3wDUre/hH/oVwpNmJLmU/ncDkc6HuV3yZicrj0CYPqfmFcnW+9ebu8fgBTOWRU9mK/Z3UPm1K8xXhfeEjRNjV/y5DSg+hA9CHXQQNqLPkNhuFtvW+/fgm/BajYXO65NbRCU0XD7rf5jDbag21Z13adz+jBr//RvmL/dI007VrhW26afUlfQbRnoGK7/ZsW4yG7qhN1r5IX7h/L5QVBLpB1vhvPrNWYxnsNRkazSoMS78Zv2MxxOePFYwuCMwymCwQgAAEIbIVAYzObNc/1y4cKe2yYHeYpGzg2i310/WC6mNy2G7q4bs1YdHbfiEhBIL9m+oqzUxBcESZDQQACEDgWgQsOK+YJ2DP/1qC1ioDpWmLP3f4CPdwDDAXBPSiXOSgICgteQQACEIAABCAAAQhAAALXI8AnBNdjyUgQgAAEIAABCEAAAhDYHQEKgt2FDIMhAAEIQAACEIAABCBwPQIUBNdjyUgQgAAEIAABCEAAAhDYHQEKgt2FDIMhAAEIQAACEIAABCBwPQIUBNdjyUgQgAAEIAABCEAAAhDYHQEKgt2FDIMhAAEIQAACEIAABCBwPQLVgiD9hcvxr/WOf0mO9+CBBtAAGkADaAANoAE0gAb2o4GlJUO1IPj8/Bz4HwZoAA2gATSABtAAGkADaGC/GqAgoKihqEMDaAANoAE0gAbQABo4sAYoCA4cfCr5/VbyxI7YoQE0gAbQABpAA9fSAAUBBQFPBNAAGkADaAANoAE0gAYOrAEKggMH/1pVJePwhAINoAE0gAbQABpAA/vVAAUBBQFPBNAAGkADaAANoAE0gAYOrAEKggMHn0p+v5U8sSN2aAANoAE0gAbQwLU0QEFAQcATATSABtAAGkADaAANoIEDa+CGBcHP4S3/sbKX4eMfqrh+FVd4vf36M3y8pj9m8fYLbn1u8Lktn6LFl+9/4mbx86v+Qytvw0+VQPU9o91/PoaXnA9Cf9Xv19uQ/3jN68fwJ49X1sTpNM0hZa7pvTqTzngN+/58fym2ZfvLfMWGxEQYhfn1Pf1+sS3ZYzhl36e6ro03Z9+n9uvrz+pmH8dV9yZjxntFB0vtLX5OfbnLPe271luHcd8upR+j089BM+vxabbTa+B0GnpjiI3duKl4Svv8U3Nxfhi9GGYl/tO12OZixnM2FT2X9ZRtPDtG67QWbKivzXEczWqD9l/MK/qncnGF+xbidJGfR/CxEjfDbAGD0H7pf6v/MFkU0ZhskqD6ojPGzzn3lPdDUrWJMXBbsjHAbt0mAK81vNJBIOswHFzUxhjX9ngdDztywIgJSOk59JN7ev3GdkXnejyj/3hgKjnEzOXuteLbG++zZZ+2VQ752X/HRrU19n1O13awMfq68PC3ZLw8ZrYvHdRS7JKtk8NPZHeaxHTSbvTNMFT+tpg/7v3gb9Gf5bdG/9LW8ouxE85af173mlGzXYhT0XbUor7WY8hrH7cwb15f1lYbA62JUYPiR9RpWYuR2Xiv6e+nncu0M+PZdiYemov4d4ef0dZTryDQrLZnv42r6HTFzzH3moczjvsW4nSRn0fw0cVswmsBA+lzo4IgLaR8cBiTVz5UzDlwyPuBWdnAQoBCwoLZigR3SN3cmk/aCJs6VJt5+zAwPkHNBw9lc+ifDzKfQ3qiqA5HOaZ2fYS5yqFVb9xq7Ny39p4dTx9+JDlOfsbEqm2zY5T2gZldy+XeaEvw++tb/BS1yTbbv2C80Nbb567N5h7HDva/DW/hE58cm368d5uTHItJPDLrmlbCey7WSvdWO4mfaFPrtNfO2hPmktzv5m3Gzdpt51L3IoeiTaOJqMnap0jWp6QzGcPZp7h4Peq5NJfEVvxVts7G5Ny2yZ+geWuHG89pZjv2OzvP4RTj9DJ8/Aqf3OqcZse2fLQubTur343cO4KPc7FfyEDid5uCIC4ktWH76zknDnnfJVYKgurXG0S4/LxX0u0fEPUmaZ5suoNH2Fjy14LyYedzmDyZd/1ynPVBY3wqWQ7S7sCyJH+Y8crTerGxjF04281RDuDKr1zYhLX8NnyorxvJATH5I2t96QY7N16ycWJf8DHbNLJWm7+0j7HJBUGySTj4p4ehbY1NjtMS9g9oY3R6zvxRl+rgpHQqHIWB5dnWT6udPXCX/nr8oKdmf/fUXvqln3at6DECo5fvH+qrvsVfw0+vnQ6XydrO/XxOsTZZe6f+X/N+8N+uTTVfc/1sx/6LWfj4mbXxJH4ewUcTN6Vheb/LoLSnIBBgD/8ph4QSnJCs9rj5XpykHh6LEgN88ZuCYhOTjDwpHN+Pm777NwL+gBLbjP3iGEXn8eBhPilL84cDatm4/UE6tSn3lY0TLdXGc/21fdK/llBjOzk0jePGg/V4qJZDtuOUDl7h32N4P1p298eLGq3YZw5xwQ9tb2g/Fgv6UOgPo/GeKir2l5NGdroIlZiu+anZhX45pk478lUwiX2e4/x2Jge14ibzRDv9+vO6GrVq1pT8O4iyno0uPuX+qVtkFi5je6Wdoj+v+ykb47P4doOfwcdW3mivn+3YfzGnSt4oYz6Jn0fwcW5tdBmU/EBBMAfybvfD4ivJOCzK/W2+RVglqfDevlmkzXpSmMYEUw7yotfy1RO/mWgd2ANAKgLSk/aXr2+Nj7B1H2+Tvqfn6b3u9ZneizZODnlu/Hxo9GtZjRe45UNSj5EeuzPemJ+q9gV78ly6IAj2lFzjD35GrzHOtu1EC3fLkZrJ2tdLWTfG9Ruq4uIPlS2e8+2STsoa8rasiFvWohtD2R3irA+9Ew3lMZR+Q6z1GB0ufELg2G9xnfj4GRtT3Mt6dzowbTfs6xF8nItFl0GJ3W0KgvHJV05sMbHYw4PZdOacOcR9v+lTEKCRslAfx8JvCnKwLIfEZNv0wNU6GH1OvvKj/PSHWLX29Xj2cDWdewkvPZ5t733218peZV85APn26To8iYyHrvzbisrXjcqmWxu7PV6y298fx3CbQD78xffL3PnrQbWCx40RmPVtrdm/hfdKDGysl9rm8nM+LI+H6syuPY89cPt2ScOtp9XR5tVx82vUFgBlzLGdX3viY5xXj6Vtb3NJhYN8gmbnvsb6PS+ONt7WDntvD/ZfzMCtbz+e5XNenvVj3v36CD7qfaj2eoaBxORGBcH4ndzx6VTaBEtikMn5qROQS6x8QsC/Iagt7Lu/lzb/fAicHA6KhsPmkR8CjA8F0gHHaVsOGsEXk6j0QcPN675ekw+3YQw9XpNPb7yOfXE8d3+co3fAM/eiffpAJczCuMsO2P3x6vbZryRptjJ/+mniFmxVnyqYe7vKSY5JR7fL9iHLz3DR+uvN02xnx15mz7jHSiHi4mbWh14Tzr7YLv+7Es0s2ZTWs7MvjiG6tfcMF7NmbTtjn+aibb3D62BvuwjT63Ob9i/VSrNdjGX7bLaVODXtX6KRI/g4x2GGgfC9WUEgTwHT06faZmg3JTHouD91Mi4bdT6IzQWc+xQQN9FA2ghFh+kA4Z8uy4aS2lafOI+HiFo+MGPKASf6kjZkGc9v3PHwEZ+26/wyXUclp3TG69hnixabt4oNuhgq61dsF37FltAm2ZPvzSRtPVfuEzj1+mm/DNviRxxX3TPxUMVBsD20NXPfRHPFNstr5fva94WFV38+pR/HRTPTfAIvrdtqO2en1Uxbz9245UN+RR/xAC5rWK8daTveMz4q392/PRAdR7tNHzee0phoKfnqbLijpnx8/HVaWyOPDdrf1+uC9TLJHVO9RZ1N8uyCse8Yxy6HI/g4x3rCoB6/GxYE9Qm7gZtz6qnv1xei3lxgh6burwFbENx//vUx//P9bbd/CPHnVymu1vt9j9iEwwE5aZuxuUf8mYPYo4Hn1QAFwWaKjPIEhr9U/LwLbn/JtDz11086t+vHn+Hjq/5rx3vS0s/hzT2F3A7nogMKgj1pClu3s4aIBbHYtgYoCDZTEGxbKCxk4oMG0AAaQANoAA2ggefUAAUBBQHfvUcDaAANoAE0gAbQABo4sAYoCA4cfKr856zyiStxRQNoAA2gATSABtZogIKAgoAnAmgADaABNIAG0AAaQAMH1gAFwYGDv6ZypC1PGtAAGkADaAANoAE08JwaoCCgIOCJABpAA2gADaABNIAG0MCBNUBBcODgU+U/Z5VPXIkrGkADaAANoAE0sEYDFxUESzvTDgIQgAAEIAABCEAAAhDYN4HTvs3HeghAAAIQgAAEIAABCEDgEgIUBJfQoy8EIAABCEAAAhCAAAR2ToCCYOcBxHwIQAACEIAABCAAAQhcQoCC4BJ69IUABCAAAQhAAAIQgMDOCVAQ7DyAmA8BCEAAAhCAAAQgAIFLCFAQXEKPvhCAAAQgAAEIQAACENg5AQqCnQcQ8yEAAQhAAAIQgAAEIHAJgbMLgv/+eh1Op9fhx7+XTH+Evr+H99NpOJ1Ow/vf/w0/vsjrI/iOjxCAgCHw93vMBafT+/D73x/D65gbQn6Q/1//+i92+f2tvBfb64F032+/y508fiXP6D5ffgxplh3npOjP+6C8Lxye+FXRhd9/y15zyvGtgei00xrRuvLDdNq17fOD3O462CDrqDrLxu2v2rzmzQVrYwtxWuPSpO0RfJw47d5YwMD16F6eVRDsXkhdJNe+GZKvTdyB3/vf156H8SAAgc0TCAf21mEtHubHA254rQ5kMefm63SgSzkkHejT4Se8rw7IerxB9xkGO1663lVOygc65e/mg3+5gfFBnOjHxFfrYBrfMnOvndaIbVf6h1ftdm377Ai3vJLzSbsg2Lb9F7NZsDa2EKeL/DyCj3OAFjCYG8LfX10QyGJLT7PsQdcPznUgEJKP5RQY7mrzJZAQgMB1CIRDnBzozIjpANbMC/rwFzeCchA2m7sZUx18Yp+Sh3yfXeWkyOJ1+PF3+ISlcDCuP+lFiFM56Kr4+n2mqRe3HzXbDYPWiJm3oz/TzhQO9whIWkOhkLZ2uLk3a7+z85zLhWvD8tE6OmfSO/c5go9zSBcymBvG3z+jIEibShAUXxnyOGvXLgGHEoGCoAaK9yDw/ARCIq8VBK33RyL6cDb4tnFzqByMTRFgn/jG/J0/cdhpTnIHu+cXjy8aVUw9CxN7RabXbqmumu069ikT7vHSHnjdjDuw31m8/tLH2YywnTgZs9ZeHMHHOSZdBnOdp/dXFwQyBAWBkJj7SUEwR4j7EDgMAX8YiY77DdrRiEm//XQ/FgiTJ+XqsJiHS++FT3fLU+Z0c5cPKa68GWZMm33hn+SqGAddaQ04zWSXOu1M0Rk6+LbjIO12HfuyAfd50SsI9mD/xZS6a2M7cbrIzyP4OAeoy2Cu8/Q+BcGUyZXfoSC4MlCGg8B+CdQKgl5Sj/fcVwz9GJOD23jwV58ADHGcdlFBQbAHSaW4lq+VqYLAa8jFO3vXazerq3GUZruOfdmA+7zoFQTtT9i2Y//FlHyczYBP4ucRfDRxq1x0GVTaz7xFQTAD6PLbFASXM2QECDwJAX+YGsbvauvDu7gaD/rlEC9vp8N9+YqQfeKZnv75TwBsm2HwBQIFQaa76Rf2oKuf9Lp9JmqnaKQ41WnnDhcTzcggnXZt+6TzfX5aO9ycO7DfWbz+0vnoB7B8tI58yw1fH8HHOfwzDOa6+/sUBJ7I1a9dAubfEFydMANCYDcEKgWB3ZxHT2KirxQD8bbewNVT4kG/dkTcePGwp75iEmwoT55d361eXnkz3Kqb2i5zSDeHfhv7EE/9W6rKGL12LV2V3ulVu13bPj/Gba+raypPuX37s6nnvphZG1uJ07nuxX5H8HEO0AyDue7+PgWBJ3L165B87Ma+y8336lwYEAIHJDApCNIBzR/G04Fd/x2C8Fo98Y0bwXhfPl3Q76m/a5DHjgdIGfMJctKVN8O9qDEe9mN8bQzl14HG3wCo/+H6hFM6EE/aBQBaQ6Kr8SGW+dSp0S4M0bbvfoR9QeCvW35uxf6LSVVjbvWyhThd5OcRfJwDNGEw16F//+yCoD8sdwsBCoLCglcQODiBSUGwDR7hcJALh22YhBUQgAAEIHBHAhQEN4ddnsbwl4pvDpsJILBtAvkpvXra/1CL0ycU4WkxBcFDA8HkEIAABB5KgILgofiZHAIQgAAEIAABCEAAAo8lQEHwWP7MDgEIQAACEIAABCAAgYcSoCB4KH4mhwAEIAABCEAAAhCAwGMJUBA8lj+zQwACEIAABCAAAQhA4KEEKAgeip/JIQABCEAAAhCAAAQg8FgCFASP5c/sEIAABCAAAQhAAAIQeCgBCoKH4mdyCEAAAhCAAAQgAAEIPJZAtSD4/Pwc+B8GaAANoAE0gAbQABpAA2hgvxpYWmZQEFD8UPyhATSABtAAGkADaAANPKEGKAieMKhU6Put0IkdsUMDaAANoAE0gAburQEKAgoCKn00gAbQABpAA2gADaCBA2uAguDAwb939cl8PPFAA2gADaABNIAG0MD2NEBBQEHAEwE0gAbQABpAA2gADaCBA2uAguDAwadC316FTkyICRpAA2gADaABNHBvDVAQUBDwRAANoAE0gAbQABpAA2jgwBq4XUHwz8fwcjoNp/H/l+9/EFpXaD+Ht5HV268/w8drYvf267wq+c/3l8T+9WP48/k55Oswx9efKhZlrhirsX2qTItNKY4vw8c/yZ6fX0tsT6e34WfFt9DGxr09Xq8S1nOZ8Xoac/eKjT1/hXWws/gabWuOJ30qP12fnu3Fvl6sPgfNwmpD++Vsz7FJ/G2/it2h/a+3vHZD7Gt9oqaMljq2GxZeL0oXC/XX00tb68XX2EbP1fNX2677ZK4pLia+nzZW/l7Lfh3fVp/YRnHXfbSO4hzaL9Unzx998/EonHI75et+3luyJtb42tKp1X1trQgzrc1WOx9f6Wt+VuPWts/0zbGs5AOtdZPX2yzP0p+ZJ+wld9agnr+2LiIj7fNpOOm1r/vHffvO9ucYrtGvbVvi1tovPofPPfupc19jD5N1sYjFFZjLfHf7aeI3r9EbFQTjQpIFFAPTEd0eQV/d5pCcLaMg0tamMScoc1gLopBYfKbYyGEjtJPXn+5eSgYVEYV4qiQaF5O6DrbJAitjS3KpjNdjaebSG1jwQ/GKwlfXoV/2uSTCrr/RDtlU1Vjh/cZ47TicZ186iMvc01hln5y/JgbB1soGKzGZ11RgoOJUGS8fbHTcTbtke5rLsoh9c2ysj8aP6KOyo6cTudfReo5VHFdv8D1/teZGXWufG1q3PoYxJKZFi9kesT3wy2PbeXPbyFgV9aaPs89oxHKO4wkHHWux5Ql+Gi0ZbXZi0PTb8muObZi7ebQNrXY+vjV7qnHr2FcbQ+m25AOruajhUY9Nf8/VX+iXc4Dj1LA3r4GL72s/LTc9R/C/7F+u3UPtv5yXyU9Rc408u1s/Q4yVTx0fF7O4WHeXx03rc/510GzZd4yfDV9uVBA4x8cEVhaXu98wbt7hZxonCLgEL/geknBJ1ut8jcHPhwvbd/aeJOqlycAstpQ4w8Em2G9ivnS8jh4mY+a2NmH3fNS6Motk1Onbr2kslo6nx7avl9k3mUcxM5vyZ2dTqx02wjhf3+KnUOs1pecaD52vH8PP8CmU0thivUb7JFk71lpLynfL0uq5d2/CMxa9L8Pb15fOgUT561gavYwF9FTrNhH37Ovdm2o92PU2vIVP5xR3M4bnpzZFo5/Y7mX4+BU+yZVYLOdq5sxrcEv97XpLDzdsfl3nQ1unVmN2Xh3DXrtkSy2+tXlrcau168Q1xN/ng/ie/vRY4ml96rJcqD/LQua500+Tf8ZPd2TP62hZr/2H2t+xcammtS4/zV5iY7B3PwuPsD7q56mlLMpYltFu3ne6r9l9l4IgbkSNYNSMOuZ7LqHfrCBwyd0llxir8bARk8H4NabwtRFzuFf9dKLUsbMLzX6s3htPj2Fftxe1T2qiufRVp3oiCGNrf8tc9VjIWJoXxGMAABbwSURBVOHnVQ7Viq2MN0nArQ22d1DVfWKcxJ8ev06Sc3MJJ2tr0NXL8PFdfdWocWg1evFJSs0V2ylGLf2JPdOfU62LzcYGpeU4hrLBf2pW10vSUbEvcH4bPuRre521M7VZ4jCNlaynlg1hLOuXHmPKovjaOTh6Nju6Niwma0I4L/zZ0anERWLZis9cO7nf6i/jV+PWsc/06+SDwOvl+0f+6qr+lHEpS9POHDSt/qKPam1L/pvaujA+a3QZtKALgIXa0HF5qP1rfK22TbEozG1sdAz27afSjsnp6v3xoc4SFprLHl/btakZlNc3LwiiEXHhP+emcz1hhM3bPsEKi7EItQRtyZyRuz+QxcTX+b6mS4w6Acp3CSf2NBeaPyS5w3fst9y/rCOdyFWyM7b6rz5FvyzbyND5W7j6WLiE2RpP2VPGSnFbbJ8bO/bTT3Dj/XoMW4zC++mwqg+IS/XkfFc+Wo2ldmWjrc2V3jMFlY+B0pNhtlIv5d9A6LwT5k/X0faqlmr+jr51DvbBVlsQqKf4yievi9p1NY5hjNFew0XFo/7ktsJ80kczWqqLfbSrstT+L33d1OlUL/X4zLRbEl9ta9SUilvTvmmcApNaPkisSp70fsyyrOq8pj/HItpe5q2tiWu+F/3Qa9+z05zltWnzWPsvZ5FiUvZy54/4vHQfze2nWrvc1muM2fIvjL2UxTXseNQYtTVYt+W2BUFcROkfnxbx1Q3ZhnAeaVsImk2KISGfyy0mPV8QyMI1yW30Ob5n5/cx8RtEs0gY5wntyyFpynYyntjX+TlJ5vKUXyf4Sf9KQuj6O42FZVEZbzJn8Tf6ucK+2H58evb2tTzNsrx0InP26I1ZHTSmya/YaP2T99O4ra+nWI2ltlqv1l4ZM/xUtvuDjbbdMW2Pp8d2r5XWQ3+xr6Yj+TTA+Ovsqffzxa/Xj4uP86vOXj/tD/3L2qxyiHYW/+KYyvdw3e6nDpYLbWvZvJ33HXMXx9V2xv6KkxovcNV5rsp55F9vtyC+Pi5Ve+r2GV9Dv5yL1DqUT5f0npH1s4BltOcM/UW/3Pje12tfB78yA/kFCoqdny9yKOvP8HyE/d6+1deJt+RCyXtam1MfQ169c5xW++Vyv4qNyelm3HNZ1Oba+nt2vddifLuCYEwQ4WngvNC2DvIe9vlDRNrAy6JdZ4M9rLm+MTYlwcW2+im0WTClr9n4ZpOkPySVcUSIZrzGnNI2/8ybVBgvLeb2Ypc57aKf93caizy/SjLzsTnPPj1XieN0MeeDh4uncAnrLvmaivJ1X3lK8/XWbrEtcfbxzPZNYpu4pLEdaxNfiV99fM2p+TqzSf5oBvF1PhjU/Y0+5jbyD+PL2pF5re9Wbzoe0n7RT2GhcqmxXw5vjbU44S/j6XjEsTuHId12T69z3EVDWnPy3pqfbZ3addCep9luLr417pO4te3TWos2qK/qiJ5iLgv60FoXvcyxvER/0Te/XtbE5Yy2jt1kjSveidfc+riz/co+Hds1r22+Srlv6X423+6MmFzBp6n/9Zzu253HYis+rrGjnZuEyY0KgjRxTDayad0k4GtgbL2tS+jjE6VzF5/ZfFyiNwlQkv4kPi7J6U1Bv570K5ztQuuM1xnD+OGeck4OPHkcx1L7qF/n9sXmtDBc//hEWx0CF43ReCIb53Tj6/HC67wph3blqZv1N91Lh2q30GN8Sj9Z7ObJfNP3wMKN12jrY2N+E5Oxwflr9GPnKj6epxdjgzz1zDxLnM0a6PlrbJV/BzM9IFitj+0k98X4Kv0s5FlYFLtDLM37zr4S6+mTz9jPs4j9p/6YcRr2bruN1dXcp5nzvtjxTAz0+l0aj047M3aL/SRuHftaY+hP6mIbvU7TeOlhix3bsOz4kb62V7RV9Kfnmep0PhZ2Paxvn3Jn2ludb5qVjqt+/8z9YL2dl/rZ7m/y31P62YmriaX+JPYRWmzH6HK9uHXWW6sjk9sUBHHi6VPJ3tPGy52/Jdh7jO2Cd82CQA5G+cmQS9L5/TFm+dCQEqd5ijQZS+JcxpRYhg3Axrw+Xmg/bVuYx41EbGzYJjbm+YwGy2HMjDUZU+acxkI2wTRPGS+9P/VdDt9il/ycs09YTNrHBTtu0mK3HDjjPcs2z2OSX2ojRebkQC9tDTuJ77TAqPWP7432GRvcmGJD0oqyPcc3xEK9rwojYWTGF9sn+qzFxm0CzjZhn22MG6ZwULFXc9b0q7WWx/KHCTWG+CXzl8JQdJl+xnHH2GveuZ/6tM/er7CIvuv3A/Pio/Fr0tbalWK5pfesflp6WW63Gs/oVArFpJES62le0/HQ7bQNOr5pDZR45HbVWDTsq7YNcUrtjR2x7ah146MaW33qr/1ZpT89j9Jb9s+ti6u/r+dXeVTrPcZBcq38FCa6/yPsvwKf4p/WV4izut6rn8Zuyd2yhzkf5SFLjLHy/QqMr67btTY5DmatV8a6TUFQmejhYDZvU12kcwFscY2JWiW6VrvNvP/Px/C20z9e9/OrPlBt6UA0Z8vP4W1PGtFreMd6+fP9Lf+Bv82sP82W1+oPN86tIe6jYTSABp5DAxQEm9n8yhOYW/yl4q0v2P0eknZ8qP71lv+R7db14e3br17+DB9f018P9z5x/RybKnEkjmgADexRAxQEmykIWEB7XEDYjG7RABpAA2gADaCBvWuAgoCCgI/H0QAaQANoAA2gATSABg6sAQqCAwd/79Us9vNEBg2gATSABtAAGkADl2uAgoCCgCcCaAANoAE0gAbQABpAAwfWAAXBgYNPRX15RQ1DGKIBNIAG0AAaQAN71wAFAQUBTwTQABpAA2gADaABNIAGDqwBCoIDB3/v1Sz280QGDaABNIAG0AAaQAOXa+CigmBpZ9pBAAIQgAAEIAABCEAAAvsmcNq3+VgPAQhAAAIQgAAEIAABCFxCgILgEnr0hQAEIAABCEAAAhCAwM4JUBDsPICYDwEIQAACEIAABCAAgUsIUBBcQo++EIAABCAAAQhAAAIQ2DkBCoKdBxDzIQABCEAAAhCAAAQgcAkBCoJL6NEXAhCAAAQgAAEIQAACOydAQbDzAGI+BCAAAQhAAAIQgAAELiGwviD498fwejoNp/H/978vmf4IfX8P75nVf8OPL4kd3I4Qe3w8JgG95hOB399Kznz96z+FpeSE0+l1+PFvufXfX685z8Z8++13uanzsH5/KHP78QZ978uPIVlR5t9dTooM3gdFpfB54ldFS1Yv9fjWQCiNZB2M7Zq6cuN02rXtc2Pc8DLYYNeZm2zj9jtr118uWBtbiNN6x1SPI/io3K2+XMCg2q/x5sqCYNw8xiSSBHW8hNxg2Xg7JF+buAO33W2+De94GwIQ8ATcmv/7fTjlQ3s6jMn6jzlU7oV2p5JPw736oUaPkXJyajfm5+p4ut0wmHmHdC02eW82eZ0PdIXXJu28slGxSJRDvNFLP77FjF67lq5K7/Sq3a5tnx/jdtdR26fW2gnzbtv+i8ksWBtbiNNFfh7BxzlACxjMDeHvrywIbPe08I6VkC2BJVch+VAQLCFFGwg8B4Hpmtd+lYO+PZwNMcFLrkj3qof02K7k3bK5+3n1GO6eOUzurCCItr8OP/4On1YXDprxs74u2gkehpjKw6VOfI1elrYbhqKrpI9cnJrxOu2MffeISNJ7KL4tJzf3Zu13dp5zuXBtWD5aR+dMeuc+R/BxDulCBnPD+PtnFgTjwpMnFX5UrhUBl4D3+DROecNLCEBgjsB0zZcedvPVh64hJnk54KZ28tVM/clBbKdzb+7n5015Oh7k3CHIFh87KwgEpvdJ3n/an7rAC04uj29G4pnF67EIDTqq6ir3Ti+a7Tr2uSFufWkPvG62HdjvLF5/6eNsRthOnIxZay+O4OMcky6Duc7T+2cWBGmguJm5p9/TKY7+jt+kd7r5Hj2M+A+BxQSmaz50TfnyZA9drff1QW18iCCHNVNEhIFzQZA2+vz1pDjG+NWJ3GZ0ojJ+9dOIxT4/oOGVN8MHeLByylQkljipgmAmvnmiTru2rnLv+KLdrmOfHeLmV72CYA/2Xwyouza2E6eL/DyCj3OAugzmOk/vX1QQpKdMve/qTSc83jvTw0FIViWpH48IHkPguQlM17z2txxI1IEuNIjJXb4ypHu4e+FQ13qSG8cY/wHzl/fh/cuYa/zG4ebaZU7yPjlkz3fZebLrWbj4Zha9dj1d5QHGArSqv459uv8dXvcKgvYnbNux/2JEPs5mwCfx8wg+mrhVLroMKu1n3rqsIIhPGzjc9hlPDwe73Hz7TnIXAhDIBKZrPt8KL+QpbUzmugBIG3X+rrbupBO/fi2fMOgDWu6n7dCvlQ1j213mJMchu/3EL+xBN8RU9t9+fAuSTjvHsxSupXd81WnXts+NceNLa4ebbAf2O4vXXzof/QCWj9aRb7nh6yP4OId/hsFcd39/ZUGQhCMfSQdRme+2+tG5Hv/hl970+coQsoDAcxOwh654sJLf/CNf/4nXrgCIyX084LmntTHX5jH0Bq7HsE/+7IFOt0s5SPJ4iEUYf3efWl55M9yDJk1MpbCMhvfjW3zrtWvpqvROr9rt2vb5MW57bQ+8fq7t2+8tXn09sza2EqfVfukOR/BR+1t7PcOg1qX33sqCQJ4sye/Utgfd3kTHvReSj+W0y833uAHEcwisJFBf8/kfCJun+elwIvf0pwNx05a/+WL6yFeIxjycCwX3/uQ38Ki53Hi7zElX3gxXBvlhzUOskl7sviK/dSje0/GdcGrrQL4GHMdQupocsOOYFf1JwRt16+27HzJvr79u+RksbPO9n/0Xz1SNuY3H7v08go9zQpgwmOvQv7++IOiPx90JgfrhYHdP4yZ+8QYEIFAnMF3z9XbbeTccDshJ24kHlkAAAhC4NwEKgpsTL09j3v9OH9eGpy9svjcHzwQQeBABveYfZMLiaclJi1HREAIQgMATE6AgeOLg4hoEIAABCEAAAhCAAATmCFAQzBHiPgQgAAEIQAACEIAABJ6YAAXBEwcX1yAAAQhAAAIQgAAEIDBHgIJgjhD3IQABCEAAAhCAAAQg8MQEKAieOLi4BgEIQAACEIAABCAAgTkCFARzhLgPAQhAAAIQgAAEIACBJyZAQfDEwcU1CEAAAhCAAAQgAAEIzBGoFgSfn58D/8MADaABNIAG0AAaQANoAA3sVwNzhYDcpyCg+KH4QwNoAA2gATSABtAAGnhCDciBf+4nBcETBp9Kfr+VPLEjdmgADaABNIAG0MC1NDBXCMh9CgIKAp4IoAE0gAbQABpAA2gADTyhBuTAP/eTguAJg3+tqpJxeEKBBtAAGkADaAANoIH9amCuEJD7FAQUBDwRQANoAA2gATSABtAAGnhCDciBf+4nBcETBp9Kfr+VPLEjdmgADaABNIAG0MC1NDBXCMh9CgIKAp4IoAE0gAbQABpAA2gADTyhBuTAP/fz/ILgn4/h5XQaTl9/IqCugH4Ob4HT6TS8/fozfLzKa6rfa1W/jIOWtqUBveZtbH5+PQ0v3/+knCk5dMwPIUeE/8P9P99f4mt5L/18GT7+Gcczfd+GnyEHmffSWPPj7TgnRX9H37s52MZgW1pZb1vQ0EQP0f+iu9Prx/CnyaTTTmuot7d32rXtW+/rubEy66zGYeP2n+t37rdgbWwhTtneWozm3juCj1dgEBgv/e/sgiCLqZc05pw5xP2QfNVG/vk5BHZvv+6XHC9adIeIEbFAI9fUwHTNB76SM3NB4NfWr7fhdKofcGPfnGvTgU5ySCwe8j3lx+LxdpiT8oGuzutZ9RxjLYd9E99U2Im2rF6UJj577bSubDvLs92ubZ+24bavZ9fZ57btt6zPYLVgbWwhThf5eQQf/f7grxcwEMa3LQhiIhqfUtQ2Im/4oa9D8qEgEGHy84wEf+j1s0defs2PT+G//oxFgRza7FpIbeSQb+7FxK8OviH/zubdFePt7SFF3H9eho9f4VNqxeUA6yQcdot+9MHWaU4XC0Y/S9t9pk+pxuLDzGvG67QzB+97rOMl60w+SSu60Ydj4+fd7b8Co4VrY9d+HsHHuVy2kIHsIzcsCFISevn+kb4KM7sxXUHkc3A2fd8l4L1tvptme3Rt4b8kvG39nK55sc9uxCp+IcHLk1+35nyfcIDJ+Td+zagcbmSezxXjhT5hjmox4mzJ42/hfXcw3ZRtN+Hji7x0HQsEzyJe2wdRkU+vnddMPHQs0FZu17HvJjzU+nHj+zVjtNH0czv2G3udb4vu+TibMZ7EzyP4aOJW0XuXQWl/s4IgVtPxqUwqDOafVBWjFgl5DsDu7k8PB7vcfHfH/ei6w//H5Zvpmhdb6gcVv0Gr2FUSfsrB5bAXxrR5eN14wbZd5qQKG+H8nD/TnlsKtxTnWBDkQ/moncimaCTz6LTTT8pje9923APa7Tr23Xn/qK+zxGYP9ud4ncutuza2E6eL/DyCj3Px7zIo+8htCoI4uTxJSqKyG1Ex4KJAz0HY1f3AySbmXW6+u2KODll/j9TAdM1LPKoHlU5Sj4cX9yns5D1/cFs5XrBtlzmp46fwfq6fvtBTBYFnEa/tvhNZ9NoFHelPqbyuZA9otuvYJ33v9LO6zmTuHdh/sW59nMX3+HM7cbrIzyP4aOJW2dO6DEr7mxQEcSNyvxEj/rYDt2FdFOQ5ALu7Pz0c7HLz3R33shjQIyzuq4Hpmpf5aweVyQE/rzW/cY9xbB5o0v3V41EQ7OY35Vn9BJ3pB3SqAGgd5uP34hvt3OEi6kgXCKLLTru2fffNQdYON/cO7Jd8cfZP56Mfx/LROnKsJOZb/HkEH+e4zzCQuN+kIJDB088kIj4hmFtA08NBWIzlY9+5/ty3uoMHPLaugemal5jZjTj5UXuv5Fh1eMubgx4/FQ06D68fj08IJD5b/2kO6ebQn3Qg/+A4aEBrovjVa5f29LQ32Xalf9Bsu13bvvuu2fYa2If9lvcZ7GYOiluJ00V+HsHHnPMbGphhIHwpCOZA3u2+3rxTUEOyoiBoCPxucWF+SRb8vLYWpmteGE8PKungVc0HvWQf742/6c08xT1vvF3mpB6fJ84jIVar/g7BhFM60McxjHbkN/BMf4PgRLdaf+4bAm37rr3O2uN5e/21+ZsdG7Rf8sXZP6sxtw8XthCns/0L6/sIPs7lsQmD+pq4Q0FQn/iiAM85v8v708NBWIjVA8Au/UMHaB4NWA1M17y9vz1e5KTtxWTrmsE+NIMG9qEBCoLNHK7L0xj+UvE+Fg9JjjhdpgG95rfOMn2iEJ4W85Bi67HCvsvWJfzgd0wNUBBspiA4pgBJPMQdDaABNIAG0AAaQAOP1QAFAQXBbn5zBsnisckC/vBHA2gADaABNPCcGqAgoCCgIEADaAANoAE0gAbQABo4sAYoCA4cfKr856zyiStxRQNoAA2gATSABtZogIKAgoAnAmgADaABNIAG0AAaQAMH1gAFwYGDv6ZypC1PGtAAGkADaAANoAE08JwaoCCgIOCJABpAA2gADaABNIAG0MCBNXBRQbC0M+0gAAEIQAACEIAABCAAgX0TOO3bfKyHAAQgAAEIQAACEIAABC4hQEFwCT36QgACEIAABCAAAQhAYOcEKAh2HkDMhwAEIAABCEAAAhCAwCUEKAguoUdfCEAAAhCAAAQgAAEI7JwABcHOA4j5EIAABCAAAQhAAAIQuIQABcEl9OgLAQhAAAIQgAAEIACBnROgINh5ADEfAhCAAAQgAAEIQAAClxCgILiEHn0hAAEIQAACEIAABCCwcwIUBDsPIOZDAAIQgAAEIAABCEDgEgL/DztFv5eMY9ADAAAAAElFTkSuQmCC)
<!-- #endregion -->

<!-- #region id="Ms964u54imRX" -->
## Matrix factorization model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HK5IyEuTiagt" executionInfo={"status": "ok", "timestamp": 1609587143019, "user_tz": -330, "elapsed": 1265, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5692ab74-c1fc-47e7-f3ad-e8a94bc22035"
# store the number of visitors and items in a variable
n_users = events.visitorid.nunique()
n_items = events.itemid.nunique()

# set the number of latent factors
n_latent_factors = 5

# import the required layers
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten

# create input layer for items
item_input = Input(shape=[1],name='Items')

# create embedding layer for items
item_embed = Embedding(n_items,
                       n_latent_factors,
                       name='ItemsEmbedding')(item_input)
item_vec = Flatten(name='ItemsFlatten')(item_embed)

# create the input and embedding layer for users also
user_input = Input(shape=[1],name='Users')
user_embed = Embedding(n_users,
                       n_latent_factors, 
                       name='UsersEmbedding')(user_input)
user_vec = Flatten(name='UsersFlatten')(user_embed)

# create a layer for the dot product of both vector space representations
dot_prod = keras.layers.dot([item_vec, user_vec],axes=[1,1],
                             name='DotProduct')

# build and compile the model
model = keras.Model([item_input, user_input], dot_prod)
model.compile('adam', 'mse')
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 422} id="yuNltIrJn_Fa" executionInfo={"status": "ok", "timestamp": 1609586509274, "user_tz": -330, "elapsed": 1502, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45d1a810-1b16-48f6-de87-366995d334ec"
keras.utils.plot_model(model, 
                       to_file='model.png', 
                       show_shapes=True, 
                       show_layer_names=True)
from IPython import display
display.display(display.Image('model.png'))
```

```python id="YzbLeB9OoNdd"
# train and evaluate the model
model.fit([train.visitorid.values, train.itemid.values], train.event.values, epochs=50)
score = model.evaluate([test.visitorid, test.itemid], test.event)
print('mean squared error:', score)
```

<!-- #region id="7T9jBk4m4oiV" -->
## Neural net model
<!-- #endregion -->

```python id="OcBOy0QcoXeV"
n_lf_visitor = 5
n_lf_item = 5

item_input = Input(shape=[1],name='Items')
item_embed = Embedding(n_items + 1,
                           n_lf_visitor, 
                           name='ItemsEmbedding')(item_input)
item_vec = Flatten(name='ItemsFlatten')(item_embed)

visitor_input = Input(shape=[1],name='Visitors')
visitor_embed = Embedding(n_visitors + 1, 
                              n_lf_item,
                              name='VisitorsEmbedding')(visitor_input)
visitor_vec = Flatten(name='VisitorsFlatten')(visitor_embed)

concat = keras.layers.concatenate([item_vec, visitor_vec], name='Concat')
fc_1 = Dense(80,name='FC-1')(concat)
fc_2 = Dense(40,name='FC-2')(fc_1)
fc_3 = Dense(20,name='FC-3', activation='relu')(fc_2)

output = Dense(1, activation='relu',name='Output')(fc_3)

optimizer = keras.optimizers.Adam(lr=0.001)
model = keras.Model([item_input, visitor_input], output)
model.compile(optimizer=optimizer,loss= 'mse')

model.fit([train.visitorid, train.itemid], train.event, epochs=50)
score = model.evaluate([test.visitorid, test.itemid], test.event)
print('mean squared error:', score)
```

<!-- #region id="XwiMEOwi55mz" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="CrTjO7i46Htd" executionInfo={"status": "ok", "timestamp": 1609588005823, "user_tz": -330, "elapsed": 58544, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="42de1144-29fd-4c04-e698-ae3b9fb1352b"
user_activity_count = dict()
for row in events.itertuples():
    if row.visitorid not in user_activity_count:
        user_activity_count[row.visitorid] = {'view':0 , 'addtocart':0, 'transaction':0};
    if row.event == 'addtocart':
        user_activity_count[row.visitorid]['addtocart'] += 1 
    elif row.event == 'transaction':
        user_activity_count[row.visitorid]['transaction'] += 1
    elif row.event == 'view':
        user_activity_count[row.visitorid]['view'] += 1 

d = pd.DataFrame(user_activity_count)
dataframe = d.transpose()

# Activity range
dataframe['activity'] = dataframe['view'] + dataframe['addtocart'] + dataframe['transaction']

# removing users with only a single view
cleaned_data = dataframe[dataframe['activity']!=1]

cleaned_data.head()
```

<!-- #region id="X-lA1bAG7Fpq" -->
Since the data is very sparse, data cleaning is required to reduce the inherent noise. Steps performed

- Found activity per item basis. Activity is view / addtocart / transaction
- Removed items with just a single view/activity (confirmed that, addtocard ones have both view+addtocart)
- Removed users with no activity
- Gave new itemId and userId to all users and items with some event attached and not removed in above steps.
<!-- #endregion -->

<!-- #region id="Z5lDAhMl7rPw" -->
---
<!-- #endregion -->
