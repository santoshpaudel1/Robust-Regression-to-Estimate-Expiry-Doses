import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from datetime import timedelta
from datetime import datetime
import time
import json

import os

from shutil import copyfile

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge
from sklearn import preprocessing

from STE_script import rmvl_cutoff, inv_cutoff, prediction_cutoff, output_dir, input_dir
from STE_Hybrid_utilities import box_plot_individuals, box_plot_by_group, box_plot_group_model


data = {"df_medfullnames": pd.read_csv(input_dir + "23sites-medfulnames.csv", dtype={'medid': str}),
        "df_dispensingdevicenames": pd.read_csv(input_dir + "23sites-devicenames.csv")}
ts = datetime.now().strftime("%Y%m%d-%I%M")
# Check to make sure we have one medfullname per medid-clientkey
# pd.DataFrame(data["df_medfullnames"].groupby(['skclientkey', 'medid']).count()).to_csv("grouped_medfullnames.csv")
# data["df_medfullnames"].drop_duplicates(['skclientkey', 'medid']).to_csv("dup_dropped_medfullnames.csv", index=False)
# print("you can quit")
# quit()


def set_log_file(output_dir):
    #ts = datetime.now().strftime("%Y%m%d-%I%M")
    logging_dir = output_dir+ "Logging/"
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging.basicConfig(filename= logging_dir+"Log_STE_Estimator_" + str(ts) + '.log', level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def log_warning(log_text):
    ts = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    logging.warning(str(ts) + '\t' + log_text)


def log_debug(log_text):
    ts = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    logging.debug(str(ts) + '\t' + log_text)


def log_empty_df(function_name, df):
    if df.empty:
        log_debug('Something is wrong!!! ' + function_name + ' outcome is empty')
        return True
    return False


def log_pair_count(setname, data, first_ymints, month_cnt, clients_list):
    if type(first_ymints) == int:
        first_ymints = {first_ymints}
    for CK in clients_list:
        for first_ymint in first_ymints:
            trng_pairs = get_pair_count(data, first_ymint, month_cnt)
            count = trng_pairs.loc[trng_pairs['skclientkey'] == CK]['count'].count()
            log_debug(str(CK) + '- ' + setname + ' pairs starts ' + str(first_ymint) + ': ' + str(count))


def log_duplicates(log_text, df, description_col, key_col):
    ts = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    test = df.groupby(['skclientkey', key_col]).count().reset_index().rename({0: 'count'})
    if ~test.loc[test[description_col] > 1].shape[1] > 0:
        test.loc[test[description_col] > 1].to_csv('Duplicated rows when adding ' + description_col + str(ts) + '.csv')
        log_debug(log_text + 'Duplicated rows when adding ' + description_col + 'to the test/training dataset')


def get_pair_count(data, first_ymint, month_cnt):
    # first_ymint = yyyymm e.g. 202002
    # first_year = yyyy00 e.g. 202000
    first_year = round(first_ymint / 100) * 100

    first_month = first_ymint - first_year

    if ((first_month + month_cnt) - 1) <= 12:
        last_month = (first_month + month_cnt) - 1
    else:
        last_month = (first_month + month_cnt) - 1 - month_cnt + 100

    if first_month == last_month:
        first_month += 100
        last_month = first_month

    Pairs = data.loc[(data['month'] >= first_month) &
                     (data['month'] <= last_month)].groupby(
        ['hdp_shardid', 'skclientkey', 'medid', 'device_naturalkey']).size().reset_index().rename(columns={0: 'count'})

    # to let pairs show up when looking at one month
    min_expected_count = min(1, (last_month - first_month))

    Pairs = Pairs.loc[Pairs['count'] > min_expected_count]

    return Pairs



def read_inputs(client_input_dir, read_json=True):
    print("Working on reading inputs for " + client_input_dir[5:-1] + "...")
    time1 = time.perf_counter()

    df_tranx = pd.DataFrame()
    unitcost = pd.DataFrame()
    usage = pd.DataFrame()
    inv_eed_ssflag = pd.DataFrame()

    with open(client_input_dir + 'STEinputFiles.json') as f:
        inputs = json.load(f)

    if read_json:
        os.chdir(client_input_dir)
        # Outdate and Destock Transactions
        df_tranx = pd.read_csv(inputs['df_tranx'], parse_dates=['transactionlocaldatetime'], dtype={'medid': str})
        empty = log_empty_df('read_inputs', df_tranx)
        if not empty:
            log_debug("df_tranx loaded successfully")

        # Unit cost table and medid-medkey mapping from MKP_ES.item
        unitcost = pd.read_csv(inputs['unitcost'], dtype={'medid': str}).rename(columns={'clientkey': 'skclientkey'})
        empty = log_empty_df('read_inputs', unitcost)
        if not empty:
            log_debug("unitcost loaded successfully")

        # usage rate for both sites
        usage = pd.read_csv(inputs['usage'], dtype={'medid': str}).rename(
            columns={'clientkey': 'skclientkey', 'devicename': 'dispensingdevicename'})
        empty = log_empty_df('read_inputs', usage)
        if not empty:
            log_debug("usage loaded successfully")

        date_parser = lambda c: pd.to_datetime(c, format='%Y%m%d', errors='coerce')

        # Inventory, EED, and Standard Stock Flag (pocket monthly snapshot)
        inv_eed_ssflag = pd.read_csv(inputs['inv_eed_ssflag'],
                                     parse_dates=['lastday_of_month', 'earliestnextexpirationdate'],
                                     date_parser=date_parser, dtype={'medid': str})
        empty = log_empty_df('read_inputs', inv_eed_ssflag)
        if not empty:
            log_debug("inv_eed_ssflag loaded successfully")

        # Add monthfirstday column and
        inv_eed_ssflag['monthfirstday'] = (inv_eed_ssflag['lastday_of_month'] + timedelta(days=1)).astype('datetime64',
                                                                                                          copy=False)

        # Fix medids by dropping 0 from the beginning of the numeric IDs
        # inv_eed_ssflag['medid'] = inv_eed_ssflag['medid'].apply(pd.to_numeric, errors='ignore').astype(str)

    global data
    data["df_tranx"] = df_tranx
    data["unitcost"] = unitcost
    data["usage"] = usage
    data["inv_eed_ssflag"] = inv_eed_ssflag

    time2 = time.perf_counter()
    log_debug("All data loaded into global dictionary successfully")
    print(f"Completed loading data for " + client_input_dir[5:-1] + f" in {time2 - time1:0.2f} seconds.")

    return get_training_months(data["usage"])


def remove_non_drug_items(df):
    drug_item_flag = 'meditemflag'
    exc_index = pd.Series()

    if drug_item_flag in df.columns:
        exc_index = ~df[drug_item_flag].astype(bool)
        log_debug(str(sum(exc_index)) + ' non-drug items are excluded.')
    else:
        log_debug(drug_item_flag + ' column is missed in input prepared dataset (all)')

    return exc_index


def remove_insulins(df):
    med_name_column = 'medfulname'
    med_name_column_alt = 'medfullname'
    insulin_key = 'insuli'
    patient_key = 'patient'

    exc_index = pd.Series()

    if med_name_column in df.columns:
        medname = df[[med_name_column, 'medid']].copy().fillna('No med full name')
        exc_index = (medname[med_name_column].str.lower().str.contains(insulin_key)) | (
            medname[med_name_column].str.lower().str.contains(patient_key))
        log_debug(str(sum(exc_index)) + ' items whose full name contain insuli or patient are excluded.')
    elif med_name_column_alt in df.columns:
        medname = df[[med_name_column_alt, 'medid']].copy().fillna('No med full name')
        exc_index = ((medname[med_name_column_alt].str.lower().str.contains(insulin_key))
                     | (medname[med_name_column_alt].str.lower().str.contains(patient_key)))
        log_debug(str(sum(exc_index)) + ' items whose full name contain insuli or patient are excluded.')
    else:
        log_debug(med_name_column + ' column is missed in input prepared dataset (all)')

    return exc_index


def remove_anesthesia_stations(df):
    exc_index = pd.Series()
    devicetype = 'devicetypekey'
    if (devicetype in df.columns):
        exc_index = ~((df[devicetype] == 1)|(df[devicetype] == 2))
        log_debug(str(sum(exc_index))+' items from non-medstation (Anesthesia stations or mini trays) are excluded.')

    else:
        log_debug(devicetype+ ' column is missed in input prepared dataset (all)')

    return exc_index


def remove_out_of_range_inv(df, inv_cutoff):
    exc_index = pd.Series()
    inv = 'inventoryquantity'
    if inv in df.columns:
        exc_index = df[inv] > inv_cutoff
        log_debug(str(sum(exc_index)) + ' items with inventory greater than ' + str(inv_cutoff) + ' are excluded.')

    else:
        log_debug(inv + ' column is missed in input prepared dataset (all)')

    return exc_index


def add_required_columns(inv_eed_ssflag, df_tranx, usage):
    # Add year month integer index for the aggregation
    inv_eed_ssflag['month'] = ((pd.DatetimeIndex(inv_eed_ssflag['monthfirstday']).year * 100 + pd.DatetimeIndex(
        inv_eed_ssflag['monthfirstday']).month) - 202000).astype(int)

    inv_eed_ssflag['ymint'] = (pd.DatetimeIndex(inv_eed_ssflag['monthfirstday']).year * 100 + pd.DatetimeIndex(
        inv_eed_ssflag['monthfirstday']).month).astype(int)

    # Distance to EED -- Fill NaNs with average dist_eed per pair
    inv_eed_ssflag = add_dist_eed(inv_eed_ssflag)

    # Categorical: (> 120) --> 0 (< 0) --> 2 (<= 120) --> 1
    inv_eed_ssflag['dist_eed_120'] = np.where(inv_eed_ssflag['dist_eed'] > 120, 0,
                                              np.where(inv_eed_ssflag['dist_eed'] < 0, 2, 1))

    # Categorical: (> 90) --> 0 (< 0) --> 2 (<= 90) --> 1
    inv_eed_ssflag['dist_eed_90'] = np.where(inv_eed_ssflag['dist_eed'] > 90, 0,
                                             np.where(inv_eed_ssflag['dist_eed'] < 0, 2, 1))

    # Add year month integer index for the aggregation
    df_tranx['ymint'] = pd.DatetimeIndex(df_tranx['transactionlocaldatetime']).year * 10000 + pd.DatetimeIndex(
        df_tranx['transactionlocaldatetime']).month * 100

    usage['ymint'] = (usage['year_month'] * 100).astype('int64') + 100

    usage['ymint'] = usage['ymint'].apply(lambda x: ((x // 10000) + 1) * 10000 + 100 if (x % 10000) > 1200 else x)

    usage['month'] = (round(usage['ymint'] / 100) - 202000).astype(int)
    usage['100avgdailyusage30'] = (usage['avgdailyusage30'] * 100).astype('int64')

    return inv_eed_ssflag, df_tranx, usage


def get_training_months(averagedaily_df):
    months = averagedaily_df.year_month.unique()
    test_months = []
    training_months = []
    n = months.size - 12 - 1
    log_debug(str(n) + ' months are found in the test set (usage)')

    if n > 0:
        for i in range(0, n):
            test_months.append(months.max())
            months = np.setdiff1d(months, np.array(test_months))

        training_months[:] = [number - 100 for number in test_months]
    else:
        log_warning('Not enough months of data in the usage input.')
    return training_months


def get_2_or_more_expiry_removal(df_all, training_months):
    for training_month in training_months:
        first_year = round(training_month / 100) * 100
        # print('first_year ',first_year)
        first_month = training_month - first_year  # e.g. 2
        if ((first_month + 12) - 1) <= 12:  # e.g. 13 > 12
            last_month = (first_month + 12) - 1
        else:
            last_month = (first_month + 12) - 1 - 12 + 100  # e.g. 101

        STE_Heu_df = df_all.loc[df_all["any expiry removal"] & (df_all['month'] >= first_month) &
                                (df_all['month'] <= last_month)]

        colname = "two_or_more_expiry_removals_" + str(int(training_month))
        two_or_more_expiry_removals = STE_Heu_df.groupby(
            by=["medid", "device_naturalkey", "skclientkey"]).count().reset_index().rename(
            columns={'any expiry removal': colname})

        bool_colname = "boolean_" + colname
        two_or_more_expiry_removals[bool_colname] = two_or_more_expiry_removals[colname] >= 2
        df_all = df_all.merge(two_or_more_expiry_removals[["medid", "device_naturalkey", "skclientkey", "month",
                                                  bool_colname]],
                        on=["medid", "device_naturalkey", "skclientkey", "month"], how="left")

        df_all[bool_colname] = df_all[bool_colname].fillna(False)

    return df_all


def build_required_dataframes(df_tranx, inv_eed_ssflag, usage, unitcost, clients):
    # month_cnt = 12
    # log_pair_count('inv_eed_ssflag', inv_eed_ssflag, training_months, month_cnt, clients_list)

    # For one site
    # aggregate at devicename-medid level
    df_tranx_grouped = df_tranx[['hdp_shardid', 'skclientkey', 'ymint', 'medid',
                                 'device_naturalkey', 'transactionquantity', 'actualbegincount']].loc[
        df_tranx['transactionquantity'] <= rmvl_cutoff].groupby(
        ['hdp_shardid', 'skclientkey', 'ymint', 'medid', 'device_naturalkey']).sum().reset_index()
    data["df_tranx_grouped"] = df_tranx_grouped

    df_all_inv_usage = pd.merge(inv_eed_ssflag[
                                    ['hdp_shardid', 'skclientkey', 'medid', 'device_naturalkey', 'ymint', 'month',
                                     'inventoryquantity', 'standardstockwithindispensingdeviceflag', 'dist_eed',
                                     'dist_eed_120', 'dist_eed_90', 'max_par_level', 'min_par_level', 'devicetypekey',
                                     'meditemflag']], usage[
                                    ['hdp_shardid', 'skclientkey', 'medid', 'device_naturalkey', 'month',
                                     'avgdailyusage30', '100avgdailyusage30']], how="left",
                                on=["hdp_shardid", "skclientkey", "medid", "device_naturalkey", "month"])

    # log_pair_count('df_all_inv_usage', df_all_inv_usage, training_months, month_cnt, clients_list)

    df_all_inv_usage_unitcost = pd.merge(df_all_inv_usage, unitcost, how='left',
                                         on=['hdp_shardid', 'skclientkey', 'medid'])

    # log_pair_count('df_all_inv_usage_unitcost', df_all_inv_usage_unitcost, training_months, month_cnt, clients_list)

    # ymint column in df_tranx_grouped is in 9 digits while in the df_all_inv_usage_unitcost, it is 5 digits
    df_tranx_grouped2 = df_tranx_grouped.copy()
    df_tranx_grouped2.loc[:, 'ymint'] = df_tranx_grouped2['ymint'] / 100
    df_all = pd.merge(df_all_inv_usage_unitcost, df_tranx_grouped2, how="left",
                      on=["hdp_shardid", "skclientkey", "medid", "device_naturalkey", "ymint"])  # .fillna(0)
    # log_pair_count('df_all', df_all, training_months, month_cnt, clients_list)

    df_all[['avgdailyusage30', '100avgdailyusage30', 'unitcost', 'transactionquantity']] = df_all[
        ['avgdailyusage30', '100avgdailyusage30', 'unitcost', 'transactionquantity']].fillna(0)

    df_all_mednames = df_all.merge(data["df_medfullnames"][['skclientkey', 'medid', 'medfullname']],
                                   on=["skclientkey", "medid"], how='left')
    df_all_names = df_all_mednames.merge(
        data["df_dispensingdevicenames"][['skclientkey', 'device_naturalkey', 'dispensingdevicename']],
        on=["skclientkey", "device_naturalkey"], how='left')

    debug_txt = df_all_names.loc[df_all_names['medfullname'].isnull()]['medid'].unique()
    if debug_txt.size > 0:
        log_warning(' These Med Ids had no medfullname:\t' + debug_txt)

    df_all_names['any expiry removal'] = ~df_all_names['actualbegincount'].isnull()

    training_months = get_training_months(usage)
    df_all_names= get_2_or_more_expiry_removal(df_all_names, training_months)

    df_all_names.to_csv('df_all_' + clients + '_before_exclusion.csv', index=False)
    return df_all_names


def apply_ste_exclusions(name, df_all, clients):
    ex1 = remove_non_drug_items(df_all)
    ex2 = remove_insulins(df_all)
    ex3 = remove_anesthesia_stations(df_all)
    ex4 = remove_out_of_range_inv(df_all, inv_cutoff)

    exclusion_marks = pd.DataFrame({'non_drug': ex1, 'insuli_patient': ex2, 'anesth_minitry': ex3, 'outrng_inv': ex4})

    assert df_all.shape[0] == exclusion_marks.shape[0]
    df_all_tmp_new = pd.concat([df_all, exclusion_marks], axis=1)
    
    df_all_tmp_new.to_csv("df_all_tmp_new.csv")
    
    df_all_tmp_new_eligibles = df_all_tmp_new.loc[~(
                df_all_tmp_new['non_drug'] | df_all_tmp_new['insuli_patient'] | df_all_tmp_new['anesth_minitry'] |
                df_all_tmp_new['outrng_inv'])]

    df_all_tmp_new_eligibles.to_csv(name + "_" + clients + '_after_exclusion.csv', index=False)

    log_empty_df('apply_STE_exclusions', df_all_tmp_new_eligibles)
    return df_all_tmp_new_eligibles


def preprocess_data(clients, client_output_dir):
    print("Working on preprocessing data ...")
    time1 = time.perf_counter()
    output_directory = client_output_dir+"\\STE_Results_"+str(ts)+"\\"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    os.chdir(output_directory)    
    #os.chdir(client_output_dir)

    global data
    data["inv_eed_ssflag"], data["df_tranx"], data["usage"] = add_required_columns(data["inv_eed_ssflag"],
                                                                                   data["df_tranx"], data["usage"])

    data["df_all"] = build_required_dataframes(
        data["df_tranx"], data["inv_eed_ssflag"], data["usage"], data["unitcost"], clients)

    data["df_all"] = apply_ste_exclusions("df_all", data["df_all"], clients)
    data["df_tranx_all_grouped"] = data["df_all"].loc[data["df_all"]["any expiry removal"]]
    data["df_tranx_all_grouped"].to_csv(clients + "_df_tranx_all_grouped.csv", index=False)

    time2 = time.perf_counter()
    print(f"Completed preprocessing data in {time2 - time1:0.2f} seconds.")


def check_merge(merged_df, left_df, right_df, merge_type):
    """
    @func - validates the merge of any type between two dataframes
    @params
        1. merged_df: merged dataframe of left and right dataframes
        2. left_df: left dataframe of the merge
        3. right_df: right dataframe of the merge
        4. mergeType: ex.) 'left', 'right', 'outer', 'inner', 'cross'
    """
    if merge_type == 'left':
        if "right_only" in merged_df["_merge"]:
            return False
        if merged_df.shape[0] == left_df.shape[0]:
            return True
    elif merge_type == 'right':
        if "left_only" in merged_df["_merge"]:
            return False
        if merged_df.shape[0] == right_df.shape[0]:
            return True
    elif merge_type == 'outer':
        if merged_df.shape[0] >= max(left_df.shape[0], right_df.shape[0]):
            return True
    elif merge_type == 'inner':
        if ("left_only" in merged_df["_merge"]) or ("right_only" in merged_df["_merge"]):
            return False
        if merged_df.shape[0] <= min(left_df.shape[0], right_df.shape[0]):
            return True
    elif merge_type == 'cross':
        if merged_df.shape[0] <= left_df.shape[0] * right_df.shape[0]:
            return True
    else:
        log_warning("Incorrect Merge Type.")
        return False


def get_latest_tranx_medid_clinetkey():
    latest_tranx_medid_clinetkey = data["df_tranx"][['skclientkey', 'medid', 'transactionlocaldatetime']].groupby(
        ['skclientkey', 'medid']).max()
    medfullnames = data["df_tranx"][['skclientkey', 'medid', 'transactionlocaldatetime', 'medfullname']].merge(
        latest_tranx_medid_clinetkey, how='inner', on=['skclientkey', 'medid', 'transactionlocaldatetime'],
        indicator=True, validate="m:1").drop_duplicates(keep='last')
    medfullnames = medfullnames.groupby(
        ['skclientkey', 'medid', 'transactionlocaldatetime', 'medfullname']).size().reset_index()

    return medfullnames


def get_latest_tranx_dispensingdevicename_clinetkey():
    latest_tranx_device_naturalkey_clinetkey = \
        data["df_tranx"][['skclientkey', 'device_naturalkey', 'transactionlocaldatetime']].groupby(
            ['skclientkey', 'device_naturalkey']).max()
    dispensingdevicenames = \
        data["df_tranx"][
            ['skclientkey', 'device_naturalkey', 'transactionlocaldatetime', 'dispensingdevicename']].merge(
            latest_tranx_device_naturalkey_clinetkey, how='inner',
            on=['skclientkey', 'device_naturalkey', 'transactionlocaldatetime'],
            indicator=True, validate="m:1").drop_duplicates(keep='last')
    dispensingdevicenames = \
        dispensingdevicenames.groupby(['skclientkey', 'device_naturalkey',
                                       'transactionlocaldatetime', 'dispensingdevicename']).size().reset_index()

    return dispensingdevicenames


def get_clientkeys():
    client_list = []
    cks = ''
    for item in os.listdir(input_dir):
        if os.path.isdir(input_dir+item) and item != "all_data":
            client_list.append(item)
            cks += str(item[:5]) + '_'
    return client_list, cks[:-1]


def add_dist_eed(inv_eed_ssflag):
    tmp_inv_eed_ssflag = inv_eed_ssflag.copy()

    NUll_tmp_rplcmt = 99999999
    tmp_inv_eed_ssflag['dist_eed'] = ((tmp_inv_eed_ssflag['earliestnextexpirationdate'] - tmp_inv_eed_ssflag[
        'monthfirstday']) / np.timedelta64(1, 'D')).fillna(NUll_tmp_rplcmt).astype(int)

    inv_eed_ssflag_no_Null_EED = tmp_inv_eed_ssflag.loc[~np.isnan(tmp_inv_eed_ssflag['earliestnextexpirationdate'])][
        ['skclientkey', 'medid', 'device_naturalkey', 'dist_eed']]
    inv_eed_ssflag_w_average_dist_eed = np.floor(
        inv_eed_ssflag_no_Null_EED.groupby(['skclientkey', 'medid', 'device_naturalkey']).mean()).reset_index().rename(
        columns={'dist_eed': 'avg_dist_eed'})

    tmp_inv_eed_ssflag = tmp_inv_eed_ssflag.merge(inv_eed_ssflag_w_average_dist_eed,
                                                  on=['skclientkey', 'medid', 'device_naturalkey'], how='left')
    tmp_inv_eed_ssflag.loc[(tmp_inv_eed_ssflag['dist_eed'] == NUll_tmp_rplcmt) & (
        ~np.isnan(tmp_inv_eed_ssflag['avg_dist_eed'])), 'dist_eed'] = tmp_inv_eed_ssflag['avg_dist_eed']

    tmp_inv_eed_ssflag.loc[(tmp_inv_eed_ssflag['dist_eed'] == NUll_tmp_rplcmt), 'dist_eed'] = 0

    log_empty_df("add_dist_eed", tmp_inv_eed_ssflag)
    return tmp_inv_eed_ssflag


def get_prediction_cutoff(clientkey, training_first_ymint, margin_percent):
    prediction_cutoff_df = \
        data["df_tranx_grouped"][['skclientkey', 'medid', 'device_naturalkey', 'transactionquantity']].loc[
            (data["df_tranx_grouped"]['skclientkey'] == clientkey) &
            (data["df_tranx_grouped"]['ymint'] / 100 >= training_first_ymint) &
            (data["df_tranx_grouped"]['ymint'] / 100 < (training_first_ymint + 100))
            ].groupby(['skclientkey', 'medid', 'device_naturalkey'])['transactionquantity'].max().reset_index()
    prediction_cutoff_df['prediction_cutoff'] = round(
        prediction_cutoff_df['transactionquantity'] * (1 + margin_percent))

    log_empty_df("get_prediction_cutoff", prediction_cutoff_df)
    return prediction_cutoff_df


def look_at_distinct_training_pairs(pairs, clients_list, clients):
    bins = range(0, 13)
    text = pd.DataFrame()
    for CK in clients_list:
        out = pd.cut(pairs.loc[(pairs['skclientkey'] == CK)]['count'], bins=bins)
        temp = pd.DataFrame(pd.value_counts(out).sort_index())
        temp['clientkey'] = CK
        text = text.append(temp)
        print(str(CK), '\n', pd.value_counts(out).sort_index())
        pairs.loc[(pairs['count'] > 1) & (pairs['skclientkey'] == CK)]['count'].hist(bins=12, cumulative=False,
                                                                                     alpha=0.7, figsize=(5, 8))

    text.to_csv('pairs_number of months with removal' + clients + '.csv')
    log_debug("outputted csv containing number of months with removal for pairs")


def get_xy(training_first_ymint, features, skclientkey, medid, devicekey):
    first_year = round(training_first_ymint / 100) * 100
    first_month = training_first_ymint - first_year  # e.g. 2

    if ((first_month + 12) - 1) <= 12:  # e.g. 13 > 12
        last_month = (first_month + 12) - 1
    else:
        last_month = (first_month + 12) - 1 - 12 + 100  # e.g. 101

    df = data["df_tranx_all_grouped"].copy().loc[
        (data["df_tranx_all_grouped"]['month'] >= first_month)
        & (data["df_tranx_all_grouped"]['month'] <= last_month)
        & (data["df_tranx_all_grouped"]['skclientkey'] == skclientkey)
        & (data["df_tranx_all_grouped"]['medid'] == medid)
        & (data["df_tranx_all_grouped"]['device_naturalkey'] == devicekey)]

    trained_set = df[['skclientkey', 'medid', 'device_naturalkey', 'month']]

    y = df['transactionquantity']

    if 'dist_eed_120' in features:
        x = pd.get_dummies(df[features], columns=['dist_eed_120', 'standardstockwithindispensingdeviceflag'], prefix='',
                           prefix_sep='')
    else:
        x = df[features]
    # x = pd.get_dummies(df[features],columns = ['standardstockwithindispensingdeviceflag'], prefix='', prefix_sep='')

    # the below may add new column because it was always 0 and now is 1 or the other way,
    # so cause issue with min_max_scaler
    # x = pd.get_dummies(df[features],columns = ['standardstockwithindispensingdeviceflag'], prefix='', prefix_sep='')

    # Normalize x
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    log_empty_df("getxy", pd.DataFrame(trained_set))
    log_empty_df("getxy", pd.DataFrame(x))
    log_empty_df("getxy", pd.DataFrame(x_scaled))
    log_empty_df("getxy", pd.DataFrame(y))
    return pd.DataFrame(trained_set), pd.DataFrame(x), pd.DataFrame(x_scaled), pd.DataFrame(y), min_max_scaler


def get_grouped_xy(training_first_ymint, features, pairs):
    # training_first_ymint = yyyymm e.g. 202002
    # first_year = yyyy00 e.g. 202000

    first_year = round(training_first_ymint / 100) * 100
    first_month = training_first_ymint - first_year  # e.g. 2

    if ((first_month + 12) - 1) <= 12:  # e.g. 13 > 12
        last_month = (first_month + 12) - 1
    else:
        last_month = (first_month + 12) - 1 - 12 + 100  # e.g. 101

    df = data["df_tranx_all_grouped"].copy().loc[
        (data["df_tranx_all_grouped"]['month'] >= first_month)
        & (data["df_tranx_all_grouped"]['month'] <= last_month)].merge(pairs, how="inner",
                                                                                    on=['skclientkey', 'medid',
                                                                                        'device_naturalkey'])

    trained_set = df[['skclientkey', 'medid', 'device_naturalkey', 'month']]

    y = df['transactionquantity']

    if 'dist_eed_120' in features:
        x = pd.get_dummies(df[features], columns=['dist_eed_120', 'standardstockwithindispensingdeviceflag'], prefix='',
                           prefix_sep='')
    else:
        x = df[features]
    # x = pd.get_dummies(df[features],columns = ['standardstockwithindispensingdeviceflag'], prefix='', prefix_sep='')

    # the below may add new column because it was always 0 and now is 1 or the other way,
    # so cause issue with min_max_scaler
    # x = pd.get_dummies(df[features],columns = ['standardstockwithindispensingdeviceflag'], prefix='', prefix_sep='')

    # Normalize x
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    log_empty_df("getgroupedxy", pd.DataFrame(trained_set))
    log_empty_df("getgroupedxy", pd.DataFrame(x))
    log_empty_df("getgroupedxy", pd.DataFrame(x_scaled))
    log_empty_df("getgroupedxy", pd.DataFrame(y))
    return pd.DataFrame(trained_set), pd.DataFrame(x), pd.DataFrame(x_scaled), pd.DataFrame(y), min_max_scaler


def get_test_xy(test_ymint, features, skclientkey, medid, device_naturalkey, min_max_scaler):
    first_year = round(test_ymint / 100) * 100
    test_month = test_ymint - first_year + 100  # e.g. 102

    x_scaled = pd.DataFrame()

    df = data["df_tranx_all_grouped"].copy().loc[(data["df_tranx_all_grouped"]['month'] == test_month) &
                                                 (data["df_tranx_all_grouped"]['skclientkey'] == skclientkey) &
                                                 (data["df_tranx_all_grouped"]['medid'] == medid) &
                                                 (data["df_tranx_all_grouped"]['device_naturalkey'] == device_naturalkey
                                                  )]
    y = df['transactionquantity']

    if 'dist_eed_120' in features:
        x = pd.get_dummies(df[features], columns=['dist_eed_120', 'standardstockwithindispensingdeviceflag'], prefix='',
                           prefix_sep='')
    else:
        x = df[features]
    # x = pd.get_dummies(df[features],columns = ['standardstockwithindispensingdeviceflag'], prefix='', prefix_sep='')

    # the below may add new column because it was always 0 and now is 1 or the other way,
    # so cause issue with min_max_scaler
    # x = pd.get_dummies(df[features],columns = ['standardstockwithindispensingdeviceflag'], prefix='', prefix_sep='')

    # Get the month number in integer from "yyyymm00"

    # update feature because of the dummy columns
    updated_features = x.columns
    # Normalize x
    if x[updated_features[0]].count() > 0:
        x_scaled = min_max_scaler.transform(x)
    else:
        usage_testset = data["usage"].copy().loc[(data["usage"]['month'] == test_month)
                                                 & (data["usage"]['skclientkey'] == skclientkey)
                                                 & (data["usage"]['medid'] == medid)
                                                 & (data["usage"]['device_naturalkey'] == device_naturalkey)][
            ['skclientkey', 'medid', 'device_naturalkey', 'month', '100avgdailyusage30']]

        inv_testset = data["inv_eed_ssflag"].copy().loc[(data["inv_eed_ssflag"]['month'] == test_month)
                                                        & (data["inv_eed_ssflag"]['skclientkey'] == skclientkey)
                                                        & (data["inv_eed_ssflag"]['medid'] == medid)
                                                        & (data["inv_eed_ssflag"][
                                                               'device_naturalkey'] == device_naturalkey)][
            ['skclientkey', 'medid', 'device_naturalkey', 'month', 'inventoryquantity', 'dist_eed', 'dist_eed_120',
             'standardstockwithindispensingdeviceflag', 'max_par_level', 'min_par_level']]

        unitcost_testset = data["unitcost"].copy().loc[(data["unitcost"]['skclientkey'] == skclientkey)
                                                       & (data["unitcost"]['medid'] == medid)]
        if inv_testset['month'].count() > 0:
            x_0 = inv_testset.merge(usage_testset, how="left",
                                    on=['skclientkey', 'medid', 'device_naturalkey', 'month']).fillna(0)
            x_1 = x_0.merge(unitcost_testset, how='left', on=['skclientkey', 'medid']).fillna(0)

            x = x_1[features]
            x_d = x

            updated_features = x_d.columns

            if x[updated_features[0]].count() > 0:
                x_scaled = min_max_scaler.transform(x_d)
                y = pd.DataFrame(0, index=np.arange(len(x_scaled)), columns=['transactionquantity'])

    log_empty_df("get_test_xy", pd.DataFrame(x_scaled))
    log_empty_df("get_test_xy", pd.DataFrame(y))
    return x, pd.DataFrame(x_scaled), pd.DataFrame(y)


def linear_regression_model(x, y):
    metric_name = 'score' #rsquared
    model = HuberRegressor(epsilon=1.01, max_iter=200, alpha=0.0001,fit_intercept=True, tol=1e-04)
    y=y.values.ravel()
    model.fit(x,y)
    metric = model.score(x,y)
    return model ,metric_name, metric


def ridge_model(x, y):
    metric_name = 'score'  # r-squared
    model = Ridge(alpha=0.1)
    model.fit(x, y)
    metric = model.score(x, y)
    return model, metric_name, metric


def run_model(model_name, x, y):
    if model_name == 'LR':
        model, metric_name, metric = linear_regression_model(x, y)
    elif model_name == 'Ridge':
        model, metric_name, metric = ridge_model(x, y)
    else:
        model, metric_name, metric = linear_regression_model(x, y)
    return model, metric_name, metric


def get_modeled_pairs_df(pairs, model_index):
    temp_modeled_pairs = pairs[['skclientkey', 'medid', 'device_naturalkey']].copy()
    # temp_modeled_pairs.columns = ['skclientkey','medid','devicekey']

    # temp_modeled_pairs['model_index'] = model_index
    temp_modeled_pairs.loc[:, 'model_index'] = model_index
    temp_modeled_pairs.loc[:, 'months_w_removals'] = pairs.iloc[0]['count']

    # temp_modeled_pairs.columns = ['skclientkey', 'medid', 'dispensingdevicename','model_index']
    log_empty_df("get_modeled_pairs_df", temp_modeled_pairs)
    return temp_modeled_pairs


def get_pair_prediction_cutoff(training_first_ymint, ste_result, margin_percent):
    STE_result_w_modified_predictions = ste_result.copy()
    clientkey = ste_result['skclientkey'].unique()[0]
    prediction_cutoff_df = get_prediction_cutoff(clientkey, training_first_ymint, margin_percent)
    STE_result_w_modified_predictions = STE_result_w_modified_predictions.merge(prediction_cutoff_df,
                                                                                on=['skclientkey', 'medid',
                                                                                    'device_naturalkey'], how='inner')

    STE_result_w_modified_predictions.loc[
        (STE_result_w_modified_predictions['modified_y_predict_less_cons'] >
         STE_result_w_modified_predictions['prediction_cutoff']), 'modified_y_predict_less_cons'] = \
        STE_result_w_modified_predictions['prediction_cutoff']

    log_empty_df("get_pair_prediction_cutoff", STE_result_w_modified_predictions)
    return STE_result_w_modified_predictions


def add_modified_predictions(training_first_ymint, ste_result, is_pair_prediction_cutoff, margin_percent):
    STE_result_w_modified_predictions = ste_result.copy()

    # Conservative y_predict to at most give the inventory quanytity at the beginning of the month
    STE_result_w_modified_predictions.loc[:, 'modified_y_predict'] = np.where(
        STE_result_w_modified_predictions['y_predict'] < 0, 0, np.floor(STE_result_w_modified_predictions['y_predict']))
    STE_result_w_modified_predictions.loc[:, 'modified_y_predict'] = STE_result_w_modified_predictions[
        ['modified_y_predict', 'inventoryquantity']].min(axis=1)

    STE_result_w_modified_predictions.loc[:, 'modified_y_predict_less_cons'] = np.floor(
        STE_result_w_modified_predictions['y_predict'])
    STE_result_w_modified_predictions.loc[
        STE_result_w_modified_predictions['modified_y_predict_less_cons'] < 0, 'modified_y_predict_less_cons'] = 0

    if is_pair_prediction_cutoff:
        # margin_percent = -0.1
        STE_result_w_modified_predictions = get_pair_prediction_cutoff(training_first_ymint,
                                                                       STE_result_w_modified_predictions,
                                                                       margin_percent)
    else:
        STE_result_w_modified_predictions.loc[STE_result_w_modified_predictions[
                                                  'modified_y_predict_less_cons'] > prediction_cutoff,
                                              'modified_y_predict_less_cons'] = prediction_cutoff
    # STE_result_w_modified_predictions.loc[:,'modified_y_predict_less_cons']=np.where
    # ((STE_result_w_modified_predictions['y_predict'] > prediction_cutoff) | (STE_result['y_predict'] < 0) ,
    # 0, np.floor(STE_result['y_predict']))

    log_empty_df("add_modified_predictions", STE_result_w_modified_predictions)
    return STE_result_w_modified_predictions


def get_features_norm(features):
    features_norm = []
    for feature in features:
        features_norm.append(feature + '_norm')
    return features_norm


def add_dashboard_estimate(ste_result, dashboard_estimate_percentage, dashboard_max_dist_EED):
    ste_result_w_Dashboard_estimate = ste_result

    if 'dist_eed' in ste_result_w_Dashboard_estimate.columns:
        ste_result_w_Dashboard_estimate.loc[:, 'Dashboard Estimate'] = np.where(
            (ste_result_w_Dashboard_estimate['dist_eed'] <= dashboard_max_dist_EED) & (
                        ste_result_w_Dashboard_estimate['dist_eed'] >= 0),
            np.floor(dashboard_estimate_percentage * ste_result_w_Dashboard_estimate['inventoryquantity']), 0)
        ste_result_w_Dashboard_estimate.loc[:, 'modified_y_predict_120'] = np.where(
            ste_result_w_Dashboard_estimate['dist_eed'] <= dashboard_max_dist_EED,
            ste_result_w_Dashboard_estimate['modified_y_predict'], 0)
    elif 'dist_eed_120' in ste_result_w_Dashboard_estimate.columns:
        # can check if EED is past or not
        ste_result_w_Dashboard_estimate.loc[:, 'Dashboard Estimate'] = np.where(
            (ste_result_w_Dashboard_estimate['dist_eed_120'] == 1),
            np.floor(dashboard_estimate_percentage * ste_result_w_Dashboard_estimate['inventoryquantity']), 0)
        ste_result_w_Dashboard_estimate.loc[:, 'modified_y_predict_120'] = np.where(
            (ste_result_w_Dashboard_estimate['dist_eed_120'] == 1) | (
                        ste_result_w_Dashboard_estimate['dist_eed_120'] == 2),
            ste_result_w_Dashboard_estimate['modified_y_predict'], 0)

    ste_result_w_Dashboard_estimate.loc[:, 'MSE_Dashboard'] = (ste_result_w_Dashboard_estimate['Dashboard Estimate'] -
                                                               ste_result_w_Dashboard_estimate['y_df']) ** 2
    ste_result_w_Dashboard_estimate.loc[:, 'MAE_Dashboard'] = abs(
        ste_result_w_Dashboard_estimate['Dashboard Estimate'] - ste_result_w_Dashboard_estimate['y_df'])

    log_empty_df("add_dashboard_estimate", ste_result_w_Dashboard_estimate)
    return ste_result_w_Dashboard_estimate


def add_estimate_errors(ste_result):
    STE_result_w_estimate_errors = ste_result.copy()

    STE_result_w_estimate_errors.loc[:, 'MSE_Estimator_Model'] = (STE_result_w_estimate_errors[
                                                                      'modified_y_predict_less_cons'] -
                                                                  STE_result_w_estimate_errors['y_df']) ** 2
    STE_result_w_estimate_errors.loc[:, 'MAE_Estimator_Model'] = abs(
        STE_result_w_estimate_errors['modified_y_predict_less_cons'] - STE_result_w_estimate_errors['y_df'])

    # STE_result.loc[:,'Predicted_to_Actual']=np.where(STE_result['y_df'] == 0,0,
    # STE_result['modified_y_predict_less_cons']/STE_result['y_df'])
    # STE_result_w_estimate_errors.loc[:,'Predicted_to_Actual'] = 0
    STE_result_w_estimate_errors['Predicted_to_Actual'] = 0

    epsolin = 0.000001  # avoind division by zero error
    STE_result_w_estimate_errors.loc[STE_result_w_estimate_errors['y_df'] > 0, 'Predicted_to_Actual'] = \
        STE_result_w_estimate_errors['modified_y_predict_less_cons'] / (STE_result_w_estimate_errors['y_df'] + epsolin)

    log_empty_df("add_estimate_errors", STE_result_w_estimate_errors)
    return STE_result_w_estimate_errors


def build_result_df(margin_percent, is_pair_prediction_cutoff, training_first_ymint,
                    included_pairs, x_df, x_scaled_df, y_df, y_predict_df):

    # Built a dataframe of STE results
    STE_result = included_pairs.reset_index()
    features = x_df.columns
    STE_result[features] = x_df.reset_index()[features]
    # [['month2', '100avgdailyusage30', 'inventoryquantity', 'dist_eed', 'standardstockwithindispensingdeviceflag']]

    # Generate column names for the x-test (normalized Xs) and add them
    features_norm = get_features_norm(features)
    x_scaled_df.columns = features_norm
    STE_result.loc[:, features_norm] = x_scaled_df.reset_index()[features_norm]
    # ['month2', '100avgdailyusage30', 'inventoryquantity', 'dist_eed', 'standardstockwithindispensingdeviceflag']]

    # Add actual Ys
    STE_result.loc[:, 'y_df'] = y_df.reset_index()['transactionquantity']

    # Add predicted Ys
    y_predict_df = y_predict_df.rename(columns={0: 'y_predict'})
    STE_result.loc[:, 'y_predict'] = y_predict_df.reset_index()['y_predict']

    # Add modifed predicts and dashboard estimate to STE result dataframe
    STE_result = add_modified_predictions(training_first_ymint, STE_result, is_pair_prediction_cutoff,
                                          margin_percent)  #

    # Add simulated Dashboard esimates as 45% of the inventory when eed is in the next 120 days
    dashboard_estimate_percentage = 0.45
    dashboard_max_dist_EED = 120
    STE_result = add_dashboard_estimate(STE_result, dashboard_estimate_percentage, dashboard_max_dist_EED)

    # Add Mean Squared Errors (MSE), Mean Absolute Errors (MAE), Error Percentage (Actual to Prediction)
    STE_result = add_estimate_errors(STE_result)

    log_empty_df("build_result_df", STE_result)
    return STE_result


def get_training_pairs(df_tranx_all_grouped_not_unloaded, training_first_ymint, min_cnt):
    # training_first_ymint = yyyymm e.g. 202002
    # first_year = yyyy00 e.g. 202000
    first_year = round(training_first_ymint / 100) * 100

    first_month = training_first_ymint - first_year  # e.g. 2
    if ((first_month + 12) - 1) <= 12:  # e.g. 13 > 12
        last_month = (first_month + 12) - 1
    else:
        last_month = (first_month + 12) - 1 - 12 + 100  # e.g. 101

    Pairs = df_tranx_all_grouped_not_unloaded.loc[(df_tranx_all_grouped_not_unloaded['month'] >= first_month) &
                                                  (df_tranx_all_grouped_not_unloaded['month'] <= last_month)].groupby(
        ['hdp_shardid', 'skclientkey', 'medid', 'device_naturalkey']).size().reset_index().rename(columns={0: 'count'})
    Pairs = Pairs.loc[Pairs['count'] > min_cnt]

    return Pairs


def Grouped_Data(training_df):
    training_df=training_df.rename(columns={"y_df":"removal qty"})
    df=training_df.groupby(["skclientkey","medid","device_naturalkey"]).agg(Last_Removal_month=('month',max),First_Removal_month=\
    ('month',min), Month_w_Removal_count=('removal qty','count'),  Avg_Inventory_Qty_training_peroid=('inventoryquantity','mean'),\
    Min_Removal_Qty=('removal qty',min),Max_Removal_Qty=('removal qty',max),Avg_Removal_Qty=('removal qty','mean')).reset_index()
    
    df['Last_Removal_month']=df['Last_Removal_month'].apply(lambda x: (x%100)+12 if x>100 else x)
    df['First_Removal_month']=df['First_Removal_month'].apply(lambda x: (x%100)+12 if x>100 else x)
    df['Month_w_Removal_count']=pd.to_numeric(df['Month_w_Removal_count'])
    df['Frequency_of_Removals']=(df['Last_Removal_month']-df['First_Removal_month'])/(df['Month_w_Removal_count']-1)
    df.loc[df['Month_w_Removal_count']==1,'Frequency_of_Removals']=1
    df['Avg_Inventory_Qty_training_peroid']=df['Avg_Inventory_Qty_training_peroid'].astype(int)
    df['Avg_Removal_Qty']=df['Avg_Removal_Qty'].astype(int)
    return df


def call_model(modelname, training_first_ymint, features, clientkey, exp, allpairs=True, grouped=True):
    log_debug("Calling " + str(modelname) + " model for " + str(clientkey) + " on " + str(training_first_ymint))
    print("Calling " + str(modelname) + " model for " + str(clientkey) + " on " + str(training_first_ymint))
    time1 = time.perf_counter()
    model_name = modelname
    lr_model = []

    modeled_pairs = pd.DataFrame(
        columns=['skclientkey', 'medid', 'device_naturalkey', 'model_index', 'months_w_removals'])
    scores = []
    min_max_scalers = []
    model_dict = {}  # (model_index: number of pairs)
    Grouped_model_param = len(features)  # ?  with dummy variable, the threshold should go higher!

    x_train_df = pd.DataFrame()
    x_scaled_train_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_predict_train_df = pd.DataFrame()
    trained_sets_df = pd.DataFrame()

    Pairs = get_training_pairs(data["df_tranx_all_grouped"], training_first_ymint, 1)

    if allpairs:
        Pairs2 = Pairs.loc[Pairs['skclientkey'] == clientkey]
    else:
        Pairs2 = Pairs.loc[Pairs['skclientkey'] == clientkey].head(10)  # temperorary get a small subset

    pair_index = 0
    model_index = 0

    if grouped:
        Pairs_count_above5 = Pairs2.loc[Pairs2['count'] > Grouped_model_param]
        Pairs_count_below6 = Pairs2.loc[Pairs2['count'] < Grouped_model_param + 1]

        for index, pair in Pairs_count_above5.iterrows():
            skclientkey = pair.skclientkey
            medid = pair.medid
            # devicekey = pair.dispensingdevicekey
            devicekey = pair.device_naturalkey

            # plotxy(df_tranx_all_grouped_not_unloaded, skclientkey, medid ,devicekey)
            trained_set, x, x_scaled, y, min_max_scaler = get_xy(training_first_ymint, features, skclientkey,
                                                                 medid, devicekey)

            min_max_scalers.append(min_max_scaler)

            model, metric_name, metric = run_model(model_name, x_scaled, y)

            y_predict = model.predict(x_scaled)

            lr_model.append(model)
            scores.append(metric)

            model_dict[model_index] = 1
            months_w_removals = pair['count']
            modeled_pair = {'skclientkey': skclientkey, 'medid': medid, 'device_naturalkey': devicekey,
                            'model_index': model_index, 'months_w_removals': months_w_removals}
            modeled_pairs = modeled_pairs.append(modeled_pair, ignore_index=True)
            # modeled_pairs.append([skclientkey, medid, devicekey, model_index])

            pair_index += 1  # recorded pair index
            model_index += 1  # recorded model index

            x_train_df = x_train_df.append(x)
            x_scaled_train_df = x_scaled_train_df.append(x_scaled)
            y_train_df = y_train_df.append(y)
            y_predict_train_df = y_predict_train_df.append(pd.DataFrame(y_predict))
            trained_sets_df = trained_sets_df.append(trained_set)

        log_debug(str(pair_index - 1) + ' pairs all modeled individually')

        for removal_count in range(2, Grouped_model_param + 1):
            pairs = Pairs_count_below6.loc[Pairs_count_below6['count'] == removal_count]

            log_debug(str(pairs['count'].count()) + ' pairs are modeled in group ' + str(removal_count))

            pair_index += pairs['count'].count()

            if pairs['count'].count() > 0:

                temp_modeled_pair = get_modeled_pairs_df(pairs, model_index)
                modeled_pairs = modeled_pairs.append(temp_modeled_pair)

                model_dict[model_index] = pairs['count'].count()

                trained_set, x, x_scaled, y, min_max_scaler = get_grouped_xy(training_first_ymint, features, pairs)
                min_max_scalers.append(min_max_scaler)

                model, metric_name, metric = run_model(model_name, x_scaled, y)

                y_predict = model.predict(x_scaled)

                lr_model.append(model)
                scores.append(metric)

                x_train_df = x_train_df.append(x)
                x_scaled_train_df = x_scaled_train_df.append(x_scaled)
                y_train_df = y_train_df.append(y)
                y_predict_train_df = y_predict_train_df.append(pd.DataFrame(y_predict))
                trained_sets_df = trained_sets_df.append(trained_set)

                model_index += 1  # recorded model index

            else:
                log_debug('no pairs w ' + str(removal_count) + ' removal count is found.')
    else:
        pair_index = 0

        for index, pair in Pairs2.iterrows():

            skclientkey = pair.skclientkey
            medid = pair.medid
            devicekey = pair.device_naturalkey

            # plotxy(df_tranx_all_grouped_not_unloaded, skclientkey, medid ,devicekey)

            trained_set, x, x_scaled, y, min_max_scaler = get_xy(features, skclientkey, medid, devicekey)

            min_max_scalers.append(min_max_scaler)

            model, metric_name, metric = run_model(model_name, x_scaled, y)

            y_predict = model.predict(x_scaled)

            lr_model.append(model)
            scores.append(metric)

            months_w_removals = pair['count']
            modeled_pair = {'skclientkey': skclientkey, 'medid': medid, 'device_naturalkey': devicekey,
                            'model_index': model_index, 'months_w_removals': months_w_removals}
            modeled_pairs.append(modeled_pair, ignore_index=True)
            # modeled_pairs.append([skclientkey, medid, devicekey, model_index])

            x_train_df = x_train_df.append(x)
            x_scaled_train_df = x_scaled_train_df.append(x_scaled)
            y_train_df = y_train_df.append(y)
            y_predict_train_df = y_predict_train_df.append(pd.DataFrame(y_predict))
            trained_sets_df = trained_sets_df.append(trained_set)

            pair_index += 1  # recorded pair index
            model_index += 1  # recorded model index

            if pair['count'] == 2:
                print('intercept_: ', model.intercept_)
                print('coef_', model.coef_)
    log_debug(str(pair_index - 1) + ' pairs are modeled via ' + str(model_index - 1) + ' ' + str(modelname) + ' models')

    # it will not be the same as features if dummy columns are added.
    updated_features = x_train_df.columns

    mar_percent = -0.1  # not-used when is_pair_prediction_cutoff = False
    is_pair_prediction_cutoff = False  # it is false for the training set
    result_df = build_result_df(mar_percent, is_pair_prediction_cutoff,
                                training_first_ymint, trained_sets_df, x_train_df, x_scaled_train_df,
                                y_train_df, y_predict_train_df)
    result_df.to_csv('STE Training result_' + str(clientkey) + '_starts_' + str(
            training_first_ymint) + '_' + exp + '.csv')

    # print(np.sqrt(result_df[['MSE_Dashboard', 'MSE_Estimator_Model']].mean()))
    # print(result_df[['MAE_Dashboard', 'MAE_Estimator_Model']].mean())
    time2 = time.perf_counter()
    print(f"Completing calling model in {time2 - time1:0.2f} seconds.")
    log_debug('Maximum Error Percent: ' + str(result_df['Predicted_to_Actual'].max()))
    
    result_grouped_df= build_result_df(mar_percent, is_pair_prediction_cutoff ,training_first_ymint, trained_sets_df, x_train_df, x_scaled_train_df, y_train_df, y_predict_train_df)
    result_grouped_df=Grouped_Data(result_df) # group model training dataframe
    result_grouped_df.to_csv('STE Training group result_'+ str(clientkey)+'_starts_'+str(training_first_ymint)+'_'+ exp +'.csv')    
    

    return updated_features, lr_model, scores, modeled_pairs, min_max_scalers, pair_index - 1, result_grouped_df


# Get model index for a pair

def get_pair_model_index(modeled_pairs, skclientkey, medid, devicekey):
    modeled_pair = modeled_pairs.loc[
        (modeled_pairs['skclientkey'] == skclientkey) & (modeled_pairs['medid'] == medid) & (
                modeled_pairs['device_naturalkey'] == devicekey)]
    model_index = modeled_pair['model_index']
    months_w_removal = modeled_pair['months_w_removals']

    if model_index.count() > 0:
        return int(model_index), int(months_w_removal)  # - 1)#as the indexed in the list starts with 0
    else:
        return 99999999, 0  # could not find it


def get_result_df(pair, y_test, months_w_removal):
    # df of pair values in size of our x_set
    included_pair = pd.DataFrame(index=np.arange(len(y_test)),
                                 columns=['skclientkey', 'medid', 'device_naturalkey', 'model_index',
                                          'months_w_removals'])
    included_pair['skclientkey'] = pair['skclientkey']
    included_pair['medid'] = pair['medid']
    included_pair['device_naturalkey'] = pair['device_naturalkey']
    included_pair['model_index'] = pair['model_index']
    included_pair['months_w_removals'] = months_w_removal

    log_empty_df("get_result_df", included_pair)
    return included_pair


def add_errors(result, errors_df, trained_clientkey, test_month, is_pair_prediction_cutoff, margin_percent):
    Errors_df_w_new_records = errors_df.copy()

    MSE_Dashboard_mean_root, MSE_Estimator_Model_mean_root = np.sqrt(
        result[['MSE_Dashboard', 'MSE_Estimator_Model']].mean())
    MAE_Estimator_Model_mean = result[['MAE_Estimator_Model']].mean()
    Predicted_to_Actual_max = result['Predicted_to_Actual'].max()

    record = {'skclientkey': trained_clientkey, 'test_month': test_month,
              'is_pair_prediction_cutoff': is_pair_prediction_cutoff, 'margin_percent': margin_percent,
              'RMSE_Dashboard': MSE_Dashboard_mean_root, 'RMSE_Estimator': MSE_Estimator_Model_mean_root,
              'MAE_Estimator': MAE_Estimator_Model_mean, 'Max_Error_Percentage': Predicted_to_Actual_max}

    Errors_df_w_new_records = Errors_df_w_new_records.append(pd.DataFrame(record))

    log_empty_df("add_errors", Errors_df_w_new_records)
    return Errors_df_w_new_records


def get_test_dataset(df_all_eligibles, training_first_ymint, min_cnt, clients_list):
    first_year = round(training_first_ymint/100)*100 #202003 -> 202000
    first_month = training_first_ymint - first_year #e.g. 2 -> 03
    if (first_month + 12) <= 12: #e.g. 13 > 12
        test_month = first_month + 12
    else:
        test_month = first_month + 100 #e.g. 3 -> 103
    
    STE_test_df = df_all_eligibles.loc[(df_all_eligibles['month'] == test_month)].copy()    #6625            
   
    pairs_in_training_1 = get_training_pairs(df_all_eligibles, training_first_ymint, min_cnt) #get_test_pairs(df_eligible_w_removals, training_first_ymint,2)
        #print('pairs_in_training count(1): ',pairs_in_training_1.groupby('skclientkey')['count'].count())
    join_type = 'left'
        #Add pairs w zero removal during test month
    STE_test_df = STE_test_df.merge(pairs_in_training_1, on = ['hdp_shardid','skclientkey','medid','device_naturalkey'], how = join_type).copy()    
    STE_test_df.loc[:,['transactionquantity','count']] = STE_test_df[['transactionquantity','count']].fillna(0)    
    
    #log_pair_count('Test set (STE_test_df) - only_pairs_in_training: '+str(only_pairs_in_training)+' - min removal: '+str(min_cnt), STE_test_df, first_year+test_month, 1, clients_list)    
    
    return STE_test_df, test_month


def group_test_month(test_month):
    test_month=test_month % 1000
    if test_month >= 101:
        test_month=(test_month) % 100 + 12
    return test_month


def dataframe_one_zero_removal(df_all_eligibles_testdata,max_dist_eed_current_dashboard):
    result_df=df_all_eligibles_testdata
    result_df.loc[:,["skclientkey","medid","device_naturalkey","count","dist_eed","inventoryquantity","transactionquantity"]]=result_df[["skclientkey","medid","device_naturalkey","count","dist_eed","inventoryquantity","transactionquantity"]]
    result_df=result_df.rename(columns={"count":"Month_w_Removal_count"})
    result_df.loc[:,'dist_eed_120'] = np.where(result_df.dist_eed > max_dist_eed_current_dashboard, 0, np.where(result_df.dist_eed< 0,2, 1))
    result_df.loc[:,'Month_w_Removal_count']=1
    return result_df


def prediction_rule_one_month_w_removal(data_all, dashboard_percentage):
    
    conditions=[(data_all['Month_w_Removal_count'] ==1) & (data_all['dist_eed_120'] ==1),(data_all['Month_w_Removal_count'] ==1) & (data_all['dist_eed_120'] ==0),\
                (data_all['Month_w_Removal_count'] ==1) & (data_all['dist_eed_120'] ==2)]
    values=[np.floor(dashboard_percentage*data_all['inventoryquantity']),0,0]
    
    data_all.loc[:,'y_removal_group'] = np.select(conditions, values, default=0)
    data_all=data_all.rename(columns={"transactionquantity":"y_df"})
    data_all.loc[:,'MSE_Dashboard']=(data_all['y_df']-data_all['y_removal_group'])**2
    data_all.loc[:,'MAE_Dashboard']=(data_all['y_df']-data_all['y_removal_group']).abs()
    data_all.loc[:,'MSE_Group_Model']=(data_all['y_df']-data_all['y_removal_group'])**2
    data_all.loc[:,'MAE_Group_Model']=(data_all['y_df']-data_all['y_removal_group']).abs()
    return data_all


def combined_test_dataframe(all_result_df,two_more_result_df):
    print('all_result_df: ',all_result_df.shape)
    print('two_more_result_df: ',two_more_result_df.shape)
    all_result_df.to_csv('all_result_df_'+str(ts)+'.csv')
    two_more_result_df.to_csv('two_more_result_df_'+str(ts)+'.csv')

    try:
        combined_test_df =all_result_df.set_index(['skclientkey', 'medid','device_naturalkey'])
        combined_test_df.update(two_more_result_df.set_index(['skclientkey', 'medid','device_naturalkey']))
        combined_test_df.reset_index()
    except Exception as e:
        log_warning('Combining test dataframe was not successful\n'+str(e) )
    return combined_test_df


def prediction_by_group_model(grouped_df,group_test_month,max_dist_eed,frequency_sample):
    
    #Appy the dashboard dist_eed concept for the group model 
    conditions = [
    (grouped_df['Month_w_Removal_count'] ==2) & (grouped_df['dist_eed']>max_dist_eed),(grouped_df['Month_w_Removal_count'] == 2) & (grouped_df['dist_eed']<0),
    (grouped_df['Month_w_Removal_count'] ==3) & (grouped_df['dist_eed']>max_dist_eed),(grouped_df['Month_w_Removal_count'] == 3) & (grouped_df['dist_eed']<0),
    (grouped_df['Month_w_Removal_count'] ==4) & (grouped_df['dist_eed']>max_dist_eed),(grouped_df['Month_w_Removal_count'] == 4) & (grouped_df['dist_eed']<0),
    (grouped_df['Month_w_Removal_count']==5) & (grouped_df['dist_eed']>max_dist_eed),(grouped_df['Month_w_Removal_count'] == 5) & (grouped_df['dist_eed']<0)]
    choices =np.zeros(8)
    
    grouped_df['y_removal_group'] = np.select(conditions, choices, default=grouped_df.modified_y_predict_less_cons)
    #grouped_df['y_removal_group'] = np.where( ( (grouped_df.dist_eed> 120) | (grouped_df.dist_eed<0)) , 0, (grouped_df.modified_y_predict_less_cons).apply(np.floor)) 
    
    conditions = [(grouped_df['Month_w_Removal_count'] ==2) & ((grouped_df['Frequency_of_Removals'] <frequency_sample['freq_removal_2'])) & ((grouped_df['Last_Removal_month'] >=group_test_month-4)),\
                 (grouped_df['Month_w_Removal_count'] ==3) & ((grouped_df['Frequency_of_Removals'] <frequency_sample['freq_removal_3'])) & ((grouped_df['Last_Removal_month'] >=group_test_month-5)),\
                 (grouped_df['Month_w_Removal_count'] ==4) & ((grouped_df['Frequency_of_Removals'] <frequency_sample['freq_removal_4'])) & ((grouped_df['Last_Removal_month'] >=group_test_month-5)),\
                 (grouped_df['Month_w_Removal_count'] ==5) & ((grouped_df['Frequency_of_Removals'] <frequency_sample['freq_removal_5'])) &((grouped_df['Last_Removal_month'] >=group_test_month-5)),\
                 (grouped_df['Month_w_Removal_count'] >5), (grouped_df['Month_w_Removal_count']==1)]

    # create a list of the values we want to assign for each condition
    values = [grouped_df['y_removal_group'], grouped_df['y_removal_group'], grouped_df['y_removal_group'], grouped_df['y_removal_group'],\
             grouped_df['y_removal_group'], grouped_df['y_removal_group']]

 # create a new column and use np.select to assign values to it using our lists as arguments
    grouped_df["y_removal_group"] = np.select(conditions, values, default=0)
    
#  # Apply the dashboard rule for months-w-removal equals 1
#     grouped_df=prediction_rule_one_month_w_removal(grouped_df, dashboard_percentage)
    grouped_df["MSE_Group_Model"]=(grouped_df['y_df']-grouped_df['y_removal_group'])**2
    grouped_df["MAE_Group_Model"]=(grouped_df['y_df']-grouped_df['y_removal_group']).abs()
    
    return grouped_df


def start_modeling(m, exp, features, training_first_ymints, clients_list):
    print("Working on modeling...")
    func_time1 = time.perf_counter()

    Errors_df = pd.DataFrame()
    Modeled_pair_count_df = pd.DataFrame()
    training_first_ymint = None
    modeled_pairs_count = None
    modeled_pairs = None
    min_max_scalers = None
    lr_model = None
    
    # Group model parameters.............
    frequency_sample = {"freq_removal_2": 3,"freq_removal_3": 2.5,"freq_removal_4": 2.3, "freq_removal_5": 2} # these frequencies are hyperparamters
    max_dist_eed=90  # group model function parameter 
    max_dist_eed_current_dashboard=120 # parameters for [dataframe_one_zero_removal](current_dashboard) function
    dashboard_percentage= 0.45
    
    for trained_clientkey in clients_list:
        for training_first_ymint in training_first_ymints:
            updated_features, lr_model, scores, modeled_pairs, min_max_scalers, modeled_pairs_count, result_grouped_df = \
                call_model(m, training_first_ymint, features, trained_clientkey, exp)
            df_scores = pd.DataFrame(scores)
            df_scores.columns = ['score']
            df_scores.loc[df_scores['score'] >= 0].plot.hist()
            plt.suptitle(m)

            Modeled_pair_count_df_record = {'clientkey': [trained_clientkey],
                                            'training_first_month': [training_first_ymint],
                                            'modeled_pairs_count': [modeled_pairs_count]}
            Modeled_pair_count_df = Modeled_pair_count_df.append(pd.DataFrame(Modeled_pair_count_df_record))
    
            Training_Pairs_One_Client = modeled_pairs  # .head(100)
    
            # e.g. training_first_ymint --> test_month ~ 202002 --> 202102
            test_month = training_first_ymint + 100
            log_debug('test_month: ' + str(test_month))
            counter = 0
    
            included_pairs = pd.DataFrame()
    
            x_df = pd.DataFrame()
            x_test_df = pd.DataFrame()
            y_df = pd.DataFrame()
            y_predict_df = pd.DataFrame()
    
            print("Working on test predictions...")
            test_time1 = time.perf_counter()
            for index, pair in Training_Pairs_One_Client.iterrows():
                skclientkey = pair.skclientkey
                medid = pair.medid
                devicekey = pair.device_naturalkey
                model_index, months_w_removal = get_pair_model_index(modeled_pairs, skclientkey, medid, devicekey)
                if model_index < 99999999:
                    min_max_scaler = min_max_scalers[model_index]
                    model = lr_model[model_index]
                    x, x_test, y_test = get_test_xy(test_month, features, skclientkey, medid, devicekey, min_max_scaler)
                    # if x_test.empty | y_test.empty:
                    #     STE_Hybrid_utilities.debug_outputs(skclientkey, test_month, medid, devicekey)
                    x_test_df = x_test_df.append(x_test)
                    x_df = x_df.append(x)
                    y_df = y_df.append(y_test)
                    if not x_test.empty:
                        y_test_predicted = model.predict(x_test)
                        y_predict_df = y_predict_df.append(pd.DataFrame(y_test_predicted))
                        # record the results
                        included_pair = get_result_df(pair, y_test, months_w_removal)
                        included_pairs = included_pairs.append(included_pair)
                    else:
                        counter += 1
                        log_warning("no data found for " + str(skclientkey) + ' ' + str(medid) + ' ' + str(devicekey))
                else:
                    counter += 1
                    log_warning('no model found for ' + str(skclientkey) + ' ' + str(medid) + ' ' + str(devicekey))
            test_time2 = time.perf_counter()
            print(f"Completed test predictions in {test_time2 - test_time1:0.2f} seconds.")
            log_debug(str(counter) + ' pairs in training set got no estimation')
    
            # Built a dataframe of STE results
    
            print("Working on outputting errors with prediction cutoff 0 margin...")
            res_time1 = time.perf_counter()
            # Print Errors with prediction cutoff 0% margin
    
            is_pair_prediction_cutoff = True  # True to use the training set max removals as prediction cutoff
            margin_percent = 0  # -0.90#
            STE_test_result = build_result_df(margin_percent, is_pair_prediction_cutoff,
                                              training_first_ymint, included_pairs, x_df, x_test_df,
                                              y_df, y_predict_df)
            Errors_df = add_errors(STE_test_result, Errors_df, trained_clientkey, test_month, is_pair_prediction_cutoff,
                                    margin_percent)
    
            box_plot_by_group(STE_test_result, margin_percent, features, trained_clientkey,
                                                    test_month, exp, output_dir)
    
            STE_test_result.to_csv('STE_estimator_results_rev2_' + str(trained_clientkey) + '_' + str(test_month) + '_' +
                exp + '_' + str(margin_percent) + '.csv')
    
            log_debug('Maximum Error Percent: ' + str(round(STE_test_result['Predicted_to_Actual'].max(), 1)))
    
            box_plot_individuals(
                STE_test_result, margin_percent, features, trained_clientkey, test_month, exp, output_dir)
    
            predict_vs_actual = STE_test_result[['y_df', 'modified_y_predict_less_cons']].rename(
                columns={'y_df': 'Actual removal', 'modified_y_predict_less_cons': 'Predicted removal'})
            ax1 = predict_vs_actual.plot.scatter(x='Actual removal',
                                                  y='Predicted removal',
                                                  c='DarkBlue')  # ,figsize=(20,20))
            
            lims = [
                np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
                np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
            ]
            plt.suptitle(trained_clientkey)
            # now plot both limits against eachother
            ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            # ax1.set_aspect('equal')
            ax1.set_xlim(lims)
            ax1.set_ylim(lims)
    
            ax1.get_figure().savefig(str(trained_clientkey) +
                                     '_' + str(test_month) + '_' + exp + '_' + str(margin_percent) + '.png')
    
            res_time2 = time.perf_counter()
            print(f"Completed outputs with prediction cutoff 0 margin in {res_time2 - res_time1:0.2f} seconds.")
    
            # Print Errors with no prediction cutoff
            res1_time1 = time.perf_counter()
    
            is_pair_prediction_cutoff = False  # True to use the training set max removals as prediction cutoff
            STE_test_result_temp = build_result_df(margin_percent, is_pair_prediction_cutoff,
                                                   training_first_ymint, included_pairs, x_df,
                                                   x_test_df, y_df, y_predict_df)
            Errors_df = add_errors(STE_test_result_temp, Errors_df, trained_clientkey, test_month,
                                   is_pair_prediction_cutoff, margin_percent)
    
            res1_time2 = time.perf_counter()
            print(f"Completed outputs with prediction cutoff 0 margin in {res1_time2 - res1_time1:0.2f} seconds.")
    
            log_debug('Maximum Error Percent: ' + str(round(STE_test_result_temp['Predicted_to_Actual'].max(), 1)))
    
            print("Working on model parameter outputs...")
            params_time1 = time.perf_counter()
            # Look at models' parameters
            columns = features
    
            model_params = pd.DataFrame()
            m_index = 0
    
            # for m_index2 in range(0,486):
            # model = lr_model[m_index2]
    
            # for model in lr_model:
            #     model_param = modeled_pairs.loc[modeled_pairs['model_index'] == m_index].copy()
            #     if model_param.skclientkey.count() == 1:
            #         model_param.loc[:, columns] = model.coef_
            #     else:
            #         model_param.loc[:, columns] = model.coef_[0, :]
    
            #     model_param.loc[:, 'intercept'] = model.intercept_[0]
            #     model_params = model_params.append(model_param)
            #     m_index += 1
    
            # # pairs, model_index, model parameters
    
            # model_params.to_csv('STE LR model Parameters_' + str(trained_clientkey) + '_starts_' + str(
            #     training_first_ymint) + '_' + exp + '.csv')
            # params_time2 = time.perf_counter()
            # print(f"Completed outputting model parameters in {params_time2 - params_time1:0.2f} seconds.")
            
            ## Prediction using group model
            group_dataframe =pd.merge(result_grouped_df,STE_test_result[['skclientkey','medid','device_naturalkey','max_par_level','standardstockwithindispensingdeviceflag','100avgdailyusage30','inventoryquantity','dist_eed','y_df','y_predict','modified_y_predict_less_cons','transactionquantity','Dashboard Estimate','MSE_Dashboard','MSE_Estimator_Model','MAE_Estimator_Model','MAE_Dashboard']], how='inner', on=['skclientkey','medid','device_naturalkey'])
            test_month_group=group_test_month(test_month)
            STE_group_test_result=prediction_by_group_model(group_dataframe,test_month_group,max_dist_eed,frequency_sample)
            
            STE_group_test_result.to_csv('STE_estimator_group_results_rev2_'+str(trained_clientkey)+'_'+ str(test_month)+'_'+ exp +'_'+str(margin_percent)+'.csv')
            box_plot_group_model(STE_group_test_result, margin_percent, trained_clientkey, test_month)
            
            # get the estimation from all eligibles pairs for the test month
            STE_group_test_result_1=STE_group_test_result[['skclientkey','medid','device_naturalkey','Month_w_Removal_count','dist_eed','inventoryquantity','y_df','y_removal_group','MAE_Dashboard','MAE_Group_Model']]
        
            
           
            all_test_df,Test_Month=get_test_dataset(data["df_all"], training_first_ymint, 0, trained_clientkey)
            
            
            all_test_dataframe= dataframe_one_zero_removal(all_test_df,max_dist_eed_current_dashboard)
            #all_test_df.to_csv("all_test_df.csv")
            all_test_result_df=prediction_rule_one_month_w_removal(all_test_dataframe, dashboard_percentage)
            all_test_result_df_1= all_test_result_df[['skclientkey','medid','device_naturalkey','Month_w_Removal_count','dist_eed','inventoryquantity','y_df','y_removal_group','MAE_Dashboard','MAE_Group_Model']]
            
            # Combine the group result with the zeors and one removals
            final_STE_result=combined_test_dataframe(all_test_result_df_1, STE_group_test_result_1)
            final_STE_result= final_STE_result.loc[:, ~final_STE_result.columns.str.contains('^Unnamed')]
            final_STE_result.to_csv('STE_estimator_all_pairs_results_'+str(trained_clientkey)+'_'+ str(test_month)+'_'+ exp +'_'+str(margin_percent)+'.csv')
            
    
            print(str(trained_clientkey)+'_'+ str(test_month)+'_'+ exp)

        func_time2 = time.perf_counter()
        print(f"Completed modeling in {func_time2 - func_time1:0.2f} seconds.")





def get_test_datasets(df_all, df_eligible_w_removals, training_first_ymint, clients_list):
    data, test_month = get_test_dataset_Heu(df_all, df_eligible_w_removals, training_first_ymint, 2, True, clients_list)
    data_1, test_month = get_test_dataset_Heu(df_all, df_eligible_w_removals, training_first_ymint, 1, True, clients_list)
    data_all, test_month = get_test_dataset_Heu(df_all, df_eligible_w_removals, training_first_ymint, 0, False, clients_list)
    return data, data_all, data_1, test_month


def print_df_errors(df_errors, CK, test_month):
    #ts = datetime.now().strftime("%Y%m%d-%I%M")
    Rule_No = int(df_errors['Rule'].unique())
    df_errors.drop_duplicates().sort_values(by=['Clientkey','test_month','Rule']).to_csv('STE_Heuristic_Errors_'+str(CK)+'_'+str(test_month)+'_'+str(Rule_No)+'_'+str(ts)+'.csv',index=False)


def Output_STE_Estimates(CK, STE_Estimate_all, df_errors, test_month, rplcmnt_prcntl, Rule_No, disteed_120):
    #### all pairs get STE estimates regardless their appearance in the training set
    min_rmvl_cnt = 0
    STE_Estimate_Errors, df_errors = print_STE_Estimate(CK, STE_Estimate_all, df_errors, min_rmvl_cnt, test_month, rplcmnt_prcntl, Rule_No, disteed_120)
    
    # Only pairs which have at least two months with removal get STE estimates
    #min_rmvl_cnt = 1
    #STE_Estimate_count1more = STE_Estimate_1.loc[STE_Estimate_1['count']>= min_rmvl_cnt]
    #STE_Estimate_Errors, df_errors = print_STE_Estimate(CK, STE_Estimate_count1more, df_errors, min_rmvl_cnt, test_month, rplcmnt_prcntl, Rule_No, disteed_120)
    
    # Only pairs which have at least two months with removal get STE estimates
    #min_rmvl_cnt = 2
    #STE_Estimate_count2more = STE_Estimate.loc[STE_Estimate['count']>= min_rmvl_cnt]
    #STE_Estimate_Errors, df_errors = print_STE_Estimate(CK, STE_Estimate_count2more, df_errors, min_rmvl_cnt, test_month, rplcmnt_prcntl, Rule_No, disteed_120)
    
    print_df_errors(df_errors, CK, test_month)

def set_outputfolder_structure(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
