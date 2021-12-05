import os
import pandas as pd
#import plotly.express as px
#import plotly.graph_objects as go
#import plotly.io as pio
import matplotlib.pyplot as plt
import time
from STE_script import output_dir

#pio.templates.default = "seaborn"

raw_data = {}

try:
    for file in os.listdir("data/all_data/"):
        if "df_all" in file:
            raw_data[int(file[7:12])] = pd.read_csv("data/all_data/" + file)
except:
    print("All_data folder not found. ")

clients = list(raw_data.keys())


def remove_non_drug_items(df):
    drug_item_flag = 'meditemflag'
    df_tmp = df.loc[df[drug_item_flag] == 1]
    return df_tmp


def remove_insulins(df):
    med_name_column = 'medfulname'
    insulin_key = 'insuli'
    patient_key = 'patient'

    df_tmp = pd.DataFrame()

    if med_name_column in df.columns:
        df_tmp = df.loc[~df[med_name_column].isnull()]
        df_tmp = df_tmp.loc[~((df_tmp[med_name_column].str.lower().str.contains(insulin_key))
                              | (df_tmp[med_name_column].str.lower().str.contains(patient_key)))]
    return df_tmp


def remove_anesthesia_stations(df):
    devicetype = 'devicetypekey'
    df_tmp = df.loc[(df[devicetype] == 1) | (df[devicetype] == 2)]
    return df_tmp


def remove_outofrange_inv(df, inv_cutoff):
    inv = 'inventoryquantity'
    df_tmp = df.loc[df[inv] <= inv_cutoff]
    return df_tmp


def get_df_to_plot(client, data, proportion):
    results = []
    if proportion:
        pairs_removed_filter_1 = (data.shape[0] - remove_non_drug_items(data).shape[0]) / data.shape[0]
        results.append([client, "non_drug_items", pairs_removed_filter_1])
        pairs_removed_filter_2 = (data.shape[0] - remove_insulins(data).shape[0]) / data.shape[0]
        results.append([client, "insulins", pairs_removed_filter_2])
        pairs_removed_filter_3 = (data.shape[0] - remove_anesthesia_stations(data).shape[0]) / data.shape[0]
        results.append([client, "anesthesia_stations", pairs_removed_filter_3])
        pairs_removed_filter_4 = (data.shape[0] - remove_outofrange_inv(data, 1000).shape[0]) / data.shape[0]
        results.append([client, "outofrange_inv", pairs_removed_filter_4])
        pairs_removed_all_filters = ((data.shape[0] - remove_non_drug_items(data).shape[0])
                                     + (data.shape[0] - remove_insulins(data).shape[0])
                                     + (data.shape[0] - remove_anesthesia_stations(data).shape[0])
                                     + (data.shape[0] - remove_outofrange_inv(data, 1000).shape[0])) / data.shape[0]
        results.append([client, "all_filters", pairs_removed_all_filters])
    else:
        pairs_removed_filter_1 = (data.shape[0] - remove_non_drug_items(data).shape[0])
        results.append([client, "non_drug_items", pairs_removed_filter_1])
        pairs_removed_filter_2 = (data.shape[0] - remove_insulins(data).shape[0])
        results.append([client, "insulins", pairs_removed_filter_2])
        pairs_removed_filter_3 = (data.shape[0] - remove_anesthesia_stations(data).shape[0])
        results.append([client, "anesthesia_stations", pairs_removed_filter_3])
        pairs_removed_filter_4 = (data.shape[0] - remove_outofrange_inv(data, 1000).shape[0])
        results.append([client, "outofrange_inv", pairs_removed_filter_4])
        pairs_removed_all_filters = ((data.shape[0] - remove_non_drug_items(data).shape[0])
                                     + (data.shape[0] - remove_insulins(data).shape[0])
                                     + (data.shape[0] - remove_anesthesia_stations(data).shape[0])
                                     + (data.shape[0] - remove_outofrange_inv(data, 1000).shape[0]))
        results.append([client, "all_filters", pairs_removed_all_filters])

    temp = pd.DataFrame(results)
    temp.columns = ["client", "filter", "proportion"]

    return temp


def coverage_statistics():
    train_df = []
    test_df = []
    for client in clients:
        train_months = list(raw_data[client]["ymint"].unique())
        for train_month in train_months:
            train = raw_data[client].copy()[raw_data[client]["ymint"] == train_month]
            pairs_removed_filter_1 = (train.shape[0] - remove_non_drug_items(train).shape[0])
            pairs_removed_filter_2 = (train.shape[0] - remove_insulins(train).shape[0])
            pairs_removed_filter_3 = (train.shape[0] - remove_anesthesia_stations(train).shape[0])
            pairs_removed_filter_4 = (train.shape[0] - remove_outofrange_inv(train, 1000).shape[0])

            tmp = remove_non_drug_items(train)
            tmp = remove_insulins(tmp)
            tmp = remove_anesthesia_stations(tmp)
            tmp = remove_outofrange_inv(tmp, 1000)
            total_pairs_removed = train.shape[0] - tmp.shape[0]

            pairs_eed_dist_between_0_and_120 = train["dist_eed_120"].value_counts()[1]

            train_df.append([client, train_month, train.shape[0], pairs_removed_filter_1, pairs_removed_filter_2,
                             pairs_removed_filter_3, pairs_removed_filter_4, total_pairs_removed,
                             train.shape[0] - total_pairs_removed, pairs_eed_dist_between_0_and_120])

        for test_month in [202103, 202104, 202105]:
            test = raw_data[client].copy()[raw_data[client]["ymint"] == test_month]
            pairs_removed_filter_1 = (test.shape[0] - remove_non_drug_items(test).shape[0])
            pairs_removed_filter_2 = (test.shape[0] - remove_insulins(test).shape[0])
            pairs_removed_filter_3 = (test.shape[0] - remove_anesthesia_stations(test).shape[0])
            pairs_removed_filter_4 = (test.shape[0] - remove_outofrange_inv(test, 1000).shape[0])
            tmp = remove_non_drug_items(test)
            tmp = remove_insulins(tmp)
            tmp = remove_anesthesia_stations(tmp)
            tmp = remove_outofrange_inv(tmp, 1000)
            total_pairs_removed = test.shape[0] - tmp.shape[0]

            pairs_eed_dist_between_0_and_120 = test["dist_eed_120"].value_counts()[1]

            test_df.append([client, test_month, test.shape[0], pairs_removed_filter_1, pairs_removed_filter_2,
                            pairs_removed_filter_3, pairs_removed_filter_4, total_pairs_removed,
                            test.shape[0] - total_pairs_removed, pairs_eed_dist_between_0_and_120])

    cols = ["clientkey", "train_month", "total_pairs", "non_drug_items", "insulins", "anesthesia_stations",
            "outofrange_inv", "total_pairs_removed", "pairs_remaining", "pairs_eed_between_0_and_120"]

    train_df = pd.DataFrame(train_df)
    train_df.columns = cols
    train_df.to_csv("All_Sites_Training_Data.csv", index=False)

    test_df = pd.DataFrame(test_df)
    cols[1] = "test_month"
    test_df.columns = cols
    test_df.to_csv("All_Sites_Testing_Data.csv", index=False)


def box_plot_individuals(result, margin_percent, features, trained_clientkey, test_month, exp, output_file):
    # last_model_index = result['model_index'].max()
    # group_result = result.loc[result['model_index'] > (last_model_index - len(features)+1)]

    individuals_result = result.loc[result['months_w_removals'] > len(features)]

    MAEs_mean_individuals = individuals_result[['months_w_removals', 'MAE_Dashboard', 'MAE_Estimator_Model']]

    boxprops = dict(linestyle='-', linewidth=2, color='b')
    medianprops = dict(linestyle='-', linewidth=2, color='b')

    # Individuals ###################################################
    ax_boxplot3 = MAEs_mean_individuals.boxplot(column=['MAE_Dashboard', 'MAE_Estimator_Model'], figsize=(10, 5),
                                                showfliers=False, showmeans=True, boxprops=boxprops,
                                                medianprops=medianprops)
    plt.suptitle(str(trained_clientkey) + '_' + str(test_month) + '_' + exp)
    plt.savefig('MAE_Boxplot_individuals' + str(trained_clientkey) + '_' + str(test_month) + '_'
                + exp + '_' + str(margin_percent) + '.png')

    ax_boxplot4 = MAEs_mean_individuals.boxplot(column=['MAE_Dashboard', 'MAE_Estimator_Model'], figsize=(10, 5),
                                                showfliers=True, showmeans=True, boxprops=boxprops,
                                                medianprops=medianprops)

    plt.suptitle(str(trained_clientkey) + '_' + str(test_month) + '_' + exp)
    plt.savefig('MAE_Boxplot_individuals' + str(trained_clientkey) + '_' + str(test_month) + '_'
                + exp + '_' + str(margin_percent) + 'outliers.png')

    MAEs_mean_individuals.drop('months_w_removals', axis=1).describe().to_csv(
        'MAEs_mean_individuals' + str(trained_clientkey) + '_' + str(test_month) + '_' + exp + '_' + str(
            margin_percent) + '.txt')

    plt.clf()


def box_plot_by_group(result, margin_percent, features, trained_clientkey, test_month, exp, output_file):
    # last_model_index = result['model_index'].max()
    # group_result = result.loc[result['model_index'] > (last_model_index - len(features)+1)]

    group_result = result.loc[result['months_w_removals'] <= len(features)]

    # MSEs_mean_root_group = np.sqrt(group_result[['model_index', 'MSE_Dashboard', 'MSE_Estimator_Model']])
    # MAEs_mean_group = group_result[['model_index','MAE_Dashboard','MAE_Estimator_Model']]
    MAEs_mean_group = group_result[['months_w_removals', 'MAE_Dashboard', 'MAE_Estimator_Model']]

    # Predicted_to_Actual_max_group = group_result[['model_index','Predicted_to_Actual']]
    # Predicted_to_Actual_max_group = group_result[['months_w_removals', 'Predicted_to_Actual']]

    boxprops = dict(linestyle='-', linewidth=2, color='b')
    medianprops = dict(linestyle='-', linewidth=2, color='b')

    # Groups ###################################################
    ax_boxplot1 = MAEs_mean_group.boxplot(column=['MAE_Dashboard', 'MAE_Estimator_Model'], figsize=(10, 5),
                                          by='months_w_removals',
                                          showfliers=False, showmeans=True, boxprops=boxprops, medianprops=medianprops)
    plt.suptitle(str(trained_clientkey) + '_' + str(test_month) + '_' + exp)
    plt.savefig('MAE_Boxplot_groups' + str(trained_clientkey) + '_' + str(test_month) + '_'
                + exp + '_' + str(margin_percent) + '.png')

    ax_boxplot2 = MAEs_mean_group.boxplot(column=['MAE_Dashboard', 'MAE_Estimator_Model'], figsize=(10, 5),
                                          by='months_w_removals',
                                          showfliers=True, showmeans=True, boxprops=boxprops, medianprops=medianprops)
    plt.suptitle(str(trained_clientkey) + '_' + str(test_month) + '_' + exp)
    plt.savefig('MAE_Boxplot_groups' + str(trained_clientkey) + '_' + str(test_month) + '_'
                + exp + '_' + str(margin_percent) + 'outliers.png')

    MAEs_mean_group.groupby('months_w_removals').describe().to_csv(
        'MAEs_mean_group' + str(trained_clientkey) + '_' + str(test_month) + '_' + exp + '_' + str(
            margin_percent) + '.txt')
    plt.clf()
    
    
def box_plot_group_model(group_df, margin_percent, trained_clientkey, test_month):
    group_individual_result=group_df

    Square_Error_group = group_individual_result[['Month_w_Removal_count','MAE_Estimator_Model','MAE_Group_Model']]
    
    boxprops = dict(linestyle='-', linewidth=2, color='b')
    medianprops = dict(linestyle='-', linewidth=2, color='g')
    
    # Groups ###################################################
    ax_boxplot1 =Square_Error_group.boxplot(column=['MAE_Estimator_Model', 'MAE_Group_Model'], figsize=(10,5), by = 'Month_w_Removal_count'
                                         ,showfliers=False, showmeans=True,boxprops=boxprops,medianprops=medianprops)
    plt.suptitle(str(trained_clientkey)+'_'+ str(test_month))
    plt.savefig('Absolute_Errors_Boxplot_group_model'+str(trained_clientkey)+'_'+ str(test_month)+'_'+str(margin_percent)+'.png')
    
    ax_boxplot2 = Square_Error_group.boxplot(column=['MAE_Estimator_Model', 'MAE_Group_Model'], figsize=(10,5), by = 'Month_w_Removal_count'
                                         ,showfliers=True, showmeans=True,boxprops=boxprops,medianprops=medianprops)
    plt.suptitle(str(trained_clientkey)+'_'+ str(test_month))
    plt.savefig('Absolute_Errors_Boxplot_group_model'+str(trained_clientkey)+'_'+ str(test_month)+'_'+ str(margin_percent)+'_'+'outliers.png')
    plt.clf()


debug_output_list = []


def debug_outputs(skclientkey, test_month, medid, devicekey):
    debug_output_list.append([skclientkey, test_month, medid, devicekey])


def main():
    print("Working on coverage statistics...")
    time1 = time.perf_counter()
    coverage_statistics()

    debug_df = pd.DataFrame(debug_output_list)
    debug_df.columns = ["clientkey", "test_month", "medid", "devicekey"]
    debug_df.to_csv("Debug_File.csv", index=False)
    time2 = time.perf_counter()
    print(f"Completed exporting coverage statistics in {time2-time1:0.2f} seconds.")


if __name__ == '__main__':
    main()
