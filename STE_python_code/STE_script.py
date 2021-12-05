import sys

# import STE_estimator_utilities
from STE_Hybrid_funcs import *
#from STE_Heuristic_funcs import *

inv_cutoff = 1000
rmvl_cutoff = 1000
prediction_cutoff = 50

input_dir = sys.argv[1]
output_dir = sys.argv[2]
default_dir = os.getcwd()

model = 'LR'
exp = 'Exp3'
features = ['100avgdailyusage30', 'inventoryquantity',
            'dist_eed', 'max_par_level', 'standardstockwithindispensingdeviceflag']


def main():
    print("Working on STE_estimator...")
    time1 = time.perf_counter()
    set_outputfolder_structure(output_dir)
    set_log_file(output_dir)
    client_list, clients = get_clientkeys()
    for client in client_list:
        skclientkey = [int(client[:5])]
        client = str(client)
        os.chdir(default_dir)
        client_input_dir,client_output_dir = input_dir + client + '/', output_dir + client + '/'
        training_months = read_inputs(client_input_dir)
        os.chdir(default_dir)
        preprocess_data(client, client_output_dir)

        # CHOOSE WHICH MODEL
        print('training_months\n',training_months)
        start_modeling(model, exp, features, training_months, skclientkey)

        #start_modeling_heuristic()
        df_all_eligibles = data['df_all']
        df_eligible_w_removals = data['df_tranx_all_grouped']
        unitcost = data['unitcost']
        usage = data['usage']
        CK = skclientkey[0]
        rplcmnt_prcntl = '30%'   

        #start_modeling_heuristic(default_dir, skclientkey[0], df_all_eligibles,df_eligible_w_removals,unitcost,usage,skclientkey, False, True, rplcmnt_prcntl)

    time2 = time.perf_counter()
    print(f"Completed working on STE_estimator in {time2-time1:0.2f} seconds.")

    quit()


if __name__ == '__main__':
    main()
    #STE_Hybrid_utilities.main()
