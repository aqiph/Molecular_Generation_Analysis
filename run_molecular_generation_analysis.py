#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  18 10:48:00 2024

@author: aqiph

"""

import sys, os
path_list = sys.path
module_path = '/Users/guohan/Documents/Codes/Molecular_Generation_Analysis'
if module_path not in sys.path:
    sys.path.append(module_path)
    print('Add module path')

import pandas as pd
from functools import reduce
from main import filter_by_dockingScore, filter_by_property, get_clusterLabel, select_cmpds_from_clusters
from tools import remove_unnamed_columns


def combine_results(input_file_list, columns=None):
    """
    Combine multiple dataset
    :param input_file_list:
    :return:
    """
    # files
    output_file = os.path.splitext(input_file_list[0])[0]

    df_list = []
    for i, file in enumerate(input_file_list):
        df = pd.read_csv(file)
        print(f'The number of raws in {file} is {df.shape[0]}')
        # remove 'SMILES' and 'Cleaned_SMILES'
        if i > 0:
            try:
                df.drop(labels=['SMILES'], axis=1, inplace=True)
            except:
                print(f'{file} does not contain SMILES')
            try:
                df.drop(labels=['Cleaned_SMILES'], axis=1, inplace=True)
            except:
                print(f'{file} does not contain Cleaned_SMILES')
        df_list.append(df)

    # merge DataFrames
    df_merge = reduce(lambda left, right: pd.merge(left, right, how='left', on=['ID']), df_list)
    if columns is not None:
        df_merge = pd.DataFrame(df_merge, columns=columns)

    # write output file
    df_merge = df_merge.reset_index(drop=True)
    print('Number of rows in the combined file:', df_merge.shape[0])
    df_merge = remove_unnamed_columns(df_merge)
    output_file = f'{output_file}_combine_{df_merge.shape[0]}.csv'
    df_merge.to_csv(output_file)



if __name__ == '__main__':

    ### 1. Apply docking score filter ###
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # input_file_dockingScore = 'tests/test_dockingScore_filter.csv'
    # id_column_name = 'ID'
    # filter_by_dockingScore(input_file_SMILES, input_file_dockingScore, id_column_name,
    #                        dockingScore_column_name='docking score', dockingScore_cutoff=-6.9)


    ### 2. Apply property filters ###
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # input_file_property = 'tests/test_property_filter.csv'
    # id_column_name = 'ID'
    # property_column_names = ['Docking_Score', 'MW', 'logP', 'HBD', 'HBA', 'TPSA']
    # property_filters = {'MW': lambda x: x <= 650, 'logP': lambda x: x <= 5.5}
    # filter_by_property(input_file_SMILES, input_file_property, id_column_name,
    #                    property_column_names=property_column_names, property_filters=property_filters)


    ### 3.1. Get cluster labels ###
    # input_file_clustering = 'tests/test_get_clusterLabel.csv'
    # input_file = 'tests/test_property_filter_Property_5154.csv'
    # get_clusterLabel(input_file_clustering, input_file, id_column_name='ID', clusterLabel_column_name='MCS Cluster 0.7')


    ### 3.2. Select representatives ###
    input_file = 'tests/test_Molecular_Generation_Analysis.csv'
    clusterLabel_column_name = 'MCS_Cluster'
    ### get best compounds
    method = 'best'
    property_column_names = ['Chiral_Center_Requirement', 'COOH_Requirement', 'MIC_Model_Prediction']
    dockingScore_column_name = 'Docking_SP_Score'
    select_cmpds_from_clusters(input_file, clusterLabel_column_name, method, outlier_processing_method='include', count_per_cluster=1,
                               property_column_names=property_column_names, dockingScore_column_name=dockingScore_column_name,
                               min_num_rules=2)
    ### get the smallest compounds
    # method = 'smallest'
    # SMILES_column_name = 'SMILES'
    # dockingScore_column_name = 'Docking_SP_Score'
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, method, outlier_processing_method='include', count_per_cluster=1,
    #                            SMILES_column_names=SMILES_column_name, dockingScore_column_name=dockingScore_column_name)

    ### Combine results ###
    # input_file_list = ['tests/test_combine_files_1.csv', 'tests/test_combine_files_2.csv']
    # columns = ['ID', 'SMILES', 'Cleaned_SMILES', 'Docking_SP_Score', 'Docking_Covalent_Fast_Score',
    #            'MW', 'TPSA', 'HBD', 'logP',
    #            'Chiral_Center_Requirement', 'COOH_Requirement', 'MIC_Model_Prediction', 'MCS Cluster']
    # combine_results(input_file_list, columns=columns)