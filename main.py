#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  18 10:48:00 2024

@author: aqiph

"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdFMCS

from tools import remove_unnamed_columns


### Apply docking score filter ###
def filter_by_dockingScore(input_file_dockingScore, input_file_SMILES, id_column_name='ID',
                           dockingScore_column_name='r_i_docking_score', dockingScore_cutoff=0.0):
    """
    Filter compounds based on docking score
    :param input_file_dockingScore: str, path of the input docking score file
    :param input_file_SMILES: str, path of the input SMILES file
    :param id_column_name: str, name of the ID column in input_file_dockingScore
    :param dockingScore_column_name: str, name of the docking score column
    :param dockingScore_cutoff: float, docking score cutoff
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file_dockingScore))[0] + '_DockingScore'

    df = pd.read_csv(input_file_dockingScore)
    df.rename(columns={id_column_name:'ID', dockingScore_column_name:'Docking_Score'}, inplace=True)
    df = pd.DataFrame(df, columns=['ID', 'Docking_Score'])
    print('Number of rows in docking score file:', df.shape[0])
    df_SMILES = pd.read_csv(input_file_SMILES)
    df_SMILES = pd.DataFrame(df_SMILES, columns=['ID', 'SMILES', 'Cleaned_SMILES'])
    print('Number of rows in SMILES file:', df_SMILES.shape[0])

    # round
    df['Docking_Score'] = df['Docking_Score'].apply(lambda score: np.round(score, decimals=3))
    # sort and deduplicate
    df.sort_values(by=['Docking_Score'], ascending=True, inplace=True)
    df = df.drop_duplicates(['ID'], keep='first', ignore_index=True)
    # filter
    df_filtered = df[df['Docking_Score'] <= dockingScore_cutoff]
    df_filtered = pd.DataFrame(df_filtered, columns=['ID', 'Docking_Score'])
    # merge
    df_filtered = pd.merge(df_filtered, df_SMILES, how='inner', on=['ID'])
    df_filtered = pd.DataFrame(df_filtered, columns=['ID', 'SMILES', 'Cleaned_SMILES', 'Docking_Score'])

    # write output file
    df_filtered = df_filtered.reset_index(drop=True)
    print('Number of rows in filtered docking score file:', df_filtered.shape[0])
    df_filtered = remove_unnamed_columns(df_filtered)
    df_filtered.to_csv(f'{output_file}_{df_filtered.shape[0]}.csv')
    print('Applying docking score filter is done.')


### Apply property filters ###
def filter_by_property(input_file_property, input_file_SMILES, id_column_name='ID',
                       property_column_names=None, property_filters=None):
    """
    Filter compounds based on property cutoff
    :param input_file_property: str, path of the input property file
    :param input_file_SMILES: str, path of the input SMILES file
    :param id_column_name: str, name of the ID column in input_file_property
    :param property_column_names: list of strs or None, names of the property columns
    :param property_filters: dict or None, dict of functions for property filters
    :return: None
    """
    if property_column_names is None:
        property_column_names = []
    if property_filters is None:
        property_filters = {'MW':lambda x: x <= 650, 'logP':lambda x: x <= 5.5}

    # files
    output_file = os.path.splitext(os.path.abspath(input_file_property))[0] + '_Property'

    df = pd.read_csv(input_file_property)
    df.rename(columns={id_column_name:'ID'}, inplace=True)
    df = pd.DataFrame(df, columns=['ID']+property_column_names)
    print('Number of rows in property file:', df.shape[0])
    df_SMILES = pd.read_csv(input_file_SMILES)
    df_SMILES = pd.DataFrame(df_SMILES, columns=['ID', 'SMILES', 'Cleaned_SMILES'])
    print('Number of rows in SMILES file:', df_SMILES.shape[0])

    # filter
    for column, filter in property_filters.items():
        try:
            df = df[df[column].apply(filter)]
        except Exception:
            print(f'Error: Filter for {column} column is not applied.')
            continue

    # merge
    df = pd.merge(df, df_SMILES, how='left', on=['ID'])
    df = pd.DataFrame(df, columns=['ID', 'SMILES', 'Cleaned_SMILES']+property_column_names)

    # write output file
    df = df.reset_index(drop=True)
    print('Number of rows in filtered property file:', df.shape[0])
    df = remove_unnamed_columns(df)
    df.to_csv(f'{output_file}_{df.shape[0]}.csv')
    print('Applying property filters is done.')


### Get cluster labels ###
def get_clusterLabel(input_file_clustering, input_file, id_column_name='ID', clusterLabel_column_name='MCS Cluster'):
    """
    Get cluster labels
    :param input_file_clustering: str, path of the input cluster label file
    :param input_file: str, path of the input docking score and property file
    :param id_column_name: str, name of the ID column in input_file_clustering
    :param clusterLabel_column_name: str, name of the cluster label column
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file))[0] + '_Clustering'

    df_clustering = pd.read_csv(input_file_clustering)
    df_clustering.rename(columns={id_column_name:'ID', clusterLabel_column_name:'MCS_Cluster'}, inplace=True)
    df_clustering = pd.DataFrame(df_clustering, columns=['ID', 'R-group', 'MCS_Cluster'])
    print('Number of rows in cluster label file:', df_clustering.shape[0])
    df = pd.read_csv(input_file)
    print('Number of rows in SMILES file:', df.shape[0])

    # merge
    df = pd.merge(df, df_clustering, how='inner', on=['ID'])

    # write output file
    df = df.reset_index(drop=True)
    print('Number of rows in combined file:', df.shape[0])
    df = remove_unnamed_columns(df)
    df.to_csv(f'{output_file}_{df.shape[0]}.csv')
    print('Getting cluster label is done.')



### Select representatives ###
def select_cmpds_from_clusters(input_file, clusterLabel_column_name, method, **kwargs):
    """
    Select compounds from each cluster
    :param input_file: str, path of the input file
    :param clusterLabel_column_name: str, name of the cluster label column
    :param method: str, method to select representatives, allowed values include 'best'
    :param outlier_label: label of outliers
    :param outlier_processing_method: str, method to process outliers, allowed values include 'include' and 'exclude'
    :param count_per_cluster: int, number of compounds from each cluster

    :param property_column_names: list of str, names of the property column
    :param dockingScore_column_name: str, name of the docking score column
    :param min_num_rules: int, minimum number of rules a compound need to satisfy

    """
    # read input file and define DataFrame for representatives
    df = pd.read_csv(input_file)
    df_representative = None

    # get cluster labels and outlier label
    clusterLabel_set = set(df[clusterLabel_column_name])
    outlier_label = kwargs.get('outlier_label', 0)
    outlier_processing_method = kwargs.get('outlier_processing_method', 'include')

    # process clusters
    for label in clusterLabel_set:
        print(f'Cluster label={label}...')
        df_cluster = df[df[clusterLabel_column_name] == label]
        # process outliers
        if label == outlier_label:
            if outlier_processing_method == 'include':
                count = df_cluster.shape[0]   # include all compounds for outlier cluster
            elif outlier_processing_method == 'exclude':
                continue
            else:
                raise Exception('Error: Invalid outlier processing method.')
        # process other clusters
        else:
            count = kwargs.get('count_per_cluster', 1)   # get the idea number of compounds per cluster, default is 1

        # get representatives from each cluster
        df_new_representative = pd.DataFrame()
        # get the best compounds
        if method == 'best':
            property_column_names = kwargs.get('property_column_names', [])
            dockingScore_column_name = kwargs.get('dockingScore_column_name', 'Docking_Score')
            min_num_rules = kwargs.get('min_num_rules', len(property_column_names))
            df_new_representative = get_best_compound(df_cluster, property_column_names, dockingScore_column_name, min_num_rules, count)
        # get the smallest compounds
        elif method == 'smallest':
            SMILES_column_name = kwargs.get('SMILES_column_name', 'SMILES')
            dockingScore_column_name = kwargs.get('dockingScore_column_name', 'Docking_Score')
            df_new_representative = get_smallest_compound(df_cluster, SMILES_column_name, dockingScore_column_name, count)
        elif method == 'MCS':
            SMILES_column_name = kwargs.get('SMILES_column_name', 'SMILES')
            dockingScore_column_name = kwargs.get('dockingScore_column_name', 'Docking_Score')
            df_new_representative = get_MCS_analog(df_cluster, SMILES_column_name, dockingScore_column_name, count, label==outlier_label)

        # concat representative df from each cluster
        if df_new_representative.shape[0] == 0:
            continue
        if df_representative is None:
            df_representative = df_new_representative
        else:
            df_representative = pd.concat([df_representative, df_new_representative], ignore_index=True, sort=False)

    # write output file
    df_representative = df_representative.reset_index(drop=True)
    print('Number of rows in the representative file:', df_representative.shape[0])
    df_representative = remove_unnamed_columns(df_representative)
    output_file = f'{os.path.splitext(input_file)[0]}_representative_{df_representative.shape[0]}.csv'
    df_representative.to_csv(output_file)


def get_best_compound(df_cluster, property_column_names, dockingScore_column_name, min_num_rules, count):
    """
    Helper function for select_cmpds_from_clusters. Select best compounds from each cluster.
    :param df_cluster: pd.DataFrame object which contains compounds from each cluster.
    :param property_column_names: list of str, names of the property column.
    :param dockingScore_column_name: str, name of the docking score column.
    :param min_num_rules: int, minimum number of rules a compound need to satisfy
    :param count: int, number of compounds from each cluster
    :return: pd.DataFrame object which contains selected representative compounds.
    """
    COLUMNS = df_cluster.columns.tolist()

    # compute compound property score (number of criteria met)
    df_cluster.insert(len(COLUMNS), 'Property_Score', df_cluster[property_column_names].sum(axis=1).values.tolist())
    # get compounds that meet minimum number of criteria
    df_cluster = df_cluster[df_cluster['Property_Score'] >= min_num_rules]
    # sort compounds based on property score first then docking score
    df_cluster = df_cluster.sort_values(by=['Property_Score', dockingScore_column_name], ascending=[False, True], ignore_index=True)
    # get representive compounds based on count
    n = min(count, df_cluster.shape[0])
    df_subset = df_cluster.loc[0:(n-1)]
    # df_subset = pd.DataFrame(df_subset, columns=COLUMNS)

    return df_subset


def get_smallest_compound(df_cluster, SMILES_column_name, dockingScore_column_name, count):
    """
    Helper function for select_cmpds_from_clusters. Select smallest compounds (i.e., least heavy atoms) from each cluster.
    :param df_cluster: pd.DataFrame object which contains compounds from each cluster.
    :param SMILES_column_name: str, name of the SMILES column
    :param count: int, number of compounds from each cluster
    :return: pd.DataFrame object which contains selected representative compounds.
    """
    COLUMNS = df_cluster.columns.tolist()

    # compute the number of heavy atoms
    NHA = [Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() for smiles in df_cluster[SMILES_column_name].values.tolist()]
    df_cluster.insert(len(COLUMNS), 'NHA', NHA)
    # sort compounds based on number of heavy atoms first then docking score
    df_cluster = df_cluster.sort_values(by=['NHA', dockingScore_column_name], ascending=[True, True], ignore_index=True)
    # get representive compounds based on count
    n = min(count, df_cluster.shape[0])
    df_subset = df_cluster.loc[0:(n-1)]
    # df_subset = pd.DataFrame(df_subset, columns=COLUMNS)

    return df_subset


def get_MCS_analog(df_cluster, SMILES_column_name, dockingScore_column_name, count, is_outlier):
    """
    Helper function for select_cmpds_from_clusters. Select smallest compounds (i.e., least heavy atoms) from each cluster.
    :param df_cluster: pd.DataFrame object which contains compounds from each cluster.
    :param SMILES_column_name: str, name of the SMILES column
    :param count: int, number of compounds from each cluster
    :param is_outlier: bool, whether this cluster is outliers
    :return: pd.DataFrame object which contains selected representative compounds.
    """
    # return outliers directly
    if is_outlier:
        return df_cluster

    COLUMNS = df_cluster.columns.tolist()

    # compute the list of MCS between all pairs
    MCS_list = []
    SMILES_list = df_cluster[SMILES_column_name].values.tolist()
    SMILES_num = len(SMILES_list)
    for i in range(SMILES_num):
        try:
            mol_1 = Chem.MolFromSmiles(SMILES_list[i])
        except Exception:
            print(f"Error: Invalid SMILES {SMILES_list[i]}.")
            continue
        for j in range(i+1, SMILES_num):
            try:
                mol_2 = Chem.MolFromSmiles(SMILES_list[j])
            except Exception:
                print(f"Error: Invalid SMILES {SMILES_list[j]}.")
                continue
            mcs = rdFMCS.FindMCS([mol_1, mol_2], completeRingsOnly=True, timeout=1).smartsString
            mcs_SMILES = Chem.MolToSmiles(Chem.MolFromSmarts(mcs))
            MCS_list.append(mcs_SMILES)
    # compute the most common MCS
    MCS_counter = Counter(MCS_list)
    most_common_MCS = MCS_counter.most_common(1)[0][0]
    most_common_MCS = Chem.MolToSmiles(Chem.MolFromSmiles(most_common_MCS))
    print(most_common_MCS)
    return df_cluster








if __name__ == '__main__':

    ### 1. Apply docking score filter ###
    # input_file_dockingScore = 'tests/test_dockingScore_filter.csv'
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # id_column_name = 'ID'
    # filter_by_dockingScore(input_file_dockingScore, input_file_SMILES, id_column_name,
    #                     dockingScore_column_name='docking score', dockingScore_cutoff=-6.9)


    ### 2. Apply property filters ###
    # input_file_property = 'tests/test_property_filter.csv'
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # id_column_name = 'ID'
    # property_column_names = ['Docking_Score', 'MW', 'logP', 'HBD', 'HBA', 'TPSA']
    # property_filters = {'MW':lambda x: x <= 650, 'logP':lambda x: x <= 5.5}
    # filter_by_property(input_file_property, input_file_SMILES, id_column_name,
    #                 property_column_names=property_column_names, property_filters=property_filters)


    ### 3.1. Get cluster labels ###
    # input_file_clustering = 'tests/test_get_clusterLabel.csv'
    # input_file = 'tests/test_property_filter_Property_5154.csv'
    # get_clusterLabel(input_file_clustering, input_file, id_column_name='ID', clusterLabel_column_name='MCS Cluster 0.7')

    ### 3.2. Select representatives ###
    input_file = 'tests/test_Molecular_Generation_Analysis.csv'
    clusterLabel_column_name = 'MCS_Cluster'
    ### get best compounds
    # method = 'best'
    # property_column_names = ['Chiral_Center_Requirement', 'COOH_Requirement', 'MIC_Model_Prediction']
    # dockingScore_column_name = 'Docking_SP_Score'
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, method, outlier_processing_method='include', count_per_cluster=1,
    #                            property_column_names=property_column_names, dockingScore_column_name=dockingScore_column_name,
    #                            min_num_rules=2)
    ### get the smallest compounds
    # method = 'smallest'
    # SMILES_column_name = 'SMILES'
    # dockingScore_column_name = 'Docking_SP_Score'
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, method, outlier_processing_method='include', count_per_cluster=1,
    #                            SMILES_column_names=SMILES_column_name, dockingScore_column_name=dockingScore_column_name)
    ### get the MCS analog
    method = 'MCS'
    SMILES_column_name = 'SMILES'
    dockingScore_column_name = 'Docking_SP_Score'
    select_cmpds_from_clusters(input_file, clusterLabel_column_name, method, outlier_processing_method='include', count_per_cluster=1,
                               SMILES_column_names=SMILES_column_name, dockingScore_column_name=dockingScore_column_name)




