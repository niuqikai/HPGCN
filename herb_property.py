#基于etcm和TCMSP生成对应的herb分数向量
# -- coding:utf-8 --
#核心代码，计算318种中药(或者也可以包含没有药性的标签的中药)对应的连边和特征向量。

import Data_input as di
import networkx as nx
import numpy as np
import pandas as pd

features = ['REN', 'PLA2G4A', 'HRAS', 'ARF4', 'FABB', 'NR0B1', 'TRPV1', 'ARALAR1', 'PFKFB1', 'LOC117005044', 'PAM',
            'LOC106431847', 'CB0940_00177', 'PDXK', 'IKBKB', 'METAP2', 'SPED', 'LOC100802485', 'GNAT1', 'RDH13', 'AZOJ',
            'LOC112575901', 'PDIP_77150', 'CACNA1S', 'GCH1', 'DHRS11', 'SPR', 'RDH11', 'CHDH', 'ARGINASE', ]

def herb_vector_score():
    herb_gene_etcm_dict = di.herb_tareget_from_etcm()
    herb_mol_etcm_dict = di.herb_mol_from_etcm()

    filepath = 'herb_class/'
    filename = 'TCMSP_DB_加工.xlsx'
    herb_mol_targets = di.herb_mol_targets(filepath, filename) # 计算每种中药对应的成分和靶点
    h_t = herb_mol_targets[['herb_cn_name','TARGET_ID']]
    h_m = herb_mol_targets[['herb_cn_name','molecule_name']]
    herbs = list(herb_mol_targets['herb_cn_name'].drop_duplicates())
    #合并etcm的herb
    herbs.extend(list(herb_gene_etcm_dict.keys()))
    t_s_d = di.targetid_SYMBOL_dict()  # 数据库中靶点数据和对应的geneid

    #合并etcm和tcmsp数据库的靶点和成分：
    herb_target_dict = {}
    herb_mol_dict = {}
    for herb in herbs :
        if herb not in herb_target_dict:
            #合并靶点
            tmp_genes = []
            h_t_tmp = h_t[h_t['herb_cn_name'] == herb]
            h_m_tmp = h_m[h_m['herb_cn_name'] == herb]

            gene_tmp = list(set(h_t_tmp['TARGET_ID']))
            mol_tmp = list(set(h_m_tmp['molecule_name']))
            for g in gene_tmp:
                if str(g).upper() in t_s_d:
                    tmp_genes.append(str(t_s_d[g]).upper())
            if herb in herb_gene_etcm_dict:
                genes = herb_gene_etcm_dict[herb]
                genes_tmp = genes.split(';')
                for g in genes_tmp:
                    tmp_genes.append(g.upper())
            tmp_genes = list(set(tmp_genes))
            herb_target_dict[herb] = tmp_genes

            tmp_mols = []
            for m in mol_tmp:
                tmp_mols.append(str(m).upper())
            if herb in herb_mol_etcm_dict:
                mols = herb_mol_etcm_dict[herb]
                mols_tmp = mols.split( ';' )
                for m in mols_tmp:
                    tmp_mols.append(m.upper())
            tmp_mols = list(set(tmp_mols))
            herb_mol_dict[herb] = tmp_mols

    return  herb_target_dict,herb_mol_dict


#返回具有寒热温凉属性中药涉及的靶点
def property_herb_targets(herb_vector_dict):
    herb_gene_dict_etcm = di.herb_tareget_from_etcm()
    targets = []
    for (k,v) in herb_gene_dict_etcm.items():
        targets.extend(herb_vector_dict[k])
    targets = list(set(targets))
    return targets

#读取中药的热 温 平 凉 寒等
def herb_Property():
    filepath = 'herb_class/'
    filename = 'herb_etcm.csv'
    herb_property_dict = {}
    with open(filepath + filename,encoding='utf-8') as f:
        for line in f:
            lines = line.split('\t')
            if (lines[2].strip()!='平'):
                herb_property_dict[lines[0]] = lines[2].strip()
    return herb_property_dict

#生成中药配伍列表,放到文本cora.cites里面。
def herb_pair_data(herb_property_dict):
    filepath = 'herb_class/'
    filename = 'herbpair.csv'
    herb_pair = {}
    with open(filepath + filename) as fl:
        for line in fl:
            lines = line.strip().split('\t')
            herb_pair[str(lines[0]) + str(lines[1])] = int(lines[2])

    herbs = list(herb_property_dict.keys())
    herb_pairs_wr = 'cora.cites'

    with open(herb_pairs_wr,'a') as f:
        for i in range(0, len(herbs)-1):
            for j in range(i+1,len(herbs)):
                if (str(herbs[i]) + str(herbs[j])) in herb_pair :#or (str(herbs[j]) + str(herbs[i])) in herb_pair:
                    #设定阈值
                    if herb_property_dict[herbs[i]]!='' and herb_property_dict[herbs[j]]!='' and herb_pair[str(herbs[i]) + str(herbs[j])] > 2:
                    #if herb_pair[str(herbs[i]) + str(herbs[j])] > 2:
                        f.write(str(i))
                        f.write('\t')
                        f.write(str(j))
                        #f.write( '\t' )
                        #f.write(str(herb_pair[str(herbs[i]) + str(herbs[j])]))
                        f.write('\n')

#生成cora.content文件
def herb_vector(herb_property_dict,herb_vector_dict):
    herbs = list(herb_property_dict.keys())

    #修改为全量中药数据
    #herbs = list(herb_vector_dict.keys())
    #修改为全量中药数据

    with open('cora.content','a') as f:
        for i in range(len(herbs)):
            if herbs[i] in herb_vector_dict and  herb_property_dict[herbs[i]]!='':
            # 修改为全量中药数据
            #if herbs[i] in herb_vector_dict:  # and  herb_property_dict[herbs[i]]!='':
            # 修改为全量中药数据
                f.write(str(i))
                f.write('\t')
                for t in features:
                    if t in herb_vector_dict[herbs[i]]:
                        f.write('1')
                    else:
                        f.write('0')
                    f.write('\t')
                if herbs[i] in herb_property_dict:
                    f.write(herb_property_dict[herbs[i]])
                f.write('\n')

herb_target_dict,herb_mol_dict = herb_vector_score()
herb_property_dict =  herb_Property()
herb_pair_data(herb_property_dict)
herb_vector(herb_property_dict,herb_target_dict)
