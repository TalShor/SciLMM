import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# phenotype is our indices of interest.
# 0 - not of interest, 1 - is of interest.
# all 0 equals to all are of interest
def translate_fam(file_path):
    df = pd.read_csv(file_path, delimiter=' ', dtype=str,
                     names=['FID', 'IID', 'F_IID', 'M_IID', 'sex', 'phenotype'])

    # add the family id to all the ids (avoids duplicates)
    df['F_IID'][df['F_IID'] != '0'] = \
        df["FID"][df['F_IID'] != '0'].map(str) + "_" + \
        df['F_IID'][df['F_IID'] != '0']
    df['M_IID'][df['M_IID'] != '0'] = \
        df["FID"][df['M_IID'] != '0'].map(str) + "_" + \
        df['M_IID'][df['M_IID'] != '0']
    df['IID'] = df["FID"].map(str) + "_" + df['IID']

    entries = {id: i for i, id in enumerate(df['IID'].values)}
    all_ids = np.array(list(entries.keys()))

    # get all parent-child edges
    child_father = np.array([[entries[child], entries[father]] for child, father in
                             df[['IID', 'F_IID']][df['F_IID'] != '0'].values])
    child_mother = np.array([[entries[child], entries[mother]] for child, mother in
                             df[['IID', 'M_IID']][df['M_IID'] != '0'].values])
    all_co = np.vstack((child_father, child_mother))

    # create the relationship matrix
    rel = csr_matrix((np.ones(all_co.shape[0]), (all_co[:, 0], all_co[:, 1])),
                     shape=(all_ids.size, all_ids.size),
                     dtype=np.bool)

    # extra data
    sex = df['sex'] == '2'
    interest = np.where(df['phenotype'] == '1')[0]
    if interest.size == 0:
        interest = None

    return rel, sex, interest

print translate_fam('temp.fam')