import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# phenotype is our indices of interest.
# 0 - not of interest, 1 - is of interest.
# all 0 equals to all are of interest
def read_fam(fam_file_path):
    df = pd.read_csv(fam_file_path, delimiter=' ', dtype=str,
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


def write_fam(fam_file_path, rel, sex, indices):
    sex = sex.astype(np.bool)
    individuals, parents = rel.nonzero()

    fathers = np.zeros((rel.shape[0]), dtype=np.int32)
    mothers = np.zeros((rel.shape[0]), dtype=np.int32)
    interest = np.zeros((rel.shape[0]), dtype=np.int32)
    # need to add one so 0 is not an entry
    fathers[individuals[sex[parents]]] = parents[sex[parents]] + 1
    mothers[individuals[~sex[parents]]] = parents[~sex[parents]] + 1
    if indices is not None:
        interest[indices] = True
    content = np.vstack((np.zeros((rel.shape[0]), dtype=np.int32),
                         np.arange(rel.shape[0]) + 1,
                         fathers, mothers,
                         ['2' if x else '1' for x in sex],
                         interest))
    np.savetxt(fam_file_path, content.T, delimiter=' ', fmt='%s')


if __name__ == "__main__":
    from Simulation.Pedigree import simulate_tree
    rel, sex, _ = simulate_tree(10000, 0.001, 1.4, 0.9)
    indices = np.array([1,2,3])
    write_fam('temp.fam', rel, sex, indices)

    rel_after, sex_after, interest_after = read_fam('temp.fam')
    assert (rel_after - rel).nnz == 0
    assert np.count_nonzero(sex - sex_after) == 0
    assert np.count_nonzero(interest_after - indices) == 0
