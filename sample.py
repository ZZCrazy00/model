import numpy as np
import pandas as pd

dataset_name = "gpcr"
if dataset_name in ["BindingDB", "davis"]:
    adj = pd.read_excel(f"dataset/{dataset_name}/dti_mat.xlsx", header=None).values
    drugs_feature = np.loadtxt(f"dataset/{dataset_name}/drugs_feature.txt", delimiter=",")
    targets_feature = np.loadtxt(f"dataset/{dataset_name}/targets_feature.txt", delimiter=",")
else:   # YAM
    adj = pd.read_csv("dataset/YAM/{}/{}_admat_dgc.txt".format(dataset_name, dataset_name), sep="\t",
                      index_col=0).values.tolist()
    drugs_feature = pd.read_csv("dataset/YAM/{}/{}_simmat_dg.txt".format(dataset_name, dataset_name), sep="\t",
                                index_col=0).values.tolist()    # 664 95 204 26
    targets_feature = pd.read_csv("dataset/YAM/{}/{}_simmat_dc.txt".format(dataset_name, dataset_name), sep="\t",
                                  index_col=0).values.tolist()  # 445 223 210 54

pos_data = [np.concatenate([[1], np.concatenate([drugs_feature[i][:64], targets_feature[j][:64]])])
            for i in range(len(adj)) for j in range(len(adj[0])) if adj[i][j] == 1]

neg_data = []
while len(neg_data) < len(pos_data):
    x = np.random.randint(len(drugs_feature))
    y = np.random.randint(len(targets_feature))
    if adj[x][y] == 0:
        neg_data.append(np.concatenate([[0], np.concatenate([drugs_feature[x][:64], targets_feature[y][:64]])]))

all_data = np.concatenate([pos_data, neg_data], axis=0)
pd_data = pd.DataFrame(all_data)
# pd_data.to_csv("dataset/{}/sample.txt".format(dataset_name), sep="\t", columns=None, header=None)
pd_data.to_csv("dataset/YAM/{}/sample.txt".format(dataset_name), sep="\t", columns=None, header=None)
