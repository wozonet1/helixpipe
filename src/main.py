import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F
import argparse
import pickle as pkl  # noqa: F401
from model import DNN
from train import aver
from utils import sparse_mx_to_torch_sparse_tensor, aug_random_walk, load_data
from tqdm import tqdm
from datetime import datetime
# TODO: use mlflow to track experiments

parser = argparse.ArgumentParser(description="RUN TRAINING")
parser.add_argument("--device", default="cuda:1", type=str, help="Device of Training")
parser.add_argument(
    "--dataset",
    default="DrugBank",
    type=str,
    help="Dataset to use",
    choices=["DrugBank1.4", "bindingdb", "biosnap", "DrugBank", "davis", "human"],
)
parser.add_argument("--seed", type=int, default=514, help="Random seed.")
parser.add_argument("--logfilename", type=str, default="log", help="Log file name.")
parser.add_argument(
    "--msg", type=str, default="", help="Message to record in log file."
)


cmdargs = parser.parse_args()

pd.set_option("display.max_rows", 10)

nodes_df = pd.read_csv("../data/DrugBank/nodes.csv").to_csv(
    "../data/DrugBank/Allnode_DrPr.csv", header=None, index=False
)

drug_ids = nodes_df[nodes_df["node_type"] == "drug"]["node_id"].tolist()
ligand_ids = nodes_df[nodes_df["node_type"] == "ligand"]["node_id"].tolist()
protein_ids = nodes_df[nodes_df["node_type"] == "protein"]["node_id"].tolist()

drug_num = len(drug_ids)
ligand_num = len(ligand_ids)
protein_num = len(protein_ids)
total_nodes = len(nodes_df)

print(f"Start Running on Dataset: {cmdargs.dataset}\n{cmdargs.msg}\n")

AllNode = pd.read_csv(
    f"../data/{cmdargs.dataset}/Allnode_DrPr.csv", names=[0, 1], skiprows=1
)
Alledge = pd.read_csv(f"../data/{cmdargs.dataset}/DrPrNum_DrPr.csv", header=None)
drug_prot_edge = pd.read_csv(f"../data/{cmdargs.dataset}/drug_prot_edge.csv")
features = pd.read_csv(
    f"../data/{cmdargs.dataset}/AllNodeAttribute_DrPr.csv", header=None
)
# features = features.iloc[:, 1:]

labels = pd.DataFrame(np.random.rand(len(AllNode), 1))
labels[:drug_num] = 0
labels[drug_num:] = 1
labels = labels[0]

adj, features, labels, idx_train, idx_val, idx_test = load_data(
    drug_prot_edge, features, labels
)


# set parameter
class item:
    def __init__(self):
        self.epochs = 300
        self.lr = 1e-1
        self.k1 = 200
        self.k2 = 10
        self.epsilon1 = 0.03
        self.epsilon2 = 0.05
        self.hidden = 64
        self.dropout = 0.8
        self.runs = 1


args = item()

node_sum = adj.shape[0]
edge_sum = adj.sum() / 2
row_sum = adj.sum(1) + 1
norm_a_inf = row_sum / (2 * edge_sum + node_sum)

adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))

features = F.normalize(features, p=1)
feature_list = []
feature_list.append(features)
for i in range(1, args.k1):
    feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
norm_fea_inf = torch.mm(norm_a_inf, features)

hops = torch.Tensor([0] * (adj.shape[0]))
mask_before = torch.Tensor([False] * (adj.shape[0])).bool()

for i in range(args.k1):
    dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
    mask = (dist < args.epsilon1).masked_fill_(mask_before, False)
    mask_before.masked_fill_(mask, True)
    hops.masked_fill_(mask, i)
mask_final = torch.Tensor([True] * (adj.shape[0])).bool()
mask_final.masked_fill_(mask_before, False)
hops.masked_fill_(mask_final, args.k1 - 1)
print("Local Smoothing Iteration calculation is done.")
input_feature = aver(hops, adj, feature_list)
print("Local Smoothing is done.")

device = torch.device(cmdargs.device if torch.cuda.is_available() else "cpu")

n_class = 64

input_feature = input_feature.to(device)

print("Start training...")

model = DNN(features.shape[1], args.hidden, n_class, args.dropout).to(device)
model.eval()
output, Emdebding = model(input_feature)  # .cpu()
Emdebding_GCN = pd.DataFrame(Emdebding.detach().cpu().numpy())
Emdebding_GCN.to_csv(
    f"../data/{cmdargs.dataset}/GCN_Node_Embeddings.csv", header=None, index=False
)

Positive = Alledge
AllNegative = pd.read_csv(
    f"../data/{cmdargs.dataset}/AllNegative_DrPr.csv", header=None
)
n_num = len(AllNegative)
print(f"positive edge num: {len(Positive)}, negative edge num: {len(AllNegative)}")
with open(f"../logs/main/{cmdargs.dataset}_{cmdargs.logfilename}.log", "a") as f:
    f.write(
        f"Dataset: {cmdargs.dataset}, with seed {cmdargs.seed},\
            starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\
            {cmdargs.msg}\n"
    )
Negative = AllNegative.sample(n=n_num, random_state=520)
Positive[2] = Positive.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive, Negative]).reset_index(drop=True)
X = pd.concat(
    [
        Emdebding_GCN.loc[result[0].values.tolist()].reset_index(drop=True),
        Emdebding_GCN.loc[result[1].values.tolist()].reset_index(drop=True),
    ],
    axis=1,
)
Y = result[2]

NmedEdge = 499  # 499
DmedEdge = 7  # 7
SmedEdge = 0.85  # 0.85
k_fold = 5
print("%d fold CV" % k_fold)
i = 1
tprs = []
aucs = []
auprs = []
f1s = []
accuracies = []
sensitivities = []
specificities = []
mean_fpr = np.linspace(0, 1, 1000)
AllResult = []

best_auc = 0

skf = StratifiedKFold(n_splits=k_fold, random_state=cmdargs.seed, shuffle=True)
for train_index, test_index in tqdm(
    skf.split(X, Y), total=k_fold, desc=f"{k_fold}-Fold CV"
):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = GradientBoostingClassifier(
        n_estimators=NmedEdge,
        max_depth=DmedEdge,
        subsample=SmedEdge,
        learning_rate=0.1,
        verbose=1,
    )

    model.fit(np.array(X_train), np.array(Y_train))
    y_score0 = model.predict(np.array(X_test))
    y_score_RandomF = model.predict_proba(np.array(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test, y_score_RandomF[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    roc_auc = auc(fpr, tpr)
    if roc_auc > best_auc:
        best_auc = roc_auc
        torch.save(model, "../model_args/n_best_GBDT.pt")
    aucs.append(roc_auc)

    aupr = average_precision_score(Y_test, y_score_RandomF[:, 1])
    auprs.append(float(aupr))

    print("ROC fold %d(AUC=%0.4f, AUPR=%0.4f)" % (i, roc_auc, float(aupr)))
    with open(f"../logs/main/{cmdargs.dataset}_{cmdargs.logfilename}.log", "a") as f:
        f.write(f"{cmdargs.dataset}:fold {i},AUC={roc_auc},AUPR={float(aupr)}\n")
    i += 1


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print("Mean ROC (AUC=%0.4f, AUPR=%0.4f)" % (mean_auc, sum(auprs) / len(auprs)))

with open(f"../logs/main/{cmdargs.dataset}_{cmdargs.logfilename}.log", "a") as f:
    f.write(
        "%s: Mean ROC (AUC=%0.4f, AUPR=%0.4f)\nEnds now\n"
        % (cmdargs.dataset, mean_auc, sum(auprs) / len(auprs))
    )
