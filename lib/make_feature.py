# -*- coding: utf-8 -*
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from processing import *
from sklearn.feature_selection import SelectFromModel
__ATTR_NAMES = ("duration",  # length (number of seconds) of the conn's
                "protocol_type",  # symbolic, type of the protocol, e.g. tcp, udp, etc.
                "service",  # symbolic, network service on the destination, e.g., http, telnet, etc.
                "flag",  # symbolic, normal or error status of the conn
                "src_bytes",  # number of data bytes from source to destination
                "dst_bytes",  # number of data bytes from destination to source
                "land",  # symbolic, 1 if conn is from/to the same host/port; 0 otherwise
                "wrong_fragment",  # number of ''wrong'' fragments 
                "urgent",  # number of urgent packets
                # ----------
                # ----- Basic features of individual TCP conn's -----
                # ----------
                "hot",  # number of ''hot'' indicators
                "num_failed_logins",  # number of failed login attempts 
                "logged_in",  # symbolic, 1 if successfully logged in; 0 otherwise
                "num_compromised",  # number of ''compromised'' conditions 
                "root_shell",  # 1 if root shell is obtained; 0 otherwise 
                "su_attempted",  # 1 if ''su root'' command attempted; 0 otherwise 
                "num_root",  # number of ''root'' accesses 
                "num_file_creations",  # number of file creation operations
                "num_shells",  # number of shell prompts 
                "num_access_files",  # number of operations on access control files
                "num_outbound_cmds",  # number of outbound commands in an ftp session 
                "is_host_login",  # symbolic, 1 if the login belongs to the ''hot'' list; 0 otherwise 
                "is_guest_login",  # symbolic, 1 if the login is a ''guest''login; 0 otherwise 
                # ----------
                # ----- Content features within a conn suggested by domain knowledge -----
                # ----------
                "count",  # number of conn's to the same host as the current conn in the past two seconds 
                # Time-based Traffic Features (examine only the conn in the past two seconds):
                # 1. Same Host, have the same destination host as the current conn
                # 2. Same Service, have the same service as the current conn.
                "srv_count",  # SH, number of conn's to the same service as the current conn
                "serror_rate",  # SH, % of conn's that have SYN errors
                "srv_serror_rate",  # SS, % of conn's that have SYN errors
                "rerror_rate",  # SH, % of conn's that have REJ errors 
                "srv_rerror_rate",  # SS, % of conn's that have REJ errors 
                "same_srv_rate",  # SH, % of conn's to the same service 
                "diff_srv_rate",  # SH, % of conn's to different services 
                "srv_diff_host_rate",  # SH,  % of conn's to different hosts 
                # ----------
                # Host-base Traffic Features, constructed using a window of 100 conn's to the same host
                "dst_host_count",
                "dst_host_srv_count",
                "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate",
                "dst_host_srv_serror_rate",
                "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate",
                # ----------
                # category
                "attack_type"
                )

df = pd.read_csv(r'../data/train10pc', header=None, names=__ATTR_NAMES)
# sparse feature merge
df = merge_sparse_feature(df)
# one hot encoding
df = one_hot(df)
# y labels mapping
df = map2major5(df)
print("data loaded")

y = df["attack_type"]
X = df.drop("attack_type", axis=1)

"""
After several experiments, it is indicated that 40+ features has major importance to this task
Select features that appear in the top 50 in all 10 trainings will yield 40+ features
"""
# first training
selected_feat_names = set()
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(X, y)
print("training finished")

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]  # descending order
for f in range(X.shape[1]):
    if f < 50:
        selected_feat_names.add(X.columns[indices[f]])
    print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))

for i in range(9):
    tmp = set()
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(X, y)
    print("training finished")

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]  # descending order
    for f in range(X.shape[1]):
        if f < 50:  # need roughly more than 40 features according to experiments
            tmp.add(X.columns[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))

    selected_feat_names &= tmp
    print(len(selected_feat_names), "features are selected")

with open(r'../data/selected_feat_names.pkl', 'wb') as f:
    pickle.dump(list(selected_feat_names), f)