import pandas as pd
import Linear_model
import SVM_model
import IF_model

# train dataset
df = pd.read_csv('df_DB.csv', index_col=0)
cycle = pd.read_csv('cycle_DB.csv', index_col=0)
'''
# train 50 / test 12
df_train = df.loc[df.index[0:77379], ["time", "infusion_vacuum", "infusion_pressure", "status"]]
df_test = df.loc[df.index[77379:96828], ["time", "infusion_vacuum", "infusion_pressure", "status"]]
'''
# train dataset
df_train = df.loc[df.index[:], ["time", "infusion_vacuum", "infusion_pressure", "status"]]

# test dataset
df_test = pd.read_csv('df_test.csv', index_col=0)
df_test = df_test.loc[df_test.index[0:3996], ["time", "infusion_vacuum", "infusion_pressure", "status"]]
cycle_test = pd.read_csv('cycle_test.csv', index_col=0)

# 抽真空
df_train_V = df_train.loc[df_train["status"] == 1]
df_test_V = df_test.loc[df_test["status"] == 1]

X_train_V = df_train_V["time"]
Y_train_V = df_train_V["infusion_vacuum"]

X_test_V = df_test_V["time"]
Y_test_V = df_test_V["infusion_vacuum"]

Linear_model.model(X_train_V, Y_train_V, X_test_V, Y_test_V)

# 加壓
df_train_P = df_train.loc[df_train["status"] == 5]
df_test_P = df_test.loc[df_test["status"] == 5]

X_train_P = df_train_P["time"]
Y_train_P = df_train_P["infusion_pressure"]

X_test_P = df_test_P["time"]
Y_test_P = df_test_P["infusion_pressure"]

Linear_model.model(X_train_P, Y_train_P, X_test_P, Y_test_P)

# 維持真空

X_train = cycle.loc[cycle.index[:], ["v_max", "v_std", "vd_max", "vd_min", "vd_std"]].values
X_test = cycle_test.loc[cycle_test.index[:], ["v_max", "v_std", "vd_max", "vd_min", "vd_std"]].values

v_pred_train, v_pred_test = SVM_model.model_OC(X_train, X_test)

cycle.loc[cycle.index[:], ["V_pass"]] = v_pred_train
cycle_test.loc[cycle_test.index[:], ["V_pass"]] = v_pred_test

# 維持壓力

X_train = cycle.loc[cycle.index[:], ["p_std", "pd_max", "pd_min", "pd_std"]].values
X_test = cycle_test.loc[cycle_test.index[:], ["p_std", "pd_max", "pd_min", "pd_std"]].values

p_pred_train, p_pred_test = SVM_model.model_OC(X_train, X_test)

cycle.loc[cycle.index[:], ["P_pass"]] = p_pred_train
cycle_test.loc[cycle_test.index[:], ["P_pass"]] = p_pred_test

cycle.to_csv("cycle_DB_1.csv")
cycle_test.to_csv("cycle_test_1.csv")
# SVM_model.model_SVC(X_train, Y_train, X_test)

