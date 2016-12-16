# load packages
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# load data
train = pd.read_csv("../Data/blogData_train.csv",header=None)
test = pd.read_csv("../Data/blogData_test.csv",header=None)

# rename
names = ["V%i" % i if i != 281 else "y" for i in range(1,282)]
train.columns = names
test.columns = names

# split to features and target varibles
X_train = train.ix[:,"V1":"V280"]
y_train = train["y"]

X_test = test.ix[:,"V1":"V280"]
y_test = test["y"]

y_train_log = np.log(train["y"]+1)
y_test_log = np.log(test["y"]+1)

# Loss evaluation function
def eva(test_pred, test_real):
    return ((test_real-test_pred)**2).mean()


# Feature Engineering
# Define group 
group = [[1,6,11,16,21],
        [2,7,12,17,22],
        [3,8,13,18,23],
        [4,9,14,19,24],
        [5,10,15,20,25],
        [26,31,36,41,46],
        [27,32,37,42,47],
        [28,33,38,43,48],
        [29,34,39,44,49],
        [30,35,40,45,50],
        [51,52,53,54,55],
        [56,57,58,59,60],
        [61,62],
        list(range(63,263)),
        [263,264,265,266,267,268,269],
        [270,271,272,273,274,275,276],
        [277,278,279,280]]

# Drop features 
drop_group = [5,7,6,9,1,2,8,13]
remove_col = []
for i in drop_group:
    remove_col += group[i]
remove_col = ['V'+str(x) for x in remove_col]
newXTrain = X_train.drop(remove_col, axis=1)
newXTest = X_test.drop(remove_col, axis=1)

# Generating new features

# 52/51 this variable is not selected
# N1_train = newXTrain["V52"].as_matrix()/newXTrain["V51"].as_matrix()
# N1_test = newXTest["V52"].as_matrix()/newXTest["V51"].as_matrix()
# newXTrain["N1"] = N1_train
# newXTest["N1"] = N1_test

# V53/V51
N2_train = newXTrain["V53"].as_matrix()/newXTrain["V51"].as_matrix()
N2_test = newXTest["V53"].as_matrix()/newXTest["V51"].as_matrix()
newXTrain["N2"] = N2_train
newXTest["N2"] = N2_test

# V54/V51
N3_train = newXTrain["V54"].as_matrix()/newXTrain["V51"].as_matrix()
N3_test = newXTest["V54"].as_matrix()/newXTest["V51"].as_matrix()
newXTrain["N3"] = N3_train
newXTest["N3"] = N3_test

# V52/V53
N4_train = newXTrain["V53"].as_matrix()/newXTrain["V52"].as_matrix()
N4_test = newXTest["V53"].as_matrix()/newXTest["V52"].as_matrix()
newXTrain["N4"] = N4_train
newXTest["N4"] = N4_test

# V54/V51
N5_train = newXTrain["V52"].as_matrix()/newXTrain["V54"].as_matrix()
N5_test = newXTest["V52"].as_matrix()/newXTest["V54"].as_matrix()
newXTrain["N5"] = N5_train
newXTest["N5"] = N5_test

# V57/V56
N6_train = newXTrain["V57"].as_matrix()/newXTrain["V56"].as_matrix()
N6_test = newXTest["V57"].as_matrix()/newXTest["V56"].as_matrix()
newXTrain["N6"] = N6_train
newXTest["N6"] = N6_test

# V58/V56
N7_train = newXTrain["V58"].as_matrix()/newXTrain["V56"].as_matrix()
N7_test = newXTest["V58"].as_matrix()/newXTest["V56"].as_matrix()
newXTrain["N7"] = N7_train
newXTest["N7"] = N7_test

# V59/V56
N8_train = newXTrain["V59"].as_matrix()/newXTrain["V56"].as_matrix()
N8_test = newXTest["V59"].as_matrix()/newXTest["V56"].as_matrix()
newXTrain["N8"] = N8_train
newXTest["N8"] = N8_test

# V57/V58
N9_train = newXTrain["V57"].as_matrix()/newXTrain["V58"].as_matrix()
N9_test = newXTest["V57"].as_matrix()/newXTest["V58"].as_matrix()
newXTrain["N9"] = N9_train
newXTest["N9"] = N9_test

# V59/V56
N10_train = newXTrain["V59"].as_matrix()/newXTrain["V56"].as_matrix()
N10_test = newXTest["V59"].as_matrix()/newXTest["V56"].as_matrix()
newXTrain["N10"] = N10_train
newXTest["N10"] = N10_test

# Binning V61
N11_train = (newXTrain["V61"].as_matrix()>24)*1
N11_test = (newXTest["V61"].as_matrix()>24)*1
newXTrain["N11"] = N11_train
newXTest["N11"] = N11_test

# V62/V61
N12_train = newXTrain["V62"].as_matrix()/newXTrain["V61"].as_matrix()
N12_test = newXTest["V62"].as_matrix()/newXTest["V61"].as_matrix()
newXTrain["N12"] = N12_train
newXTest["N12"] = N12_test

# Get a binary varible indicating whether the publication day is at weekend 
# 1 -> in weekend, 0 -> not in weekend
pubWeekendTrain = newXTrain.ix[:,"V268":"V269"].apply(lambda x:x.sum(),axis = 1)
pubWeekendTest = newXTest.ix[:,"V268":"V269"].apply(lambda x:x.sum(),axis = 1)

# Get a binary varible indicating whether the basement day is at weekend 
# 1 -> in weekend, 0 -> not in weekend
bsWeekendTrain = newXTrain.ix[:,"V275":"V276"].apply(lambda x:x.sum(),axis = 1)
bsWeekendTest = newXTest.ix[:,"V275":"V276"].apply(lambda x:x.sum(),axis = 1)

# Combine the previous to varible into one dataframe
pubBsDayTrain = pd.concat([pubWeekendTrain,bsWeekendTrain],axis=1)
pubBsDayTest = pd.concat([pubWeekendTest,bsWeekendTest],axis=1)

# Define for patterns of the pubWeekend and bsWeekend as (1,0),(0,1),(1,1),(0,0)
N13_train = pubBsDayTrain.apply(lambda x: ((x[0]==1) & (x[1]==1))*1, axis=1)
N14_train = pubBsDayTrain.apply(lambda x: ((x[0]==1) & (x[1]==0))*1, axis=1)
N15_train = pubBsDayTrain.apply(lambda x: ((x[0]==0) & (x[1]==1))*1, axis=1)
N16_train = pubBsDayTrain.apply(lambda x: ((x[0]==0) & (x[1]==0))*1, axis=1)

N13_test = pubBsDayTest.apply(lambda x: ((x[0]==1) & (x[1]==1))*1, axis=1)
N14_test = pubBsDayTest.apply(lambda x: ((x[0]==1) & (x[1]==0))*1, axis=1)
N15_test = pubBsDayTest.apply(lambda x: ((x[0]==0) & (x[1]==1))*1, axis=1)
N16_test = pubBsDayTest.apply(lambda x: ((x[0]==0) & (x[1]==0))*1, axis=1)

# Adding new variables
newXTrain["N13"] = N13_train
newXTest["N13"] = N13_test

newXTrain["N14"] = N14_train
newXTest["N14"] = N14_test

newXTrain["N15"] = N15_train
newXTest["N15"] = N15_test

newXTrain["N16"] = N16_train
newXTest["N16"] = N16_test

# Correct the na value and the inf values occuring during the feature creation
newXTrain = newXTrain.fillna(-1)
newXTest = newXTest.fillna(-1)

newXTrain = newXTrain.replace([np.inf, -np.inf], 10000)
newXTest = newXTest.replace([np.inf, -np.inf], 10000)

# Creating the interaction items from the add_interaction.txt file, which contains the 
# names of variables involved in the interaction
f = open("../Data/add_interaction.txt", "r")
for inter_var in f.readlines():
    inter_var = inter_var.strip()
    new_col_name = inter_var
    inter_var = inter_var.split('_')
    # add new column for train data
    X_train_new_col =newXTrain[inter_var[0]]
    for var in inter_var[1:]:
        X_train_new_col = X_train_new_col * newXTrain[var]
    newXTrain[new_col_name] = X_train_new_col
    # add new column for test data
    X_test_new_col = newXTest[inter_var[0]]
    for var in inter_var[1:]:
        X_test_new_col = X_test_new_col * newXTest[var]
    newXTest[new_col_name] = X_test_new_col
f.close()


# Ensemble model

# Creat cv-folds
nfolds = 8
folds = KFold(len(y_train_log), n_folds = nfolds, shuffle = True, random_state = 42)

model_group = []
pred = []
i = 1

# Train and ensemble the models
for (Tr, Te) in folds:
    train_x = newXTrain.ix[Tr,:]
    train_y = y_train_log[Tr]
    test_x = newXTrain.ix[Te,:]
    test_y = y_train_log[Te]
    
    # parameter settings
    params = {
    'min_child_weight': 1,
    'eta': 0.01,
    'colsample_bytree': 1,
    'max_depth': 12,
    'subsample': 0.2,
    'reg_alpha': 1,
    'gamma': 0.04,
    'silent':True,
    "eval_metric":"rmse"}
    
    # model training on 7 sections and evaluation on the left section
    xgtrain = xgb.DMatrix(train_x, label=train_y)
    xgtest = xgb.DMatrix(newXTest)
    xgval = xgb.DMatrix(test_x, label=test_y)
    print("The model %s is training......" % i)
    gb_model = xgb.train(params, 
                     dtrain=xgtrain, 
                     verbose_eval = 50,
                     evals=[(xgval,"validation")], 
                     early_stopping_rounds = 30,
                     num_boost_round = 2000)

    # model prediction
    gb_pred = gb_model.predict(xgtest)

    # store the predicted results for each model and the models
    pred.append(gb_pred)
    model_group.append(gb_model)
    i += 1

# Ensemble the results and print the final results
gb_pred = np.array(pred).mean(0)
print("The final loss value from the ensemble model is: ", eva(gb_pred,y_test_log))









