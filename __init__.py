import pandas as pd
import numpy as np
from os.path import isfile
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import kpss
import warnings

warnings.filterwarnings("ignore")


class ML_Fraud:
    __version__ = '1.1'

    def __init__(self, sample_start=1991, test_sample=range(2001, 2011),
                 OOS_per=1, OOS_gap=0, sampling='expanding', adjust_serial=True,
                 cv_type='kfold', temp_year=1, cv_flag=False, cv_k=10, write=True, IS_per=10):
        """
        Parameters:
            – sample_start: Calendar year marking the start of the sample (Default=1991)
            – test_sample: testing/out-of-sample period(Default=range(2001,2011))
            – OOS_per: out-of-sample rolling period in years (Default=OOS_per=1)
            – OOS_gap: Gap between training and testing samples in year (Default=0)
            – sampling: sampling style either "expanding"/"rolling" (Default="expanding")
            – adjust_serial: A boolean variable to adjust for serial frauds (Default=True)
            – cv_type: A string to determine whether to do a temporal or k-fold cv
            – cv_flag: A boolean variable whether to replicate the cross-validation (Default=False)
            – cv_k: The number of folds (k) in the cross-validation (Default=10)
            – write: A boolean variable whether to write results into csv files (Default=True)
            – IS_per: Number of calendar years in case a rolling training sample is used (Default=10)

        """
        if isfile('FraudDB2020.csv') == False:
            df = pd.DataFrame()
            for s in range(1, 5):
                fl_name = 'FraudDB2020_Part' + str(s) + '.csv'
                new_df = pd.read_csv(fl_name)
                df = df.append(new_df)
            df.to_csv('FraudDB2020.csv', index=False)

        df = pd.read_csv('FraudDB2020.csv')
        self.df = df
        self.ss = sample_start
        self.se = np.max(df.fyear)
        self.ts = test_sample
        self.cv_t = cv_type
        self.cv = cv_flag
        self.cv_k = cv_k
        self.cv_t_y = temp_year

        sampling_set = ['expanding', 'rolling']
        if sampling in sampling_set:
            pass
        else:
            raise ValueError('Invalid sampling choice. Permitted options are "expanding" and "rolling"')

        self.sa = sampling
        self.w = write
        self.ip = IS_per
        self.op = OOS_per
        self.og = OOS_gap
        self.a_s = adjust_serial
        print('Module initiated successfully ...')


    def ratio_analyse(self, C_FN=30, C_FP=1):
        """
        This code uses 11 financial ratios for Table 5. RUSBoost and SVM-FK23 are not included.
        Combined with raw_analyse Table 7 can be reproduced.

        Methodological choices:
            - 11 financial ratios from Dechow et al.(2011)
            - 10-fold validation
            - Bertomeu et al. (2021)'s serial fraud treatment

        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM

        Predictive models:
            – Support Vector Machine (SVM)
            – Logistic Regression (LR)
            – Adaptive Boosting with Logistic Regression/LogitBoost (LogitBoost)
            – FUSED (weighted average of estimated probs of other methods)

        Outputs:
        Main results are stored in the table variable "perf_tbl_general" written into

        Steps:
            1. Cross-validate to find optimal hyperparameters.
            2. Estimating the performance for each OOS period.


        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, -14:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]

        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -11:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        # redo cross-validation if you wish
        if cv_type == 'kfold':
            if cross_val == True:

                print('Grid search hyperparameter optimisation started for LogitBoost')
                t1 = datetime.now()

                best_perf_ada = 0

                base_lr = LogisticRegression(random_state=0, solver='newton-cg')
                estimators = list(range(10,3001,10))
                learning_rates = [x/1000 for x in range(10,1001,10)]
                pipe_ada = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_ada', AdaBoostClassifier(estimator=base_lr, random_state=0))])
                param_grid_ada = {'base_mdl_ada__n_estimators': estimators, \
                                  'base_mdl_ada__learning_rate': learning_rates}

                clf_ada = GridSearchCV(pipe_ada, param_grid_ada, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_ada.fit(X_CV, Y_CV)
                score_ada = clf_ada.best_score_
                if score_ada >= best_perf_ada:
                    best_perf_ada = score_ada
                    opt_params_ada = clf_ada.best_params_

                print('LogitBoost: The optimal number of estimators is ' + \
                      str(opt_params_ada['base_mdl_ada__n_estimators']) + ', and learning rate is' + \
                      str(opt_params_ada['base_mdl_ada__learning_rate']) + ', and score is' + \
                      str(score_ada))

                print('Computing CV ROC for LR ...')
                score_lr = []
                for m in range(0, k_fold):
                    train_sample, test_sample = train_test_split(Y_CV, test_size=1 /
                                                                                 k_fold, shuffle=False, random_state=m)
                    X_train = X_CV.iloc[train_sample.index]
                    X_train = add_constant(X_train)
                    Y_train = train_sample
                    X_test = X_CV.iloc[test_sample.index]
                    X_test = add_constant(X_test)
                    Y_test = test_sample

                    logit_model = Logit(Y_train, X_train)
                    logit_model = logit_model.fit(disp=0)
                    pred_LR_CV = logit_model.predict(X_test)
                    score_lr.append(roc_auc_score(Y_test, pred_LR_CV))

                score_lr = np.mean(score_lr)
                print('Logit: The optimal score is ' + str(score_lr))

                # optimize SVM grid

                print('Grid search hyperparameter optimisation started for SVM')
                t1 = datetime.now()

                pipe_svm = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_svm', SVC(shrinking=False, \
                                                          probability=False, random_state=0, max_iter=-1, \
                                                          tol=X_CV.shape[-1] * 1e-3))])

                C = [0.001, 0.01, 0.1, 1, 10, 100]
                gamma = [0.0001, 0.001, 0.01, 0.1]
                kernel = ['rbf', 'linear', 'poly']
                class_weight = [{0: 1/x, 1: 1} for x in range(10,501,10)]
                param_grid_svm = {'base_mdl_svm__kernel': kernel,\
                                  'base_mdl_svm__C': C, \
                                  'base_mdl_svm__gamma': gamma,\
                                  'base_mdl_svm__class_weight':class_weight}


                clf_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_svm.fit(X_CV, Y_CV)
                opt_params_svm = clf_svm.best_params_
                gamma_opt = opt_params_svm['base_mdl_svm__gamma']
                cw_opt = opt_params_svm['base_mdl_svm__class_weight']
                c_opt = opt_params_svm['base_mdl_svm__C']
                kernel_opt = opt_params_svm['base_mdl_svm__kernel']
                score_svm = clf_svm.best_score_

                t2 = datetime.now()
                dt = t2 - t1
                print('SVM CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('SVM: The optimal C+/C- ratio is ' + str(cw_opt) + ', gamma is' + str(
                    gamma_opt) + ', C is' + str(c_opt) + ',score is' + str(score_svm) +\
                      ', kernel is' + str(kernel_opt))


        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)

        roc_svm = np.zeros(len(range_oos))
        roc_svm_training = np.zeros(len(range_oos))
        sensitivity_OOS_svm1 = np.zeros(len(range_oos))
        sensitivity_svm1_training = np.zeros(len(range_oos))
        specificity_OOS_svm1 = np.zeros(len(range_oos))
        specificity_svm1_training = np.zeros(len(range_oos))
        precision_svm1 = np.zeros(len(range_oos))
        precision_svm1_training = np.zeros(len(range_oos))
        ndcg_svm1 = np.zeros(len(range_oos))
        ndcg_svm1_training = np.zeros(len(range_oos))
        ecm_svm1 = np.zeros(len(range_oos))
        ecm_svm1_training = np.zeros(len(range_oos))

        roc_lr = np.zeros(len(range_oos))
        roc_lr_training = np.zeros(len(range_oos))
        sensitivity_OOS_lr1 = np.zeros(len(range_oos))
        sensitivity_lr1_training = np.zeros(len(range_oos))
        specificity_OOS_lr1 = np.zeros(len(range_oos))
        specificity_lr1_training = np.zeros(len(range_oos))
        precision_lr1 = np.zeros(len(range_oos))
        precision_lr1_training = np.zeros(len(range_oos))
        ndcg_lr1 = np.zeros(len(range_oos))
        ndcg_lr1_training = np.zeros(len(range_oos))
        ecm_lr1 = np.zeros(len(range_oos))
        ecm_lr1_training = np.zeros(len(range_oos))


        roc_ada = np.zeros(len(range_oos))
        roc_ada_training = np.zeros(len(range_oos))
        sensitivity_OOS_ada1 = np.zeros(len(range_oos))
        sensitivity_ada1_training = np.zeros(len(range_oos))
        specificity_OOS_ada1 = np.zeros(len(range_oos))
        specificity_ada1_training = np.zeros(len(range_oos))
        precision_ada1 = np.zeros(len(range_oos))
        precision_ada1_training = np.zeros(len(range_oos))
        ndcg_ada1 = np.zeros(len(range_oos))
        ndcg_ada1_training = np.zeros(len(range_oos))
        ecm_ada1 = np.zeros(len(range_oos))
        ecm_ada1_training = np.zeros(len(range_oos))


        roc_fused = np.zeros(len(range_oos))
        roc_fused_training = np.zeros(len(range_oos))
        sensitivity_OOS_fused1 = np.zeros(len(range_oos))
        sensitivity_fused1_training = np.zeros(len(range_oos))
        specificity_OOS_fused1 = np.zeros(len(range_oos))
        specificity_fused1_training = np.zeros(len(range_oos))
        precision_fused1 = np.zeros(len(range_oos))
        precision_fused1_training = np.zeros(len(range_oos))
        ndcg_fused1 = np.zeros(len(range_oos))
        ndcg_fused1_training = np.zeros(len(range_oos))
        ecm_fused1 = np.zeros(len(range_oos))
        ecm_fused1_training = np.zeros(len(range_oos))

        m = 0
        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period

            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]

            print(f'before dropping the number of observations is: {len(tbl_year_OOS)}')

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)
            print(f'after dropping the number of observations is: {len(tbl_year_OOS)}')

            X = tbl_year_IS.iloc[:, -11:]
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS = (X_OOS - mean) / std
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            n_P_training = np.sum(Y == 1)
            n_N_training = np.sum(Y == 0)

            # Support Vector Machines

            clf_svm = SVC(class_weight=opt_params_svm['base_mdl_svm__class_weight'],
                          kernel=opt_params_svm['base_mdl_svm__kernel'], \
                          C=opt_params_svm['base_mdl_svm__C'],\
                          gamma=opt_params_svm['base_mdl_svm__gamma'], shrinking=False, \
                          probability=False, random_state=0, max_iter=-1, \
                          tol=X.shape[-1] * 1e-3)

            clf_svm = clf_svm.fit(X, Y)

            #performance on training sample- SVM
            pred_train_svm = clf_svm.decision_function(X)
            probs_fraud_svm = np.exp(pred_train_svm) / (1 + np.exp(pred_train_svm))
            cutoff_svm = np.percentile(probs_fraud_svm, 99)
            roc_svm_training[m] = roc_auc_score(Y, probs_fraud_svm)
            sensitivity_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                                            Y == 1)) / np.sum(Y)
            specificity_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm < cutoff_svm, \
                                                            Y == 0)) / np.sum(Y == 0)
            precision_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                                      Y == 1)) / np.sum(probs_fraud_svm >= cutoff_svm)
            ndcg_svm1_training[m] = ndcg_k(Y, probs_fraud_svm, 99)
            FN_svm3 = np.sum(np.logical_and(probs_fraud_svm < cutoff_svm, \
                                            Y == 1))
            FP_svm3 = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                            Y == 0))
            ecm_svm1_training[m] = C_FN * P_f * FN_svm3 / n_P_training + C_FP * P_nf * FP_svm3 / n_N_training


            #performance on testing sample- SVM
            pred_test_svm = clf_svm.decision_function(X_OOS)
            probs_oos_fraud_svm = np.exp(pred_test_svm) / (1 + np.exp(pred_test_svm))
            roc_svm[m] = roc_auc_score(Y_OOS, probs_oos_fraud_svm)
            cutoff_OOS_svm = np.percentile(probs_oos_fraud_svm, 99)
            sensitivity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_svm >= cutoff_OOS_svm)
            ndcg_svm1[m] = ndcg_k(Y_OOS, probs_oos_fraud_svm, 99)
            FN_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                            Y_OOS == 1))
            FP_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                            Y_OOS == 0))
            ecm_svm1[m] = C_FN * P_f * FN_svm2 / n_P + C_FP * P_nf * FP_svm2 / n_N

            # Logistic Regression – Dechow et al (2011)
            X_lr = add_constant(X)
            X_OOS_lr = add_constant(X_OOS)
            clf_lr = Logit(Y, X_lr)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS_lr)

            #performance on training sample- logit
            probs_fraud_lr = clf_lr.predict(X_lr)
            cutoff_lr = np.percentile(probs_fraud_lr, 99)
            roc_lr_training[m] = roc_auc_score(Y, probs_fraud_lr)
            sensitivity_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                                           Y == 1)) / np.sum(Y)
            specificity_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr < cutoff_lr, \
                                                           Y == 0)) / np.sum(Y == 0)
            precision_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                                     Y == 1)) / np.sum(probs_fraud_lr >= cutoff_lr)
            ndcg_lr1_training[m] = ndcg_k(Y, probs_fraud_lr, 99)
            FN_lr3 = np.sum(np.logical_and(probs_fraud_lr < cutoff_lr, \
                                           Y == 1))
            FP_lr3 = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                           Y == 0))
            ecm_lr1_training[m] = C_FN * P_f * FN_lr3 / n_P_training + C_FP * P_nf * FP_lr3 / n_N_training

            # performance on testing sample- logit
            roc_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_lr)

            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)
            sensitivity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                           Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                                           Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                     Y_OOS == 1)) / np.sum(probs_oos_fraud_lr >= cutoff_OOS_lr)
            ndcg_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_lr, 99)
            FN_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                           Y_OOS == 1))
            FP_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                           Y_OOS == 0))
            ecm_lr1[m] = C_FN * P_f * FN_lr2 / n_P + C_FP * P_nf * FP_lr2 / n_N


            # LogitBoost
            base_lr = LogisticRegression(random_state=0, solver='newton-cg')

            clf_ada = AdaBoostClassifier(n_estimators=opt_params_ada['base_mdl_ada__n_estimators'], \
                                         learning_rate=opt_params_ada['base_mdl_ada__learning_rate'], \
                                         estimator=base_lr, random_state=0)
            clf_ada = clf_ada.fit(X, Y)
            probs_oos_fraud_ada = clf_ada.predict_proba(X_OOS)[:, -1]


            #performance on training sample- LogitBoost
            probs_fraud_ada = clf_ada.predict_proba(X)[:, -1]
            cutoff_ada = np.percentile(probs_fraud_ada, 99)
            roc_ada_training[m] = roc_auc_score(Y, probs_fraud_ada)
            sensitivity_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                                            Y == 1)) / np.sum(Y)
            specificity_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada < cutoff_ada, \
                                                            Y == 0)) / np.sum(Y == 0)
            precision_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                                      Y == 1)) / np.sum(probs_fraud_ada >= cutoff_ada)
            ndcg_ada1_training[m] = ndcg_k(Y, probs_fraud_ada, 99)
            FN_ada3 = np.sum(np.logical_and(probs_fraud_ada < cutoff_ada, \
                                            Y == 1))
            FP_ada3 = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                            Y == 0))
            ecm_ada1_training[m] = C_FN * P_f * FN_ada3 / n_P_training + C_FP * P_nf * FP_ada3 / n_N_training

            # performance on testing sample- LogitBoost
            roc_ada[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada)
            cutoff_OOS_ada = np.percentile(probs_oos_fraud_ada, 99)
            sensitivity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_ada >= cutoff_OOS_ada)
            ndcg_ada1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada, 99)
            FN_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                            Y_OOS == 1))
            FP_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                            Y_OOS == 0))
            ecm_ada1[m] = C_FN * P_f * FN_ada2 / n_P + C_FP * P_nf * FP_ada2 / n_N


            # Fused approach
            weight_ser = np.array([score_svm, score_lr, score_ada])
            weight_ser = weight_ser / np.sum(weight_ser)

            # performance on training sample- Fused
            probs_fraud_svm_fused = (1 + np.exp(-1 * probs_fraud_svm)) ** -1
            probs_fraud_lr_fused = (1 + np.exp(-1 * probs_fraud_lr)) ** -1
            probs_fraud_ada_fused = (1 + np.exp(-1 * probs_fraud_ada)) ** -1
            clf_fused_training = np.dot(np.array([probs_fraud_svm_fused, \
                                         probs_fraud_lr_fused, probs_fraud_ada_fused]).T, weight_ser)
            probs_fraud_fused = clf_fused_training
            cutoff_fused = np.percentile(probs_fraud_fused, 99)
            roc_fused_training[m] = roc_auc_score(Y, probs_fraud_fused)
            sensitivity_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                                              Y == 1)) / np.sum(Y)
            specificity_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused < cutoff_fused, \
                                                              Y == 0)) / np.sum(Y == 0)
            precision_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                                        Y == 1)) / np.sum(probs_fraud_fused >= cutoff_fused)
            ndcg_fused1_training[m] = ndcg_k(Y, probs_fraud_fused, 99)
            FN_fused3 = np.sum(np.logical_and(probs_fraud_fused < cutoff_fused, \
                                              Y == 1))
            FP_fused3 = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                              Y == 0))
            ecm_fused1_training[m] = C_FN * P_f * FN_fused3 / n_P_training + C_FP * P_nf * FP_fused3 / n_N_training



            # performance on testing sample- Fused
            probs_oos_fraud_svm = (1 + np.exp(-1 * probs_oos_fraud_svm)) ** -1
            probs_oos_fraud_lr = (1 + np.exp(-1 * probs_oos_fraud_lr)) ** -1
            probs_oos_fraud_ada = (1 + np.exp(-1 * probs_oos_fraud_ada)) ** -1
            clf_fused = np.dot(np.array([probs_oos_fraud_svm, \
                                         probs_oos_fraud_lr, probs_oos_fraud_ada]).T, weight_ser)
            probs_oos_fraud_fused = clf_fused
            roc_fused[m] = roc_auc_score(Y_OOS, probs_oos_fraud_fused)
            cutoff_OOS_fused = np.percentile(probs_oos_fraud_fused, 99)
            sensitivity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                              Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                                              Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                        Y_OOS == 1)) / np.sum(probs_oos_fraud_fused >= cutoff_OOS_fused)
            ndcg_fused1[m] = ndcg_k(Y_OOS, probs_oos_fraud_fused, 99)
            FN_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                              Y_OOS == 1))
            FP_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                              Y_OOS == 0))
            ecm_fused1[m] = C_FN * P_f * FN_fused2 / n_P + C_FP * P_nf * FP_fused2 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1



        f1_score_svm1_training = 2 * (precision_svm1_training * sensitivity_svm1_training) / \
                        (precision_svm1_training + sensitivity_svm1_training + 1e-8)
        f1_score_lr1_training = 2 * (precision_lr1_training * sensitivity_lr1_training) / \
                       (precision_lr1_training + sensitivity_lr1_training + 1e-8)
        f1_score_ada1_training = 2 * (precision_ada1_training * sensitivity_ada1_training) / \
                        (precision_ada1_training + sensitivity_ada1_training + 1e-8)
        f1_score_fused1_training = 2 * (precision_fused1_training * sensitivity_fused1_training) / \
                          (precision_fused1_training + sensitivity_fused1_training + 1e-8)


        f1_score_svm1 = 2 * (precision_svm1 * sensitivity_OOS_svm1) / \
                        (precision_svm1 + sensitivity_OOS_svm1 + 1e-8)
        f1_score_lr1 = 2 * (precision_lr1 * sensitivity_OOS_lr1) / \
                       (precision_lr1 + sensitivity_OOS_lr1 + 1e-8)
        f1_score_ada1 = 2 * (precision_ada1 * sensitivity_OOS_ada1) / \
                        (precision_ada1 + sensitivity_OOS_ada1 + 1e-8)
        f1_score_fused1 = 2 * (precision_fused1 * sensitivity_OOS_fused1) / \
                          (precision_fused1 + sensitivity_OOS_fused1 + 1e-8)


        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['SVM', 'LR', 'LogitBoost', 'FUSED']

        perf_tbl_general['Training Roc'] = [str(np.round(
            np.mean(roc_svm_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm_training) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr_training) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada_training) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Roc'] = [str(np.round(
            np.mean(roc_svm) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused) * 100, 2)) + '%)']

        gap_roc_svm = roc_svm - roc_svm_training
        gap_roc_lr = roc_lr - roc_lr_training
        gap_roc_ada = roc_ada - roc_ada_training
        gap_roc_fused = roc_fused - roc_fused_training

        mean_gap_roc_svm = np.round(np.mean(gap_roc_svm) * 100, 2)
        mean_gap_roc_lr = np.round(np.mean(gap_roc_lr) * 100, 2)
        mean_gap_roc_ada = np.round(np.mean(gap_roc_ada) * 100, 2)
        mean_gap_roc_fused = np.round(np.mean(gap_roc_fused) * 100, 2)

        perf_tbl_general['Gap Roc'] = [str(mean_gap_roc_svm) + '%', str(mean_gap_roc_lr) + '%', str(mean_gap_roc_ada) + '%', str(mean_gap_roc_fused) + '%']


        perf_tbl_general['Training Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_svm1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_svm1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_lr1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_lr1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_ada1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_ada1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_fused1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_fused1) * 100, 2)) + '%)']

        gap_sensitivity_svm = sensitivity_OOS_svm1 - sensitivity_svm1_training
        gap_sensitivity_lr = sensitivity_OOS_lr1 - sensitivity_lr1_training
        gap_sensitivity_ada = sensitivity_OOS_ada1 - sensitivity_ada1_training
        gap_sensitivity_fused = sensitivity_OOS_fused1 - sensitivity_fused1_training


        mean_gap_sensitivity_svm = np.round(np.mean(gap_sensitivity_svm) * 100, 2)
        mean_gap_sensitivity_lr = np.round(np.mean(gap_sensitivity_lr) * 100, 2)
        mean_gap_sensitivity_ada = np.round(np.mean(gap_sensitivity_ada) * 100, 2)
        mean_gap_sensitivity_fused = np.round(np.mean(gap_sensitivity_fused) * 100, 2)

        perf_tbl_general['Gap Sensitivity'] = [str(mean_gap_sensitivity_svm) + '%',\
                                               str(mean_gap_sensitivity_lr) + '%', \
                                               str(mean_gap_sensitivity_ada) + '%',\
                                       str(mean_gap_sensitivity_fused) + '%']


        perf_tbl_general['Training Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_svm1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_svm1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_lr1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_lr1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_ada1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_ada1_training) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_fused1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_fused1_training) * 100, 2)) + '%)']


        perf_tbl_general['Testing Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_fused1) * 100, 2)) + '%)']


        gap_specificity_svm = specificity_OOS_svm1 - specificity_svm1_training
        gap_specificity_lr = specificity_OOS_lr1 - specificity_lr1_training
        gap_specificity_ada = specificity_OOS_ada1 - specificity_ada1_training
        gap_specificity_fused = specificity_OOS_fused1 - specificity_fused1_training

        mean_gap_specificity_svm = np.round(np.mean(gap_specificity_svm) * 100, 2)
        mean_gap_specificity_lr = np.round(np.mean(gap_specificity_lr) * 100, 2)
        mean_gap_specificity_ada = np.round(np.mean(gap_specificity_ada) * 100, 2)
        mean_gap_specificity_fused = np.round(np.mean(gap_specificity_fused) * 100, 2)

        perf_tbl_general['Gap Specificity'] = [str(mean_gap_specificity_svm) + '%', str(mean_gap_specificity_lr) + '%', str(mean_gap_specificity_ada) + '%',\
                                       str(mean_gap_specificity_fused) + '%']


        perf_tbl_general['Training Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1_training) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1_training) * 100, 2)) + '%)']


        perf_tbl_general['Testing Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1) * 100, 2)) + '%)']

        gap_precision_svm = precision_svm1 - precision_svm1_training
        gap_precision_lr = precision_lr1 - precision_lr1_training
        gap_precision_ada = precision_ada1 - precision_ada1_training
        gap_precision_fused = precision_fused1 - precision_fused1_training


        mean_gap_precision_svm = np.round(np.mean(gap_precision_svm) * 100, 2)
        mean_gap_precision_lr = np.round(np.mean(gap_precision_lr) * 100, 2)
        mean_gap_precision_ada = np.round(np.mean(gap_precision_ada) * 100, 2)
        mean_gap_precision_fused = np.round(np.mean(gap_precision_fused) * 100, 2)

        perf_tbl_general['Gap Precision'] = [str(mean_gap_precision_svm) + '%', str(mean_gap_precision_lr) + '%', str(mean_gap_precision_ada) + '%',\
                                       str(mean_gap_precision_fused) + '%']


        perf_tbl_general['Training F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1_training) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1) * 100, 2)) + '%)']


        gap_f1_score_svm = f1_score_svm1 - f1_score_svm1_training
        gap_f1_score_lr = f1_score_lr1 - f1_score_lr1_training
        gap_f1_score_ada = f1_score_ada1 - f1_score_ada1_training
        gap_f1_score_fused = f1_score_fused1 - f1_score_fused1_training


        mean_gap_f1_score_svm = np.round(np.mean(gap_f1_score_svm) * 100, 2)
        mean_gap_f1_score_lr = np.round(np.mean(gap_f1_score_lr) * 100, 2)
        mean_gap_f1_score_ada = np.round(np.mean(gap_f1_score_ada) * 100, 2)
        mean_gap_f1_score_fused = np.round(np.mean(gap_f1_score_fused) * 100, 2)

        perf_tbl_general['Gap F1 Score'] = [str(mean_gap_f1_score_svm) + '%', str(mean_gap_f1_score_lr) + '%', str(mean_gap_f1_score_ada) + '%',\
                                       str(mean_gap_f1_score_fused) + '%']


        perf_tbl_general['Training NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1_training) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1) * 100, 2)) + '%)']

        gap_ndcg_svm = ndcg_svm1 - ndcg_svm1_training
        gap_ndcg_lr = ndcg_lr1 - ndcg_lr1_training
        gap_ndcg_ada = ndcg_ada1 - ndcg_ada1_training
        gap_ndcg_fused = ndcg_fused1 - ndcg_fused1_training

        mean_gap_ndcg_svm = np.round(np.mean(gap_ndcg_svm) * 100, 2)
        mean_gap_ndcg_lr = np.round(np.mean(gap_ndcg_lr) * 100, 2)
        mean_gap_ndcg_ada = np.round(np.mean(gap_ndcg_ada) * 100, 2)
        mean_gap_ndcg_fused = np.round(np.mean(gap_ndcg_fused) * 100, 2)

        perf_tbl_general['Gap NDCG'] = [str(mean_gap_ndcg_svm) + '%', str(mean_gap_ndcg_lr) + '%', str(mean_gap_ndcg_ada) + '%',\
                                       str(mean_gap_ndcg_fused) + '%']


        perf_tbl_general['Training ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1_training) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1) * 100, 2)) + '%)']

        gap_ecm_svm = ecm_svm1 - ecm_svm1_training
        gap_ecm_lr = ecm_lr1 - ecm_lr1_training
        gap_ecm_ada = ecm_ada1 - ecm_ada1_training
        gap_ecm_fused = ecm_fused1 - ecm_fused1_training

        mean_gap_ecm_svm = np.round(np.mean(gap_ecm_svm) * 100, 2)
        mean_gap_ecm_lr = np.round(np.mean(gap_ecm_lr) * 100, 2)
        mean_gap_ecm_ada = np.round(np.mean(gap_ecm_ada) * 100, 2)
        mean_gap_ecm_fused = np.round(np.mean(gap_ecm_fused) * 100, 2)

        perf_tbl_general['Gap ECM'] = [str(mean_gap_ecm_svm) + '%', str(mean_gap_ecm_lr) + '%', str(mean_gap_ecm_ada) + '%',\
                                       str(mean_gap_ecm_fused) + '%']



        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_11ratios_Table5_with_gap.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')



    def raw_analyse(self, C_FN=30, C_FP=1):
        """
        This code uses 28 raw accounting items for Table 6. RUSBoost and SVM-FK23 are not included.
        Combined with ratio_analyse, Table 7 can be reproduced.

        Methodological choices:
            - 28 raw accounting items from Bao et al. (2020)
            - 10-fold validation
            - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-14]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]  # 1991
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]  # 2010

        # Setting the cross-validation setting
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)

        X_CV = tbl_year_IS_CV.iloc[:, -28:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        # redo cross-validation if you wish
        if cv_type == 'kfold':
            if cross_val == True:

                # optimise LogitBoost
                print('Grid search hyperparameter optimisation started for LogitBoost')

                best_perf_ada = 0

                base_lr = LogisticRegression(random_state=0, solver='newton-cg')

                pipe_ada = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_ada', AdaBoostClassifier(estimator=base_lr, random_state=0))])
                estimators = list(range(10,3001,10))
                learning_rate = [x/1000 for x in range(10, 1001,10)]
                param_grid_ada = {'base_mdl_ada__n_estimators': estimators, \
                                  'base_mdl_ada__learning_rate': learning_rate}

                clf_ada = GridSearchCV(pipe_ada, param_grid_ada, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_ada.fit(X_CV, Y_CV)
                score_ada = clf_ada.best_score_
                if score_ada >= best_perf_ada:
                    best_perf_ada = score_ada
                    opt_params_ada = clf_ada.best_params_

                print('LogitBoost: The optimal number of estimators is ' + \
                      str(opt_params_ada['base_mdl_ada__n_estimators']) + ', and learning rate ' + \
                      str(opt_params_ada['base_mdl_ada__learning_rate']))

                print('Computing CV ROC for LR ...')
                score_lr = []
                for m in range(0, k_fold):
                    train_sample, test_sample = train_test_split(Y_CV, test_size=1 /
                                                                                 k_fold, shuffle=False, random_state=m)
                    X_train = X_CV.iloc[train_sample.index]
                    X_train = add_constant(X_train)
                    Y_train = train_sample
                    X_test = X_CV.iloc[test_sample.index]
                    X_test = add_constant(X_test)
                    Y_test = test_sample

                    logit_model = Logit(Y_train, X_train)
                    logit_model = logit_model.fit(disp=0)
                    pred_LR_CV = logit_model.predict(X_test)
                    score_lr.append(roc_auc_score(Y_test, pred_LR_CV))

                score_lr = np.mean(score_lr)

                # optimize SVM grid

                print('Grid search hyperparameter optimisation started for SVM')
                t1 = datetime.now()

                pipe_svm = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_svm', SVC(shrinking=False, \
                                                          probability=False, random_state=0, max_iter=-1, \
                                                          tol=X_CV.shape[-1] * 1e-3))])

                C = [0.001, 0.01, 0.1, 1, 10, 100]
                gamma = [0.0001, 0.001, 0.01, 0.1]
                kernel = ['rbf', 'linear', 'poly']
                class_weight = [{0: 1/x, 1: 1} for x in range(10,501,10)]
                param_grid_svm = {'base_mdl_svm__kernel': kernel,\
                                  'base_mdl_svm__C': C, \
                                  'base_mdl_svm__gamma': gamma,\
                                  'base_mdl_svm__class_weight':class_weight}


                clf_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_svm.fit(X_CV, Y_CV)
                opt_params_svm = clf_svm.best_params_
                gamma_opt = opt_params_svm['base_mdl_svm__gamma']
                cw_opt = opt_params_svm['base_mdl_svm__class_weight']
                c_opt = opt_params_svm['base_mdl_svm__C']
                kernel_opt = opt_params_svm['base_mdl_svm__kernel']
                score_svm = clf_svm.best_score_

                t2 = datetime.now()
                dt = t2 - t1
                print('SVM CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('SVM: The optimal C+/C- ratio is ' + str(cw_opt) + ', gamma is' + str(
                    gamma_opt) + ', C is' + str(c_opt) + ',score is' + str(score_svm) +\
                      ', kernel is' + str(kernel_opt))

        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)

        roc_svm = np.zeros(len(range_oos))
        roc_svm_training = np.zeros(len(range_oos))
        sensitivity_OOS_svm1 = np.zeros(len(range_oos))
        sensitivity_svm1_training = np.zeros(len(range_oos))
        specificity_OOS_svm1 = np.zeros(len(range_oos))
        specificity_svm1_training = np.zeros(len(range_oos))
        precision_svm1 = np.zeros(len(range_oos))
        precision_svm1_training = np.zeros(len(range_oos))
        ndcg_svm1 = np.zeros(len(range_oos))
        ndcg_svm1_training = np.zeros(len(range_oos))
        ecm_svm1 = np.zeros(len(range_oos))
        ecm_svm1_training = np.zeros(len(range_oos))


        roc_lr = np.zeros(len(range_oos))
        roc_lr_training = np.zeros(len(range_oos))
        sensitivity_OOS_lr1 = np.zeros(len(range_oos))
        sensitivity_lr1_training = np.zeros(len(range_oos))
        specificity_OOS_lr1 = np.zeros(len(range_oos))
        specificity_lr1_training = np.zeros(len(range_oos))
        precision_lr1 = np.zeros(len(range_oos))
        precision_lr1_training = np.zeros(len(range_oos))
        ndcg_lr1 = np.zeros(len(range_oos))
        ndcg_lr1_training = np.zeros(len(range_oos))
        ecm_lr1 = np.zeros(len(range_oos))
        ecm_lr1_training = np.zeros(len(range_oos))

        roc_ada = np.zeros(len(range_oos))
        roc_ada_training = np.zeros(len(range_oos))
        sensitivity_OOS_ada1 = np.zeros(len(range_oos))
        sensitivity_ada1_training = np.zeros(len(range_oos))
        specificity_OOS_ada1 = np.zeros(len(range_oos))
        specificity_ada1_training = np.zeros(len(range_oos))
        precision_ada1 = np.zeros(len(range_oos))
        precision_ada1_training = np.zeros(len(range_oos))
        ndcg_ada1 = np.zeros(len(range_oos))
        ndcg_ada1_training = np.zeros(len(range_oos))
        ecm_ada1 = np.zeros(len(range_oos))
        ecm_ada1_training = np.zeros(len(range_oos))

        roc_fused = np.zeros(len(range_oos))
        roc_fused_training = np.zeros(len(range_oos))
        sensitivity_OOS_fused1 = np.zeros(len(range_oos))
        sensitivity_fused1_training = np.zeros(len(range_oos))
        specificity_OOS_fused1 = np.zeros(len(range_oos))
        specificity_fused1_training = np.zeros(len(range_oos))
        precision_fused1 = np.zeros(len(range_oos))
        precision_fused1_training = np.zeros(len(range_oos))
        ndcg_fused1 = np.zeros(len(range_oos))
        ndcg_fused1_training = np.zeros(len(range_oos))
        ecm_fused1 = np.zeros(len(range_oos))
        ecm_fused1_training = np.zeros(len(range_oos))

        m = 0
        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period

            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -28:]
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -28:]
            X_OOS = (X_OOS - mean) / std
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            n_P_training = np.sum(Y == 1)
            n_N_training = np.sum(Y == 0)

            # Support Vector Machines

            clf_svm = SVC(class_weight=opt_params_svm['base_mdl_svm__class_weight'],
                          kernel=opt_params_svm['base_mdl_svm__kernel'], C=opt_params_svm['base_mdl_svm__C'], \
                          gamma=opt_params_svm['base_mdl_svm__gamma'], shrinking=False, \
                          probability=False, random_state=0, max_iter=-1, \
                          tol=X.shape[-1] * 1e-3)

            clf_svm = clf_svm.fit(X, Y)


            #performance on training sample- svm
            pred_train_svm = clf_svm.decision_function(X)
            probs_fraud_svm = np.exp(pred_train_svm) / (1 + np.exp(pred_train_svm))
            cutoff_svm = np.percentile(probs_fraud_svm, 99)
            roc_svm_training[m] = roc_auc_score(Y, probs_fraud_svm)
            sensitivity_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                                            Y == 1)) / np.sum(Y)
            specificity_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm < cutoff_svm, \
                                                            Y == 0)) / np.sum(Y == 0)
            precision_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                                      Y == 1)) / np.sum(probs_fraud_svm >= cutoff_svm)
            ndcg_svm1_training[m] = ndcg_k(Y, probs_fraud_svm, 99)
            FN_svm3 = np.sum(np.logical_and(probs_fraud_svm < cutoff_svm, \
                                            Y == 1))
            FP_svm3 = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                            Y == 0))
            ecm_svm1_training[m] = C_FN * P_f * FN_svm3 / n_P_training + C_FP * P_nf * FP_svm3 / n_N_training

            #performance on testing sample- svm
            pred_test_svm = clf_svm.decision_function(X_OOS)
            probs_oos_fraud_svm = np.exp(pred_test_svm) / (1 + np.exp(pred_test_svm))
            roc_svm[m] = roc_auc_score(Y_OOS, probs_oos_fraud_svm)

            cutoff_OOS_svm = np.percentile(probs_oos_fraud_svm, 99)
            sensitivity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_svm >= cutoff_OOS_svm)
            ndcg_svm1[m] = ndcg_k(Y_OOS, probs_oos_fraud_svm, 99)
            FN_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                            Y_OOS == 1))
            FP_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                            Y_OOS == 0))
            ecm_svm1[m] = C_FN * P_f * FN_svm2 / n_P + C_FP * P_nf * FP_svm2 / n_N

            # Logistic Regression – Dechow et al (2011)
            X_lr = add_constant(X)
            X_OOS_lr = add_constant(X_OOS)
            clf_lr = Logit(Y, X_lr)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS_lr)

            #performance on training sample- logit
            probs_fraud_lr = clf_lr.predict(X_lr)
            cutoff_lr = np.percentile(probs_fraud_lr, 99)
            roc_lr_training[m] = roc_auc_score(Y, probs_fraud_lr)
            sensitivity_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                                           Y == 1)) / np.sum(Y)
            specificity_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr < cutoff_lr, \
                                                           Y == 0)) / np.sum(Y == 0)
            precision_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                                     Y == 1)) / np.sum(probs_fraud_lr >= cutoff_lr)
            ndcg_lr1_training[m] = ndcg_k(Y, probs_fraud_lr, 99)
            FN_lr3 = np.sum(np.logical_and(probs_fraud_lr < cutoff_lr, \
                                           Y == 1))
            FP_lr3 = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                           Y == 0))
            ecm_lr1_training[m] = C_FN * P_f * FN_lr3 / n_P_training + C_FP * P_nf * FP_lr3 / n_N_training

            # performance on testing sample- logit
            roc_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_lr)
            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)
            sensitivity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                           Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                                           Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                     Y_OOS == 1)) / np.sum(probs_oos_fraud_lr >= cutoff_OOS_lr)
            ndcg_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_lr, 99)
            FN_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                           Y_OOS == 1))
            FP_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                           Y_OOS == 0))
            ecm_lr1[m] = C_FN * P_f * FN_lr2 / n_P + C_FP * P_nf * FP_lr2 / n_N


            # LogitBoost
            base_lr = LogisticRegression(random_state=0, solver='newton-cg')

            clf_ada = AdaBoostClassifier(n_estimators=opt_params_ada['base_mdl_ada__n_estimators'], \
                                         learning_rate=opt_params_ada['base_mdl_ada__learning_rate'], \
                                         estimator=base_lr, random_state=0)
            clf_ada = clf_ada.fit(X, Y)
            probs_oos_fraud_ada = clf_ada.predict_proba(X_OOS)[:, -1]


            # performance on training sample- LogitBoost
            probs_fraud_ada = clf_ada.predict_proba(X)[:, -1]
            cutoff_ada = np.percentile(probs_fraud_ada, 99)
            roc_ada_training[m] = roc_auc_score(Y, probs_fraud_ada)
            sensitivity_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                                            Y == 1)) / np.sum(Y)
            specificity_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada < cutoff_ada, \
                                                            Y == 0)) / np.sum(Y == 0)
            precision_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                                      Y == 1)) / np.sum(probs_fraud_ada >= cutoff_ada)
            ndcg_ada1_training[m] = ndcg_k(Y, probs_fraud_ada, 99)
            FN_ada3 = np.sum(np.logical_and(probs_fraud_ada < cutoff_ada, \
                                            Y == 1))
            FP_ada3 = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                            Y == 0))
            ecm_ada1_training[m] = C_FN * P_f * FN_ada3 / n_P_training + C_FP * P_nf * FP_ada3 / n_N_training

            # performance on testing sample- LogitBoost
            roc_ada[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada)
            cutoff_OOS_ada = np.percentile(probs_oos_fraud_ada, 99)
            sensitivity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_ada >= cutoff_OOS_ada)
            ndcg_ada1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada, 99)
            FN_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                            Y_OOS == 1))
            FP_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                            Y_OOS == 0))
            ecm_ada1[m] = C_FN * P_f * FN_ada2 / n_P + C_FP * P_nf * FP_ada2 / n_N


            # Fused approach
            weight_ser = np.array([score_svm, score_lr, score_ada])
            weight_ser = weight_ser / np.sum(weight_ser)

            # performance on training sample- Fused
            probs_fraud_svm_fused = (1 + np.exp(-1 * probs_fraud_svm)) ** -1
            probs_fraud_lr_fused = (1 + np.exp(-1 * probs_fraud_lr)) ** -1
            probs_fraud_ada_fused = (1 + np.exp(-1 * probs_fraud_ada)) ** -1

            clf_fused_training = np.dot(np.array([probs_fraud_svm_fused, \
                                         probs_fraud_lr_fused, probs_fraud_ada_fused]).T, weight_ser)

            probs_fraud_fused = clf_fused_training
            cutoff_fused = np.percentile(probs_fraud_fused, 99)
            roc_fused_training[m] = roc_auc_score(Y, probs_fraud_fused)
            sensitivity_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                                              Y == 1)) / np.sum(Y)
            specificity_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused < cutoff_fused, \
                                                              Y == 0)) / np.sum(Y == 0)
            precision_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                                        Y == 1)) / np.sum(probs_fraud_fused >= cutoff_fused)
            ndcg_fused1_training[m] = ndcg_k(Y, probs_fraud_fused, 99)
            FN_fused3 = np.sum(np.logical_and(probs_fraud_fused < cutoff_fused, \
                                              Y == 1))
            FP_fused3 = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                              Y == 0))
            ecm_fused1_training[m] = C_FN * P_f * FN_fused3 / n_P_training + C_FP * P_nf * FP_fused3 / n_N_training


            # performance on testing sample- Fused
            probs_oos_fraud_svm = (1 + np.exp(-1 * probs_oos_fraud_svm)) ** -1
            probs_oos_fraud_lr = (1 + np.exp(-1 * probs_oos_fraud_lr)) ** -1
            probs_oos_fraud_ada = (1 + np.exp(-1 * probs_oos_fraud_ada)) ** -1

            clf_fused = np.dot(np.array([probs_oos_fraud_svm, \
                                         probs_oos_fraud_lr, probs_oos_fraud_ada]).T, weight_ser)
            probs_oos_fraud_fused = clf_fused
            roc_fused[m] = roc_auc_score(Y_OOS, probs_oos_fraud_fused)
            cutoff_OOS_fused = np.percentile(probs_oos_fraud_fused, 99)
            sensitivity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                              Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                                              Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                        Y_OOS == 1)) / np.sum(probs_oos_fraud_fused >= cutoff_OOS_fused)
            ndcg_fused1[m] = ndcg_k(Y_OOS, probs_oos_fraud_fused, 99)

            FN_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                              Y_OOS == 1))
            FP_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                              Y_OOS == 0))
            ecm_fused1[m] = C_FN * P_f * FN_fused2 / n_P + C_FP * P_nf * FP_fused2 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1


        f1_score_svm1_training = 2 * (precision_svm1_training * sensitivity_svm1_training) / \
                        (precision_svm1_training + sensitivity_svm1_training + 1e-8)
        f1_score_lr1_training = 2 * (precision_lr1_training * sensitivity_lr1_training) / \
                       (precision_lr1_training + sensitivity_lr1_training + 1e-8)
        f1_score_ada1_training = 2 * (precision_ada1_training * sensitivity_ada1_training) / \
                        (precision_ada1_training + sensitivity_ada1_training + 1e-8)
        f1_score_fused1_training = 2 * (precision_fused1_training * sensitivity_fused1_training) / \
                          (precision_fused1_training + sensitivity_fused1_training + 1e-8)

        f1_score_svm1 = 2 * (precision_svm1 * sensitivity_OOS_svm1) / \
                        (precision_svm1 + sensitivity_OOS_svm1 + 1e-8)
        f1_score_lr1 = 2 * (precision_lr1 * sensitivity_OOS_lr1) / \
                       (precision_lr1 + sensitivity_OOS_lr1 + 1e-8)
        f1_score_ada1 = 2 * (precision_ada1 * sensitivity_OOS_ada1) / \
                        (precision_ada1 + sensitivity_OOS_ada1 + 1e-8)
        f1_score_fused1 = 2 * (precision_fused1 * sensitivity_OOS_fused1) / \
                          (precision_fused1 + sensitivity_OOS_fused1 + 1e-8)

        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['SVM', 'LR', 'LogitBoost', 'FUSED']

        perf_tbl_general['Training Roc'] = [str(np.round(
            np.mean(roc_svm_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm_training) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr_training) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada_training) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Roc'] = [str(np.round(
            np.mean(roc_svm) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused) * 100, 2)) + '%)']

        gap_roc_svm = roc_svm - roc_svm_training
        gap_roc_lr = roc_lr - roc_lr_training
        gap_roc_ada = roc_ada - roc_ada_training
        gap_roc_fused = roc_fused - roc_fused_training

        mean_gap_roc_svm = np.round(np.mean(gap_roc_svm) * 100, 2)
        mean_gap_roc_lr = np.round(np.mean(gap_roc_lr) * 100, 2)
        mean_gap_roc_ada = np.round(np.mean(gap_roc_ada) * 100, 2)
        mean_gap_roc_fused = np.round(np.mean(gap_roc_fused) * 100, 2)

        perf_tbl_general['Gap Roc'] = [str(mean_gap_roc_svm) + '%', str(mean_gap_roc_lr) + '%', str(mean_gap_roc_ada) + '%', str(mean_gap_roc_fused) + '%']

        perf_tbl_general['Training Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_svm1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_svm1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_lr1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_lr1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_ada1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_ada1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_fused1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_fused1) * 100, 2)) + '%)']

        gap_sensitivity_svm = sensitivity_OOS_svm1 - sensitivity_svm1_training
        gap_sensitivity_lr = sensitivity_OOS_lr1 - sensitivity_lr1_training
        gap_sensitivity_ada = sensitivity_OOS_ada1 - sensitivity_ada1_training
        gap_sensitivity_fused = sensitivity_OOS_fused1 - sensitivity_fused1_training


        mean_gap_sensitivity_svm = np.round(np.mean(gap_sensitivity_svm) * 100, 2)
        mean_gap_sensitivity_lr = np.round(np.mean(gap_sensitivity_lr) * 100, 2)
        mean_gap_sensitivity_ada = np.round(np.mean(gap_sensitivity_ada) * 100, 2)
        mean_gap_sensitivity_fused = np.round(np.mean(gap_sensitivity_fused) * 100, 2)

        perf_tbl_general['Gap Sensitivity'] = [str(mean_gap_sensitivity_svm) + '%',\
                                               str(mean_gap_sensitivity_lr) + '%', \
                                               str(mean_gap_sensitivity_ada) + '%',\
                                       str(mean_gap_sensitivity_fused) + '%']

        perf_tbl_general['Training Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_svm1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_svm1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_lr1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_lr1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_ada1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_ada1_training) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_fused1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_fused1_training) * 100, 2)) + '%)']


        perf_tbl_general['Testing Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_fused1) * 100, 2)) + '%)']


        gap_specificity_svm = specificity_OOS_svm1 - specificity_svm1_training
        gap_specificity_lr = specificity_OOS_lr1 - specificity_lr1_training
        gap_specificity_ada = specificity_OOS_ada1 - specificity_ada1_training
        gap_specificity_fused = specificity_OOS_fused1 - specificity_fused1_training

        mean_gap_specificity_svm = np.round(np.mean(gap_specificity_svm) * 100, 2)
        mean_gap_specificity_lr = np.round(np.mean(gap_specificity_lr) * 100, 2)
        mean_gap_specificity_ada = np.round(np.mean(gap_specificity_ada) * 100, 2)
        mean_gap_specificity_fused = np.round(np.mean(gap_specificity_fused) * 100, 2)

        perf_tbl_general['Gap Specificity'] = [str(mean_gap_specificity_svm) + '%', str(mean_gap_specificity_lr) + '%', str(mean_gap_specificity_ada) + '%',\
                                       str(mean_gap_specificity_fused) + '%']

        perf_tbl_general['Training Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1_training) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1_training) * 100, 2)) + '%)']


        perf_tbl_general['Testing Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1) * 100, 2)) + '%)']

        gap_precision_svm = precision_svm1 - precision_svm1_training
        gap_precision_lr = precision_lr1 - precision_lr1_training
        gap_precision_ada = precision_ada1 - precision_ada1_training
        gap_precision_fused = precision_fused1 - precision_fused1_training


        mean_gap_precision_svm = np.round(np.mean(gap_precision_svm) * 100, 2)
        mean_gap_precision_lr = np.round(np.mean(gap_precision_lr) * 100, 2)
        mean_gap_precision_ada = np.round(np.mean(gap_precision_ada) * 100, 2)
        mean_gap_precision_fused = np.round(np.mean(gap_precision_fused) * 100, 2)

        perf_tbl_general['Gap Precision'] = [str(mean_gap_precision_svm) + '%', str(mean_gap_precision_lr) + '%', str(mean_gap_precision_ada) + '%',\
                                       str(mean_gap_precision_fused) + '%']

        perf_tbl_general['Training F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1_training) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1) * 100, 2)) + '%)']


        gap_f1_score_svm = f1_score_svm1 - f1_score_svm1_training
        gap_f1_score_lr = f1_score_lr1 - f1_score_lr1_training
        gap_f1_score_ada = f1_score_ada1 - f1_score_ada1_training
        gap_f1_score_fused = f1_score_fused1 - f1_score_fused1_training


        mean_gap_f1_score_svm = np.round(np.mean(gap_f1_score_svm) * 100, 2)
        mean_gap_f1_score_lr = np.round(np.mean(gap_f1_score_lr) * 100, 2)
        mean_gap_f1_score_ada = np.round(np.mean(gap_f1_score_ada) * 100, 2)
        mean_gap_f1_score_fused = np.round(np.mean(gap_f1_score_fused) * 100, 2)

        perf_tbl_general['Gap F1 Score'] = [str(mean_gap_f1_score_svm) + '%', str(mean_gap_f1_score_lr) + '%', str(mean_gap_f1_score_ada) + '%',\
                                       str(mean_gap_f1_score_fused) + '%']


        perf_tbl_general['Training NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1_training) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1) * 100, 2)) + '%)']

        gap_ndcg_svm = ndcg_svm1 - ndcg_svm1_training
        gap_ndcg_lr = ndcg_lr1 - ndcg_lr1_training
        gap_ndcg_ada = ndcg_ada1 - ndcg_ada1_training
        gap_ndcg_fused = ndcg_fused1 - ndcg_fused1_training

        mean_gap_ndcg_svm = np.round(np.mean(gap_ndcg_svm) * 100, 2)
        mean_gap_ndcg_lr = np.round(np.mean(gap_ndcg_lr) * 100, 2)
        mean_gap_ndcg_ada = np.round(np.mean(gap_ndcg_ada) * 100, 2)
        mean_gap_ndcg_fused = np.round(np.mean(gap_ndcg_fused) * 100, 2)

        perf_tbl_general['Gap NDCG'] = [str(mean_gap_ndcg_svm) + '%', str(mean_gap_ndcg_lr) + '%', str(mean_gap_ndcg_ada) + '%',\
                                       str(mean_gap_ndcg_fused) + '%']

        perf_tbl_general['Training ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1_training) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1) * 100, 2)) + '%)']

        gap_ecm_svm = ecm_svm1 - ecm_svm1_training
        gap_ecm_lr = ecm_lr1 - ecm_lr1_training
        gap_ecm_ada = ecm_ada1 - ecm_ada1_training
        gap_ecm_fused = ecm_fused1 - ecm_fused1_training

        mean_gap_ecm_svm = np.round(np.mean(gap_ecm_svm) * 100, 2)
        mean_gap_ecm_lr = np.round(np.mean(gap_ecm_lr) * 100, 2)
        mean_gap_ecm_ada = np.round(np.mean(gap_ecm_ada) * 100, 2)
        mean_gap_ecm_fused = np.round(np.mean(gap_ecm_fused) * 100, 2)

        perf_tbl_general['Gap ECM'] = [str(mean_gap_ecm_svm) + '%', str(mean_gap_ecm_lr) + '%', str(mean_gap_ecm_ada) + '%',\
                                       str(mean_gap_ecm_fused) + '%']


        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_28raw_Table6_with_gap.csv.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')
        
        
    def ratio_temporal(self, C_FN=30, C_FP=1):
        """
        This code uses temporal validation selecting optimal hyperparameters for Table 8.
        RUSBoost and SVM-FK23 are not included.

        Methodological choices:
            - 11 financial ratios from Dechow et al.(2011)
            - temporal validation
            - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, -14:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]  # 1991
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]  # 2010

        # Setting the cross-validation setting
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)

        X_CV = tbl_year_IS_CV.iloc[:, -11:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        # redo cross-validation if you wish
        cutoff_temporal = 2001 - temp_year
        X_CV_train = X_CV[tbl_year_IS_CV['fyear'] < cutoff_temporal]
        Y_CV_train = Y_CV[tbl_year_IS_CV['fyear'] < cutoff_temporal]

        mean_CV = np.mean(X_CV_train)
        std_CV = np.std(X_CV_train)
        X_CV_train = (X_CV_train - mean_CV) / std_CV

        X_CV_test = X_CV[tbl_year_IS_CV['fyear'] >= cutoff_temporal]
        Y_CV_test = Y_CV[tbl_year_IS_CV['fyear'] >= cutoff_temporal]

        X_CV_test = (X_CV_test - mean_CV) / std_CV


        # optimise LogitBoost
        print('Grid search hyperparameter optimisation started for LogitBoost')
        t1 = datetime.now()
        estimators = list(range(10, 3001, 10))
        learning_rate = [x / 1000 for x in range(10, 1001, 10)]
        param_grid_ada = {'n_estimators': estimators, \
                          'learning_rate': learning_rate}

        temp_ada = {'n_estimators': [], 'learning_rate': [], 'score': []}

        for n in param_grid_ada['n_estimators']:
            for r in param_grid_ada['learning_rate']:
                temp_ada['n_estimators'].append(n)
                temp_ada['learning_rate'].append(r)

                base_lr = LogisticRegression(random_state=0, solver='newton-cg')
                base_mdl_ada = AdaBoostClassifier(estimator=base_lr,
                                                  learning_rate=r,
                                                  n_estimators=n,
                                                  random_state=0).fit(
                    X_CV_train,
                    Y_CV_train)
                predicted_ada_test = base_mdl_ada.predict_proba(X_CV_test)[:, -1]
                temp_ada['score'].append(roc_auc_score(Y_CV_test, predicted_ada_test))

        idx_opt_ada = temp_ada['score'].index(np.max(temp_ada['score']))

        score_ada = temp_ada['score'][idx_opt_ada]
        opt_params_ada = {'n_estimators': temp_ada['n_estimators'][idx_opt_ada],
                          'learning_rate': temp_ada['learning_rate'][idx_opt_ada]}

        t2 = datetime.now()
        dt = t2 - t1
        print('LogitBoost Temporal validation finished after ' + str(dt.total_seconds()) + ' sec')

        print('LogitBoost: The optimal number of estimators is ' + \
              str(opt_params_ada['n_estimators']) + ', and learning rate ' + \
              str(opt_params_ada['learning_rate']))

        # optimize SVM grid

        print('Grid search hyperparameter optimisation started for SVM')
        t1 = datetime.now()
        C = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma = [0.0001, 0.001, 0.01, 0.1]
        kernel = ['rbf', 'linear', 'poly']
        class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]
        param_grid_svm = {'kernel': kernel, 'class_weight': class_weight,\
                          'C':C, 'gamma':gamma}

        temp_svm = {'kernel': [], 'class_weight': [], 'C': [], 'gamma': [], 'score': []}

        for k in param_grid_svm['kernel']:
            for w in param_grid_svm['class_weight']:
                for c in param_grid_svm['C']:
                    for g in param_grid_svm['gamma']:
                        temp_svm['kernel'].append(k)
                        temp_svm['class_weight'].append(w)
                        temp_svm['C'].append(c)
                        temp_svm['gamma'].append(g)

                        base_mdl_svm = SVC(shrinking=False, \
                                           probability=False,
                                           kernel=k, C=c, class_weight=w, gamma=g,\
                                           random_state=0, max_iter=-1, \
                                           tol=X_CV.shape[-1] * 1e-3).fit(X_CV_train, Y_CV_train)
                        predicted_test_svc = base_mdl_svm.decision_function(X_CV_test)
                        predicted_test_svc = np.exp(predicted_test_svc) / (1 + np.exp(predicted_test_svc))

                        temp_svm['score'].append(roc_auc_score(Y_CV_test, predicted_test_svc))

        idx_opt_svm = temp_svm['score'].index(np.max(temp_svm['score']))

        cw_opt = temp_svm['class_weight'][idx_opt_svm][0]
        C_opt = temp_svm['C'][idx_opt_svm]
        kernel_opt = temp_svm['kernel'][idx_opt_svm]
        gamma_opt = temp_svm['gamma'][idx_opt_svm]
        score_svm = temp_svm['score'][idx_opt_svm]
        opt_params_svm = {'class_weight': temp_svm['class_weight'][idx_opt_svm],
                          'kernel': kernel_opt, 'C': C_opt, 'gamma': gamma_opt}
        print(opt_params_svm)
        t2 = datetime.now()
        dt = t2 - t1
        print('SVM Temporal validation finished after ' + str(dt.total_seconds()) + ' sec')
        print('SVM: The optimal C+/C- ratio is ' + str(cw_opt) + ', C is' + str(C_opt) +\
              'kernel is ' + str(kernel_opt) + 'gamma is ' + str(gamma_opt))

        print('Computing Temporal validation ROC for LR ...')
        t1 = datetime.now()

        logit_model = Logit(Y_CV_train, X_CV_train).fit(disp=0)
        pred_LR_CV = logit_model.predict(X_CV_test)
        score_lr = (roc_auc_score(Y_CV_test, pred_LR_CV))

        t2 = datetime.now()
        dt = t2 - t1
        print('LR Temporal validation finished after ' + str(dt.total_seconds()) + ' sec')



        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)  # (2001,2010+1,1)

        roc_svm = np.zeros(len(range_oos))
        sensitivity_OOS_svm1 = np.zeros(len(range_oos))
        specificity_OOS_svm1 = np.zeros(len(range_oos))
        precision_svm1 = np.zeros(len(range_oos))
        ndcg_svm1 = np.zeros(len(range_oos))
        ecm_svm1 = np.zeros(len(range_oos))

        roc_lr = np.zeros(len(range_oos))
        sensitivity_OOS_lr1 = np.zeros(len(range_oos))
        specificity_OOS_lr1 = np.zeros(len(range_oos))
        precision_lr1 = np.zeros(len(range_oos))
        ndcg_lr1 = np.zeros(len(range_oos))
        ecm_lr1 = np.zeros(len(range_oos))

        roc_ada = np.zeros(len(range_oos))
        sensitivity_OOS_ada1 = np.zeros(len(range_oos))
        specificity_OOS_ada1 = np.zeros(len(range_oos))
        precision_ada1 = np.zeros(len(range_oos))
        ndcg_ada1 = np.zeros(len(range_oos))
        ecm_ada1 = np.zeros(len(range_oos))

        roc_fused = np.zeros(len(range_oos))
        sensitivity_OOS_fused1 = np.zeros(len(range_oos))
        specificity_OOS_fused1 = np.zeros(len(range_oos))
        precision_fused1 = np.zeros(len(range_oos))
        ndcg_fused1 = np.zeros(len(range_oos))
        ecm_fused1 = np.zeros(len(range_oos))

        m = 0
        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]

            print(f'before dropping the number of observations is: {len(tbl_year_OOS)}')

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True


            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)
            print(f'after dropping the number of observations is: {len(tbl_year_OOS)}')

            X = tbl_year_IS.iloc[:, -11:]
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS = (X_OOS - mean) / std
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)


            # Support Vector Machines

            clf_svm = SVC(class_weight=opt_params_svm['class_weight'],
                          kernel=opt_params_svm['kernel'], \
                          C=opt_params_svm['C'],\
                          gamma=opt_params_svm['gamma'], shrinking=False, \
                          probability=False, random_state=0, max_iter=-1, \
                          tol=X.shape[-1] * 1e-3)

            clf_svm = clf_svm.fit(X, Y)

            pred_test_svm = clf_svm.decision_function(X_OOS)
            probs_oos_fraud_svm = np.exp(pred_test_svm) / (1 + np.exp(pred_test_svm))

            roc_svm[m] = roc_auc_score(Y_OOS, probs_oos_fraud_svm)

            cutoff_OOS_svm = np.percentile(probs_oos_fraud_svm, 99)
            sensitivity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_svm >= cutoff_OOS_svm)
            ndcg_svm1[m] = ndcg_k(Y_OOS, probs_oos_fraud_svm, 99)
            FN_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                            Y_OOS == 1))
            FP_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                            Y_OOS == 0))

            ecm_svm1[m] = C_FN * P_f * FN_svm2 / n_P + C_FP * P_nf * FP_svm2 / n_N

            # Logistic Regression – Dechow et al (2011)
            X_lr = add_constant(X)
            X_OOS_lr = add_constant(X_OOS)
            clf_lr = Logit(Y, X_lr)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS_lr)

            roc_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_lr)
            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)
            sensitivity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                           Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                                           Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                     Y_OOS == 1)) / np.sum(probs_oos_fraud_lr >= cutoff_OOS_lr)
            ndcg_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_lr, 99)
            FN_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                           Y_OOS == 1))
            FP_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                           Y_OOS == 0))
            ecm_lr1[m] = C_FN * P_f * FN_lr2 / n_P + C_FP * P_nf * FP_lr2 / n_N


            # LogitBoost
            base_lr = LogisticRegression(random_state=0, solver='newton-cg')

            clf_ada = AdaBoostClassifier(n_estimators=opt_params_ada['n_estimators'], \
                                         learning_rate=opt_params_ada['learning_rate'], \
                                         estimator=base_lr, random_state=0)
            clf_ada = clf_ada.fit(X, Y)
            probs_oos_fraud_ada = clf_ada.predict_proba(X_OOS)[:, -1]

            roc_ada[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada)
            cutoff_OOS_ada = np.percentile(probs_oos_fraud_ada, 99)
            sensitivity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_ada >= cutoff_OOS_ada)
            ndcg_ada1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada, 99)
            FN_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                            Y_OOS == 1))
            FP_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                            Y_OOS == 0))
            ecm_ada1[m] = C_FN * P_f * FN_ada2 / n_P + C_FP * P_nf * FP_ada2 / n_N


            # Fused approach
            weight_ser = np.array([score_svm, score_lr, score_ada])
            weight_ser = weight_ser / np.sum(weight_ser)

            probs_oos_fraud_svm = (1 + np.exp(-1 * probs_oos_fraud_svm)) ** -1
            probs_oos_fraud_lr = (1 + np.exp(-1 * probs_oos_fraud_lr)) ** -1
            probs_oos_fraud_ada = (1 + np.exp(-1 * probs_oos_fraud_ada)) ** -1
            clf_fused = np.dot(np.array([probs_oos_fraud_svm, \
                                         probs_oos_fraud_lr, probs_oos_fraud_ada]).T, weight_ser)

            probs_oos_fraud_fused = clf_fused
            roc_fused[m] = roc_auc_score(Y_OOS, probs_oos_fraud_fused)
            cutoff_OOS_fused = np.percentile(probs_oos_fraud_fused, 99)
            sensitivity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                              Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                                              Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                        Y_OOS == 1)) / np.sum(probs_oos_fraud_fused >= cutoff_OOS_fused)
            ndcg_fused1[m] = ndcg_k(Y_OOS, probs_oos_fraud_fused, 99)
            FN_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                              Y_OOS == 1))
            FP_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                              Y_OOS == 0))
            ecm_fused1[m] = C_FN * P_f * FN_fused2 / n_P + C_FP * P_nf * FP_fused2 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1


        f1_score_svm1 = 2 * (precision_svm1 * sensitivity_OOS_svm1) / \
                        (precision_svm1 + sensitivity_OOS_svm1 + 1e-8)
        f1_score_lr1 = 2 * (precision_lr1 * sensitivity_OOS_lr1) / \
                       (precision_lr1 + sensitivity_OOS_lr1 + 1e-8)
        f1_score_ada1 = 2 * (precision_ada1 * sensitivity_OOS_ada1) / \
                        (precision_ada1 + sensitivity_OOS_ada1 + 1e-8)
        f1_score_fused1 = 2 * (precision_fused1 * sensitivity_OOS_fused1) / \
                          (precision_fused1 + sensitivity_OOS_fused1 + 1e-8)


        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['SVM', 'LR', 'LogitBoost', 'FUSED']


        perf_tbl_general['Roc'] = [str(np.round(
            np.mean(roc_svm) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused) * 100, 2)) + '%)']

        perf_tbl_general['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_fused1) * 100, 2)) + '%)']


        perf_tbl_general['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_fused1) * 100, 2)) + '%)']


        perf_tbl_general['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1) * 100, 2)) + '%)']

        perf_tbl_general['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1) * 100, 2)) + '%)']

        perf_tbl_general['NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1) * 100, 2)) + '%)']

        perf_tbl_general['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1) * 100, 2)) + '%)']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_11ratio_temporal.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


    def ratio_biased(self, C_FN=30, C_FP=1):
        """
        This code uses Bao et al. (2020)'s serial fraud treatment for Table 9.
        RUSBoost and SVM-FK23 are not included.

        Methodological choices:
            - 11 financial ratios from Dechow et al.(2011)
            - 10-fold validation
            - Bao et al. (2020)'s serial fraud treatment
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, -14:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]  # 1991
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]  # 2010

        # Setting the cross-validation setting
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)

        X_CV = tbl_year_IS_CV.iloc[:, -11:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        # redo cross-validation if you wish
        if cv_type == 'kfold':
            if cross_val == True:

                # optimise LogitBoost
                print('Grid search hyperparameter optimisation started for AdaBoost')
                t1 = datetime.now()

                best_perf_ada = 0

                base_lr = LogisticRegression(random_state=0, solver='newton-cg')

                pipe_ada = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_ada', AdaBoostClassifier(estimator=base_lr, random_state=0))])
                estimators = list(range(10, 3001, 10))
                learning_rates = [x/1000 for x in range(10,1001,10)]
                param_grid_ada = {'base_mdl_ada__n_estimators': estimators, \
                                  'base_mdl_ada__learning_rate': learning_rates}

                clf_ada = GridSearchCV(pipe_ada, param_grid_ada, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_ada.fit(X_CV, Y_CV)
                score_ada = clf_ada.best_score_
                if score_ada >= best_perf_ada:
                    best_perf_ada = score_ada
                    opt_params_ada = clf_ada.best_params_

                print('LogitBoost: The optimal number of estimators is ' + \
                      str(opt_params_ada['base_mdl_ada__n_estimators']) + ', and learning rate ' + \
                      str(opt_params_ada['base_mdl_ada__learning_rate']))

                print('Computing CV ROC for LR ...')
                score_lr = []
                for m in range(0, k_fold):
                    train_sample, test_sample = train_test_split(Y_CV, test_size=1 /
                                                                                 k_fold, shuffle=False, random_state=m)
                    X_train = X_CV.iloc[train_sample.index]
                    X_train = add_constant(X_train)
                    Y_train = train_sample
                    X_test = X_CV.iloc[test_sample.index]
                    X_test = add_constant(X_test)
                    Y_test = test_sample

                    logit_model = Logit(Y_train, X_train)
                    logit_model = logit_model.fit(disp=0)
                    pred_LR_CV = logit_model.predict(X_test)
                    score_lr.append(roc_auc_score(Y_test, pred_LR_CV))

                score_lr = np.mean(score_lr)



                # optimize SVM grid

                print('Grid search hyperparameter optimisation started for SVM')
                t1 = datetime.now()

                pipe_svm = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_svm', SVC(shrinking=False, \
                                                          probability=False, random_state=0, max_iter=-1, \
                                                          tol=X_CV.shape[-1] * 1e-3))])
                C = [0.001, 0.01, 0.1, 1, 10, 100]
                gamma = [0.0001, 0.001, 0.01, 0.1]
                kernel = ['rbf', 'linear', 'poly']
                class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]
                param_grid_svm = {'base_mdl_svm__kernel': kernel, \
                                  'base_mdl_svm__C': C, \
                                  'base_mdl_svm__gamma': gamma, \
                                  'base_mdl_svm__class_weight': class_weight}

                clf_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_svm.fit(X_CV, Y_CV)
                opt_params_svm = clf_svm.best_params_
                gamma_opt = opt_params_svm['base_mdl_svm__gamma']
                cw_opt = opt_params_svm['base_mdl_svm__class_weight']
                c_opt = opt_params_svm['base_mdl_svm__C']
                kernel_opt = opt_params_svm['base_mdl_svm__kernel']
                score_svm = clf_svm.best_score_

                t2 = datetime.now()
                dt = t2 - t1
                print('SVM CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('SVM: The optimal C+/C- ratio is ' + str(cw_opt) + ', gamma is' + str(
                    gamma_opt) + ', C is' + str(c_opt) + ',score is' + str(score_svm) + \
                      ', kernel is' + str(kernel_opt))


        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)

        roc_svm = np.zeros(len(range_oos))
        sensitivity_OOS_svm1 = np.zeros(len(range_oos))
        specificity_OOS_svm1 = np.zeros(len(range_oos))
        precision_svm1 = np.zeros(len(range_oos))
        ndcg_svm1 = np.zeros(len(range_oos))
        ecm_svm1 = np.zeros(len(range_oos))

        roc_lr = np.zeros(len(range_oos))
        sensitivity_OOS_lr1 = np.zeros(len(range_oos))
        specificity_OOS_lr1 = np.zeros(len(range_oos))
        precision_lr1 = np.zeros(len(range_oos))
        ndcg_lr1 = np.zeros(len(range_oos))
        ecm_lr1 = np.zeros(len(range_oos))

        roc_ada = np.zeros(len(range_oos))
        sensitivity_OOS_ada1 = np.zeros(len(range_oos))
        specificity_OOS_ada1 = np.zeros(len(range_oos))
        precision_ada1 = np.zeros(len(range_oos))
        ndcg_ada1 = np.zeros(len(range_oos))
        ecm_ada1 = np.zeros(len(range_oos))

        roc_fused = np.zeros(len(range_oos))
        sensitivity_OOS_fused1 = np.zeros(len(range_oos))
        specificity_OOS_fused1 = np.zeros(len(range_oos))
        precision_fused1 = np.zeros(len(range_oos))
        ndcg_fused1 = np.zeros(len(range_oos))
        ecm_fused1 = np.zeros(len(range_oos))

        m = 0
        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -11:]
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS = (X_OOS - mean) / std
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            misstate_test = np.unique(tbl_year_OOS[tbl_year_OOS['AAER_DUMMY'] == 1]['gvkey'])
            Y[np.isin(tbl_year_IS['gvkey'], misstate_test)] = 0

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            # Support Vector Machines

            clf_svm = SVC(class_weight=opt_params_svm['base_mdl_svm__class_weight'],
                          kernel=opt_params_svm['base_mdl_svm__kernel'], shrinking=False, \
                          C=opt_params_svm['base_mdl_svm__C'], gamma=opt_params_svm['base_mdl_svm__gamma'],\
                          probability=False, random_state=0, max_iter=-1, \
                          tol=X.shape[-1] * 1e-3)

            clf_svm = clf_svm.fit(X, Y)

            pred_test_svm = clf_svm.decision_function(X_OOS)
            probs_oos_fraud_svm = np.exp(pred_test_svm) / (1 + np.exp(pred_test_svm))

            roc_svm[m] = roc_auc_score(Y_OOS, probs_oos_fraud_svm)

            cutoff_OOS_svm = np.percentile(probs_oos_fraud_svm, 99)
            sensitivity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_svm >= cutoff_OOS_svm)
            ndcg_svm1[m] = ndcg_k(Y_OOS, probs_oos_fraud_svm, 99)

            FN_svm1 = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                            Y_OOS == 1))
            FP_svm1 = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                            Y_OOS == 0))

            ecm_svm1[m] = C_FN * P_f * FN_svm1 / n_P + C_FP * P_nf * FP_svm1 / n_N

            # Logistic Regression – Dechow et al (2011)
            X_lr = add_constant(X)
            X_OOS_lr = add_constant(X_OOS)
            clf_lr = Logit(Y, X_lr)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS_lr)

            roc_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_lr)

            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)
            sensitivity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                           Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                                           Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                     Y_OOS == 1)) / np.sum(probs_oos_fraud_lr >= cutoff_OOS_lr)
            ndcg_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_lr, 99)

            FN_lr1 = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                           Y_OOS == 1))
            FP_lr1 = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                           Y_OOS == 0))

            ecm_lr1[m] = C_FN * P_f * FN_lr1 / n_P + C_FP * P_nf * FP_lr1 / n_N


            # LogitBoost
            base_lr = LogisticRegression(random_state=0, solver='newton-cg')

            clf_ada = AdaBoostClassifier(n_estimators=opt_params_ada['base_mdl_ada__n_estimators'], \
                                         learning_rate=opt_params_ada['base_mdl_ada__learning_rate'], \
                                         estimator=base_lr, random_state=0)
            clf_ada = clf_ada.fit(X, Y)
            probs_oos_fraud_ada = clf_ada.predict_proba(X_OOS)[:, -1]

            roc_ada[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada)
            cutoff_OOS_ada = np.percentile(probs_oos_fraud_ada, 99)
            sensitivity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_ada >= cutoff_OOS_ada)
            ndcg_ada1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada, 99)

            FN_ada1 = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                            Y_OOS == 1))
            FP_ada1 = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                            Y_OOS == 0))

            ecm_ada1[m] = C_FN * P_f * FN_ada1 / n_P + C_FP * P_nf * FP_ada1 / n_N


            # Fused approach
            weight_ser = np.array([score_svm, score_lr, score_ada])
            weight_ser = weight_ser / np.sum(weight_ser)
            probs_oos_fraud_svm = (1 + np.exp(-1 * probs_oos_fraud_svm)) ** -1

            probs_oos_fraud_lr = (1 + np.exp(-1 * probs_oos_fraud_lr)) ** -1

            probs_oos_fraud_ada = (1 + np.exp(-1 * probs_oos_fraud_ada)) ** -1

            clf_fused = np.dot(np.array([probs_oos_fraud_svm, \
                                         probs_oos_fraud_lr, probs_oos_fraud_ada]).T, weight_ser)

            probs_oos_fraud_fused = clf_fused

            roc_fused[m] = roc_auc_score(Y_OOS, probs_oos_fraud_fused)

            cutoff_OOS_fused = np.percentile(probs_oos_fraud_fused, 99)
            sensitivity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                              Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                                              Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                        Y_OOS == 1)) / np.sum(probs_oos_fraud_fused >= cutoff_OOS_fused)
            ndcg_fused1[m] = ndcg_k(Y_OOS, probs_oos_fraud_fused, 99)

            FN_fused1 = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                              Y_OOS == 1))
            FP_fused1 = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                              Y_OOS == 0))

            ecm_fused1[m] = C_FN * P_f * FN_fused1 / n_P + C_FP * P_nf * FP_fused1 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        f1_score_svm1 = 2 * (precision_svm1 * sensitivity_OOS_svm1) / \
                        (precision_svm1 + sensitivity_OOS_svm1 + 1e-8)
        f1_score_lr1 = 2 * (precision_lr1 * sensitivity_OOS_lr1) / \
                       (precision_lr1 + sensitivity_OOS_lr1 + 1e-8)
        f1_score_ada1 = 2 * (precision_ada1 * sensitivity_OOS_ada1) / \
                        (precision_ada1 + sensitivity_OOS_ada1 + 1e-8)
        f1_score_fused1 = 2 * (precision_fused1 * sensitivity_OOS_fused1) / \
                          (precision_fused1 + sensitivity_OOS_fused1 + 1e-8)

        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['SVM', 'LR', 'LogitBoost', 'FUSED']
        perf_tbl_general['Roc'] = [str(np.round(
            np.mean(roc_svm) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused) * 100, 2)) + '%)']

        perf_tbl_general['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_fused1) * 100, 2)) + '%)']

        perf_tbl_general['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_fused1) * 100, 2)) + '%)']

        perf_tbl_general['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1) * 100, 2)) + '%)']

        perf_tbl_general['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1) * 100, 2)) + '%)']

        perf_tbl_general['NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1) * 100, 2)) + '%)']

        perf_tbl_general['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1) * 100, 2)) + '%)']


        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_11ratio_biased.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


    def RUS_28(self, C_FN=30, C_FP=1):
        """
        This code uses Bao et al. (2020)'s RUSBoost model for Table 5 with 28 raw accounting items.
        Combined with RUS_11 Table 7 can be replicated.

        Methodological choices:
            - 28 raw accounting items from Bao et al.(2020)
            - 10-fold validation
            - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-14]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        range_oos = range(start_OOS_year, end_OOS_year + 1)
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= sample_start)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -28:]
        X_CV = X_CV.apply(lambda x: x/np.linalg.norm(x))
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        # optimize RUSBoost number of estimators
        if cv_type == 'kfold':
            if cross_val == True:

                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1 = datetime.now()
                n_estimators = list(range(10,3001,10))
                learning_rate = list(x/1000 for x in range(10, 1001,10))

                base_tree = DecisionTreeClassifier(min_samples_leaf=5)

                bao_RUSboost = RUSBoostClassifier(estimator=base_tree, \
                                                  sampling_strategy=1, random_state=0)

                param_grid_rusboost = {'n_estimators': n_estimators,
                                       'learning_rate': learning_rate}

                clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_rus.fit(X_CV, Y_CV)
                opt_params_rus = clf_rus.best_params_
                n_opt_rus = opt_params_rus['n_estimators']
                r_opt_rus = opt_params_rus['learning_rate']
                score_rus = clf_rus.best_score_
                t2 = datetime.now()
                dt = t2 - t1
                print('RUSBoost CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('RUSBoost: The optimal number of estimators is ' + str(n_opt_rus) + 'learning rate=' + str(
                    r_opt_rus))


        roc_rusboost = np.zeros(len(range_oos))
        roc_rusboost_training = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1 = np.zeros(len(range_oos))
        sensitivity_rusboost1_training = np.zeros(len(range_oos))
        specificity_OOS_rusboost1 = np.zeros(len(range_oos))
        specificity_rusboost1_training = np.zeros(len(range_oos))
        precision_rusboost1 = np.zeros(len(range_oos))
        precision_rusboost1_training = np.zeros(len(range_oos))
        ndcg_rusboost1 = np.zeros(len(range_oos))
        ndcg_rusboost1_training = np.zeros(len(range_oos))
        ecm_rusboost1 = np.zeros(len(range_oos))
        ecm_rusboost1_training = np.zeros(len(range_oos))


        m = 0

        for yr in range_oos:
            t1 = datetime.now()

            year_start_IS = sample_start
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[reduced_tbl.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -28:]
            X_rus = X.apply(lambda x: x / np.linalg.norm(x))

            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -28:]
            X_OOS_rus = X_OOS.apply(lambda x: x / np.linalg.norm(x))

            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            n_P_training = np.sum(Y == 1)
            n_N_training = np.sum(Y == 0)


            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=n_opt_rus, \
                                              learning_rate=r_opt_rus, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X_rus, Y)

            #performance on training sample- rusboost
            probs_fraud_rusboost = clf_rusboost.predict_proba(X_rus)[:, -1]

            roc_rusboost_training[m] = roc_auc_score(Y, probs_fraud_rusboost)

            cutoff_rusboost = np.percentile(probs_fraud_rusboost, 99)
            sensitivity_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                                      Y == 1)) / np.sum(Y)
            specificity_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost < cutoff_rusboost, \
                                                                      Y == 0)) / np.sum(Y == 0)
            precision_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                                    Y == 1)) / np.sum(
                probs_fraud_rusboost >= cutoff_rusboost)
            ndcg_rusboost1_training[m] = ndcg_k(Y, probs_fraud_rusboost, 99)

            FN_rusboost3 = np.sum(np.logical_and(probs_fraud_rusboost < cutoff_rusboost, \
                                                 Y == 1))
            FP_rusboost3 = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                 Y == 0))

            ecm_rusboost1_training[m] = C_FN * P_f * FN_rusboost3 / n_P_training + C_FP * P_nf * FP_rusboost3 / n_N_training

            # performance on testing sample- rusboost
            probs_oos_fraud_rusboost = clf_rusboost.predict_proba(X_OOS_rus)[:, -1]
            roc_rusboost[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rusboost)
            cutoff_OOS_rusboost = np.percentile(probs_oos_fraud_rusboost, 99)
            sensitivity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_rusboost >= cutoff_OOS_rusboost)
            ndcg_rusboost1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rusboost, 99)
            FN_rusboost2 = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                 Y_OOS == 1))
            FP_rusboost2 = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                 Y_OOS == 0))
            ecm_rusboost1[m] = C_FN * P_f * FN_rusboost2 / n_P + C_FP * P_nf * FP_rusboost2 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1


        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['RUSBoost-28']

        perf_tbl_general['Training Roc'] = [str(np.round(
            np.mean(roc_rusboost_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(roc_rusboost_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Roc'] = [str(np.round(
            np.mean(roc_rusboost) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(roc_rusboost) * 100, 2)) + '%)']

        gap_roc_rusboost = roc_rusboost - roc_rusboost_training
        mean_gap_roc_rusboost = np.round(np.mean(gap_roc_rusboost) * 100, 2)

        perf_tbl_general['Gap Roc'] = [str(mean_gap_roc_rusboost) + '%']

        perf_tbl_general['Training Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_rusboost1_training) * 100, 2)) + '% (' + \
                                                            str(np.round(np.std(sensitivity_rusboost1_training) * 100,
                                                                         2)) + '%)']

        perf_tbl_general['Testing Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                           str(np.round(np.std(sensitivity_OOS_rusboost1) * 100,
                                                                        2)) + '%)']

        gap_sensitivity_rusboost = sensitivity_OOS_rusboost1 - sensitivity_rusboost1_training

        mean_gap_sensitivity_rusboost = np.round(np.mean(gap_sensitivity_rusboost) * 100, 2)

        perf_tbl_general['Gap Sensitivity'] = [str(mean_gap_sensitivity_rusboost) + '%']

        perf_tbl_general[('Training Specificity @ 1 Prc')] = [str(np.round(
            np.mean(specificity_rusboost1_training) * 100, 2)) + '% (' + \
                                                              str(np.round(np.std(specificity_rusboost1_training) * 100,
                                                                           2)) + '%)']

        perf_tbl_general['Testing Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                           str(np.round(np.std(specificity_OOS_rusboost1) * 100,
                                                                        2)) + '%)']

        gap_specificity_rusboost = specificity_OOS_rusboost1 - specificity_rusboost1_training

        mean_gap_specificity_rusboost = np.round(np.mean(gap_specificity_rusboost) * 100, 2)

        perf_tbl_general['Gap Specificity'] = [str(mean_gap_specificity_rusboost) + '%']

        perf_tbl_general['Training Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rusboost1_training) * 100, 2)) + '% (' + \
                                                          str(np.round(np.std(precision_rusboost1_training) * 100,
                                                                       2)) + '%)']

        perf_tbl_general['Testing Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rusboost1) * 100, 2)) + '% (' + \
                                                         str(np.round(np.std(precision_rusboost1) * 100, 2)) + '%)']

        gap_precision_rusboost = precision_rusboost1 - precision_rusboost1_training
        mean_gap_precision_rusboost = np.round(np.mean(gap_precision_rusboost) * 100, 2)

        perf_tbl_general['Gap Precision'] = [str(mean_gap_precision_rusboost) + '%']

        f1_score_rusboost1_training = 2 * (precision_rusboost1_training * sensitivity_rusboost1_training) / \
                                      (precision_rusboost1_training + sensitivity_rusboost1_training + 1e-8)
        f1_score_rusboost1 = 2 * (precision_rusboost1 * sensitivity_OOS_rusboost1) / \
                             (precision_rusboost1 + sensitivity_OOS_rusboost1 + 1e-8)

        perf_tbl_general['Training F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rusboost1_training) * 100, 2)) + '% (' + \
                                                         str(np.round(np.std(f1_score_rusboost1_training) * 100,
                                                                      2)) + '%)']

        perf_tbl_general['Testing F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rusboost1) * 100, 2)) + '% (' + \
                                                        str(np.round(np.std(f1_score_rusboost1) * 100, 2)) + '%)']

        gap_f1_score_rusboost = f1_score_rusboost1 - f1_score_rusboost1_training

        mean_gap_f1_score_rusboost = np.round(np.mean(gap_f1_score_rusboost) * 100, 2)

        perf_tbl_general['Gap F1 Score'] = [str(mean_gap_f1_score_rusboost) + '%']

        perf_tbl_general['Training NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rusboost1_training) * 100, 2)) + '% (' + \
                                                     str(np.round(np.std(ndcg_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rusboost1) * 100, 2)) + '% (' + \
                                                    str(np.round(np.std(ndcg_rusboost1) * 100, 2)) + '%)']

        gap_ndcg_rusboost = ndcg_rusboost1 - ndcg_rusboost1_training
        mean_gap_ndcg_rusboost = np.round(np.mean(gap_ndcg_rusboost) * 100, 2)
        perf_tbl_general['Gap NDCG'] = [str(mean_gap_ndcg_rusboost) + '%']

        perf_tbl_general['Training ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rusboost1_training) * 100, 2)) + '% (' + \
                                                    str(np.round(np.std(ecm_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rusboost1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(ecm_rusboost1) * 100, 2)) + '%)']

        gap_ecm_rusboost = ecm_rusboost1 - ecm_rusboost1_training
        mean_gap_ecm_rusboost = np.round(np.mean(gap_ecm_rusboost) * 100, 2)
        perf_tbl_general['Gap ECM'] = [str(mean_gap_ecm_rusboost) + '%']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_RUS28_with_gap.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


    def RUS_11(self, C_FN=30, C_FP=1):
        """
        This code uses Bao et al. (2020)'s RUSBoost model for Table 6 with 11 financial ratios.
        Combined with RUS_28 Table 7 can be replicated.

        Methodological choices:
            - 11 financial ratios from Dechow et al.(2011)
            - 10-fold validation
            - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, -14:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        range_oos = range(start_OOS_year, end_OOS_year + 1)
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= sample_start)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)

        X_CV = tbl_year_IS_CV.iloc[:, -11:]
        X_CV = X_CV.apply(lambda x: x / np.linalg.norm(x))
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        # optimize RUSBoost number of estimators
        if cv_type == 'kfold':
            if cross_val == True:

                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1 = datetime.now()
                n_estimators = list(range(10, 3001, 10))
                learning_rate = list(x / 1000 for x in range(10, 1001, 10))
                param_grid_rusboost = {'n_estimators': n_estimators,
                                       'learning_rate': learning_rate}

                base_tree = DecisionTreeClassifier(min_samples_leaf=5)
                bao_RUSboost = RUSBoostClassifier(estimator=base_tree, \
                                                  sampling_strategy=1, random_state=0)
                clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_rus.fit(X_CV, Y_CV)
                opt_params_rus = clf_rus.best_params_
                n_opt_rus = opt_params_rus['n_estimators']
                r_opt_rus = opt_params_rus['learning_rate']
                score_rus = clf_rus.best_score_
                t2 = datetime.now()
                dt = t2 - t1
                print('RUSBoost CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('RUSBoost: The optimal number of estimators is ' + str(n_opt_rus) + \
                      'The optimal learning rate is '+ str(r_opt_rus))


        roc_rusboost = np.zeros(len(range_oos))
        roc_rusboost_training = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1 = np.zeros(len(range_oos))
        sensitivity_rusboost1_training = np.zeros(len(range_oos))
        specificity_OOS_rusboost1 = np.zeros(len(range_oos))
        specificity_rusboost1_training = np.zeros(len(range_oos))
        precision_rusboost1 = np.zeros(len(range_oos))
        precision_rusboost1_training = np.zeros(len(range_oos))
        ndcg_rusboost1 = np.zeros(len(range_oos))
        ndcg_rusboost1_training = np.zeros(len(range_oos))
        ecm_rusboost1 = np.zeros(len(range_oos))
        ecm_rusboost1_training = np.zeros(len(range_oos))


        m = 0

        for yr in range_oos:
            t1 = datetime.now()

            year_start_IS = sample_start
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[reduced_tbl.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -11:]
            X_rus = X.apply(lambda x: x / np.linalg.norm(x))

            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS_rus = X_OOS.apply(lambda x: x / np.linalg.norm(x))

            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            n_P_training = np.sum(Y == 1)
            n_N_training = np.sum(Y == 0)

            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=n_opt_rus, \
                                              learning_rate=r_opt_rus, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X_rus, Y)


            # performance on training sample- rusboost
            probs_fraud_rusboost = clf_rusboost.predict_proba(X_rus)[:, -1]
            roc_rusboost_training[m] = roc_auc_score(Y, probs_fraud_rusboost)
            cutoff_rusboost = np.percentile(probs_fraud_rusboost, 99)
            sensitivity_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                                 Y == 1)) / np.sum(Y)
            specificity_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost < cutoff_rusboost, \
                                                                 Y == 0)) / np.sum(Y == 0)
            precision_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                           Y == 1)) / np.sum(probs_fraud_rusboost >= cutoff_rusboost)
            ndcg_rusboost1_training[m] = ndcg_k(Y, probs_fraud_rusboost, 99)

            FN_rusboost3 = np.sum(np.logical_and(probs_fraud_rusboost < cutoff_rusboost, \
                                                 Y == 1))
            FP_rusboost3 = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                 Y == 0))
            ecm_rusboost1_training[m] = C_FN * P_f * FN_rusboost3 / n_P_training + C_FP * P_nf * FP_rusboost3 / n_N_training


            # performance on testing sample- rusboost
            probs_oos_fraud_rusboost = clf_rusboost.predict_proba(X_OOS_rus)[:, -1]
            roc_rusboost[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rusboost)
            cutoff_OOS_rusboost = np.percentile(probs_oos_fraud_rusboost, 99)
            sensitivity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_rusboost >= cutoff_OOS_rusboost)
            ndcg_rusboost1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rusboost, 99)
            FN_rusboost2 = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                 Y_OOS == 1))
            FP_rusboost2 = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                 Y_OOS == 0))

            ecm_rusboost1[m] = C_FN * P_f * FN_rusboost2 / n_P + C_FP * P_nf * FP_rusboost2 / n_N


            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['RUSBoost-11']

        perf_tbl_general['Training Roc'] = [str(np.round(
            np.mean(roc_rusboost_training) * 100, 2)) + '% (' + \
                                  str(np.round(np.std(roc_rusboost_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Roc'] = [str(np.round(
            np.mean(roc_rusboost) * 100, 2)) + '% (' + \
                                  str(np.round(np.std(roc_rusboost) * 100, 2)) + '%)']

        gap_roc_rusboost = roc_rusboost - roc_rusboost_training

        mean_gap_roc_rusboost = np.round(np.mean(gap_roc_rusboost) * 100, 2)
        perf_tbl_general['Gap Roc'] = [str(mean_gap_roc_rusboost) + '%']

        perf_tbl_general['Training Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_rusboost1_training) * 100, 2)) + '% (' + \
                                                          str(np.round(np.std(sensitivity_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(sensitivity_OOS_rusboost1) * 100, 2)) + '%)']

        gap_sensitivity_rusboost = sensitivity_OOS_rusboost1 - sensitivity_rusboost1_training

        mean_gap_sensitivity_rusboost = np.round(np.mean(gap_sensitivity_rusboost) * 100, 2)
        perf_tbl_general['Gap Sensitivity'] = [str(mean_gap_sensitivity_rusboost) + '%']

        perf_tbl_general[('Training Specificity @ 1 Prc')] = [str(np.round(
            np.mean(specificity_rusboost1_training) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(specificity_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(specificity_OOS_rusboost1) * 100, 2)) + '%)']

        gap_specificity_rusboost = specificity_OOS_rusboost1 - specificity_rusboost1_training

        mean_gap_specificity_rusboost = np.round(np.mean(gap_specificity_rusboost) * 100, 2)
        perf_tbl_general['Gap Specificity'] = [str(mean_gap_specificity_rusboost) + '%']

        perf_tbl_general['Training Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rusboost1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(precision_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rusboost1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(precision_rusboost1) * 100, 2)) + '%)']

        gap_precision_rusboost = precision_rusboost1 - precision_rusboost1_training

        mean_gap_precision_rusboost = np.round(np.mean(gap_precision_rusboost) * 100, 2)
        perf_tbl_general['Gap Precision'] = [str(mean_gap_precision_rusboost) + '%']

        f1_score_rusboost1_training = 2 * (precision_rusboost1_training * sensitivity_rusboost1_training) / \
                             (precision_rusboost1_training + sensitivity_rusboost1_training + 1e-8)
        f1_score_rusboost1 = 2 * (precision_rusboost1 * sensitivity_OOS_rusboost1) / \
                             (precision_rusboost1 + sensitivity_OOS_rusboost1 + 1e-8)



        perf_tbl_general['Training F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rusboost1_training) * 100, 2)) + '% (' + \
                                               str(np.round(np.std(f1_score_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rusboost1) * 100, 2)) + '% (' + \
                                               str(np.round(np.std(f1_score_rusboost1) * 100, 2)) + '%)']

        gap_f1_score_rusboost = f1_score_rusboost1 - f1_score_rusboost1_training

        mean_gap_f1_score_rusboost = np.round(np.mean(gap_f1_score_rusboost) * 100, 2)
        perf_tbl_general['Gap F1 Score'] = [str(mean_gap_f1_score_rusboost) + '%']

        perf_tbl_general['Training NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rusboost1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ndcg_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rusboost1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ndcg_rusboost1) * 100, 2)) + '%)']

        gap_ndcg_rusboost = ndcg_rusboost1 - ndcg_rusboost1_training

        mean_gap_ndcg_rusboost = np.round(np.mean(gap_ndcg_rusboost) * 100, 2)
        perf_tbl_general['Gap NDCG'] = [str(mean_gap_ndcg_rusboost) + '%']

        perf_tbl_general['Training ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rusboost1_training) * 100, 2)) + '% (' + \
                                          str(np.round(np.std(ecm_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rusboost1) * 100, 2)) + '% (' + \
                                          str(np.round(np.std(ecm_rusboost1) * 100, 2)) + '%)']

        gap_ecm_rusboost = ecm_rusboost1 - ecm_rusboost1_training

        mean_gap_ecm_rusboost = np.round(np.mean(gap_ecm_rusboost) * 100, 2)
        perf_tbl_general['Gap ECM'] = [str(mean_gap_ecm_rusboost) + '%']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_RUS11_with_gap.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')

    def RUS28_temporal(self, C_FN=30, C_FP=1):
        """
        This code uses temporal validation selecting optimal hyperparameters for Table 8 with RUSBoost model.

        Methodological choices:
            - 28 raw accounting items from Bao et al.(2020)
            - temporal validation
            - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.tree import DecisionTreeClassifier
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-14]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        range_oos = range(start_OOS_year, end_OOS_year + 1)
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= sample_start)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)

        X_CV = tbl_year_IS_CV.iloc[:, -28:]

        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f


        cutoff_temporal = 2001 - temp_year
        X_CV_train = X_CV[tbl_year_IS_CV['fyear'] < cutoff_temporal]
        X_CV_train = X_CV_train.apply(lambda x: x / np.linalg.norm(x))
        Y_CV_train = Y_CV[tbl_year_IS_CV['fyear'] < cutoff_temporal]

        X_CV_test = X_CV[tbl_year_IS_CV['fyear'] >= cutoff_temporal]
        X_CV_test = X_CV_test.apply(lambda x: x / np.linalg.norm(x))
        Y_CV_test = Y_CV[tbl_year_IS_CV['fyear'] >= cutoff_temporal]

        print('Grid search hyperparameter optimisation started for RUSBoost')
        t1 = datetime.now()
        n_estimators = list(range(10, 3001, 10))
        learning_rate = list(x / 1000 for x in range(10, 1001, 10))
        param_grid_rusboost = {'n_estimators': n_estimators,
                               'learning_rate': learning_rate}

        temp_rusboost = {'n_estimators': [], 'learning_rate': [], 'score': []}

        for n in param_grid_rusboost['n_estimators']:
            for r in param_grid_rusboost['learning_rate']:
                temp_rusboost['n_estimators'].append(n)
                temp_rusboost['learning_rate'].append(r)
                base_tree = DecisionTreeClassifier(min_samples_leaf=5)
                bao_RUSboost = RUSBoostClassifier(estimator=base_tree, \
                                                  sampling_strategy=1, n_estimators=n,
                                                  learning_rate=r, random_state=0).fit(
                    X_CV_train, Y_CV_train)

                predicted_test_RUS = bao_RUSboost.predict_proba(X_CV_test)[:, -1]

                temp_rusboost['score'].append(roc_auc_score(Y_CV_test, predicted_test_RUS))

        idx_opt_rus = temp_rusboost['score'].index(np.max(temp_rusboost['score']))
        n_opt_rus = temp_rusboost['n_estimators'][idx_opt_rus]
        r_opt_rus = temp_rusboost['learning_rate'][idx_opt_rus]
        score_rus = temp_rusboost['score'][idx_opt_rus]

        t2 = datetime.now()
        dt = t2 - t1
        print('RUSBoost Temporal validation finished after ' + str(dt.total_seconds()) + ' sec')
        print('RUSBoost: The optimal number of estimators is ' + str(n_opt_rus) +\
              ' learning rate is,'+ str(r_opt_rus))


        roc_rusboost = np.zeros(len(range_oos))
        specificity_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost = np.zeros(len(range_oos))
        precision_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1 = np.zeros(len(range_oos))
        specificity_OOS_rusboost1 = np.zeros(len(range_oos))
        precision_rusboost1 = np.zeros(len(range_oos))
        ndcg_rusboost1 = np.zeros(len(range_oos))
        ecm_rusboost1 = np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()

            year_start_IS = sample_start
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[reduced_tbl.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -28:]
            X = X.apply(lambda x: x / np.linalg.norm(x))
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -28:]
            X_OOS = X_OOS.apply(lambda x: x / np.linalg.norm(x))
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)
            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=n_opt_rus, \
                                              learning_rate=r_opt_rus, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X, Y)

            probs_oos_fraud_rusboost = clf_rusboost.predict_proba(X_OOS)[:, -1]

            labels_rusboost = clf_rusboost.predict(X_OOS)

            roc_rusboost[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rusboost)
            specificity_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 0, Y_OOS == 0)) / \
                                      np.sum(Y_OOS == 0)

            sensitivity_OOS_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, \
                                                                Y_OOS == 1)) / np.sum(Y_OOS)
            precision_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, Y_OOS == 1)) / np.sum(labels_rusboost)

            cutoff_OOS_rusboost = np.percentile(probs_oos_fraud_rusboost, 99)
            sensitivity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_rusboost >= cutoff_OOS_rusboost)
            ndcg_rusboost1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rusboost, 99)

            FN_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                 Y_OOS == 1))
            FP_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                 Y_OOS == 0))

            ecm_rusboost1[m] = C_FN * P_f * FN_rusboost1 / n_P + C_FP * P_nf * FP_rusboost1 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1


        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['RUSBoost-28']

        perf_tbl_general['Roc'] = str(np.round(
            np.mean(roc_rusboost) * 100, 2)) + '% (' + \
                                  str(np.round(np.std(roc_rusboost) * 100, 2)) + '%)'

        perf_tbl_general['Sensitivity @ 1 Prc'] = str(np.round(
            np.mean(sensitivity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(sensitivity_OOS_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['Specificity @ 1 Prc'] = str(np.round(
            np.mean(specificity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(specificity_OOS_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['Precision @ 1 Prc'] = str(np.round(
            np.mean(precision_rusboost1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(precision_rusboost1) * 100, 2)) + '%)'


        f1_score_rusboost1 = 2 * (precision_rusboost1 * sensitivity_OOS_rusboost1) / \
                             (precision_rusboost1 + sensitivity_OOS_rusboost1 + 1e-8)

        perf_tbl_general['F1 Score @ 1 Prc'] = str(np.round(
            np.mean(f1_score_rusboost1) * 100, 2)) + '% (' + \
                                               str(np.round(np.std(f1_score_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['NDCG @ 1 Prc'] = str(np.round(
            np.mean(ndcg_rusboost1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ndcg_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['ECM @ 1 Prc'] = str(np.round(
            np.mean(ecm_rusboost1) * 100, 2)) + '% (' + \
                                          str(np.round(np.std(ecm_rusboost1) * 100, 2)) + '%)'


        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_RUS28_temporal.csv'
        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')

    def RUS28_biased(self, C_FN=30, C_FP=1):
        """
        This code uses Bao at al. (2020)'s serial fraud treatment for Table 9 with RUSBoost model.

        Methodological choices:
            - 28 raw accounting items from Bao et al.(2020)
            - 10-fold validation
            - Bao et al. (2020)'s serial fraud treatment
        """

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k

        t0 = datetime.now()

        IS_period=self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-14]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        range_oos = range(start_OOS_year, end_OOS_year + 1)
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= sample_start)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -28:]
        X_CV = X_CV.apply(lambda x: x / np.linalg.norm(x))
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        # optimize RUSBoost number of estimators
        if cv_type == 'kfold':
            if cross_val == True:

                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1 = datetime.now()
                n_estimators = list(range(10, 3001, 10))
                learning_rate = list(x / 1000 for x in range(10, 1001, 10))
                param_grid_rusboost = {'n_estimators': n_estimators,
                                       'learning_rate': learning_rate}

                base_tree = DecisionTreeClassifier(min_samples_leaf=5)
                bao_RUSboost = RUSBoostClassifier(estimator=base_tree, \
                                                  sampling_strategy=1, random_state=0)
                clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_rus.fit(X_CV, Y_CV)
                opt_params_rus = clf_rus.best_params_
                n_opt_rus = opt_params_rus['n_estimators']
                r_opt_rus = opt_params_rus['learning_rate']
                score_rus = clf_rus.best_score_
                t2 = datetime.now()
                dt = t2 - t1
                print('RUSBoost CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('RUSBoost: The optimal number of estimators is ' + str(n_opt_rus) + 'learning rate is' + str(
                    r_opt_rus))


        roc_rusboost = np.zeros(len(range_oos))
        specificity_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost = np.zeros(len(range_oos))
        precision_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1 = np.zeros(len(range_oos))
        specificity_OOS_rusboost1 = np.zeros(len(range_oos))
        precision_rusboost1 = np.zeros(len(range_oos))
        ndcg_rusboost1 = np.zeros(len(range_oos))
        ecm_rusboost1 = np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()

            year_start_IS = sample_start
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            tbl_year_OOS = reduced_tbl.loc[reduced_tbl.fyear == yr]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -28:]
            X = X.apply(lambda x: x / np.linalg.norm(x))
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -28:]
            X_OOS = X_OOS.apply(lambda x: x / np.linalg.norm(x))
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            misstate_test = np.unique(tbl_year_OOS[tbl_year_OOS['AAER_DUMMY'] == 1]['gvkey'])
            Y[np.isin(tbl_year_IS['gvkey'], misstate_test)] = 0

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)
            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=n_opt_rus, \
                                              learning_rate=r_opt_rus, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X, Y)

            probs_oos_fraud_rusboost = clf_rusboost.predict_proba(X_OOS)[:, -1]

            labels_rusboost = clf_rusboost.predict(X_OOS)

            roc_rusboost[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rusboost)
            specificity_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 0, Y_OOS == 0)) / \
                                      np.sum(Y_OOS == 0)

            sensitivity_OOS_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, \
                                                                Y_OOS == 1)) / np.sum(Y_OOS)
            precision_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, Y_OOS == 1)) / np.sum(labels_rusboost)

            cutoff_OOS_rusboost = np.percentile(probs_oos_fraud_rusboost, 99)
            sensitivity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_rusboost >= cutoff_OOS_rusboost)
            ndcg_rusboost1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rusboost, 99)

            FN_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                 Y_OOS == 1))
            FP_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                 Y_OOS == 0))

            ecm_rusboost1[m] = C_FN * P_f * FN_rusboost1 / n_P + C_FP * P_nf * FP_rusboost1 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1


        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['RUSBoost-28']

        perf_tbl_general['Roc'] = str(np.round(
            np.mean(roc_rusboost) * 100, 2)) + '% (' + \
                                  str(np.round(np.std(roc_rusboost) * 100, 2)) + '%)'

        perf_tbl_general['Sensitivity @ 1 Prc'] = str(np.round(
            np.mean(sensitivity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(sensitivity_OOS_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['Specificity @ 1 Prc'] = str(np.round(
            np.mean(specificity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(specificity_OOS_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['Precision @ 1 Prc'] = str(np.round(
            np.mean(precision_rusboost1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(precision_rusboost1) * 100, 2)) + '%)'

        f1_score_rusboost1 = 2 * (precision_rusboost1 * sensitivity_OOS_rusboost1) / \
                             (precision_rusboost1 + sensitivity_OOS_rusboost1 + 1e-8)

        perf_tbl_general['F1 Score @ 1 Prc'] = str(np.round(
            np.mean(f1_score_rusboost1) * 100, 2)) + '% (' + \
                                               str(np.round(np.std(f1_score_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['NDCG @ 1 Prc'] = str(np.round(
            np.mean(ndcg_rusboost1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ndcg_rusboost1) * 100, 2)) + '%)'

        perf_tbl_general['ECM @ 1 Prc'] = str(np.round(
            np.mean(ecm_rusboost1) * 100, 2)) + '% (' + \
                                          str(np.round(np.std(ecm_rusboost1) * 100, 2)) + '%)'

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_RUS28_biased.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


    def FK23_pickle(self, record_matrix=True):

        import pickle

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cross_val = self.cv
        cv_type = self.cv_t
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w
        # First discard five variables that are in Bao et al (2020)
        # but not in Cecchini et al (2010)
        fraud_df.pop('act')
        fraud_df.pop('ap')
        fraud_df.pop('ppegt')
        fraud_df.pop('dltis')
        fraud_df.pop('sstk')

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-14]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        if isfile('features_fk.pkl') == False:
            print('No pickle file found ...')
            print('processing data to extract last years financial figures')
            t1 = datetime.now()
            tbl_ratio_fk = reduced_tbl
            new_cols = tbl_ratio_fk.columns[-23:] + '_last'
            addtbl = pd.DataFrame(columns=new_cols)
            progress_unit = len(tbl_ratio_fk) // 20
            for m in range(0, len(tbl_ratio_fk)):
                if np.mod(m, progress_unit) == 0:
                    print(str(int(m // progress_unit * 100 / 20)) + '% lagged data processing completed')
                sel_gvkey = tbl_ratio_fk.gvkey[m]
                data_gvkey = tbl_ratio_fk[tbl_ratio_fk['gvkey'] == sel_gvkey]
                current_idx = np.where(data_gvkey.index == m)[0][0]
                if current_idx > 0:
                    last_data = data_gvkey.iloc[current_idx - 1, -23:]
                    addtbl.loc[m, :] = last_data.values
                else:
                    last_data = np.ones_like(data_gvkey.iloc[current_idx, -23:]) * float('nan')
                    addtbl.loc[m, :] = last_data

            dt = round((datetime.now() - t1).total_seconds() / 60, 3)
            print('processing data to extract last years financial figures')
            print('elapsed time ' + str(dt) + ' mins')
            tbl_ratio_fk = pd.concat([tbl_ratio_fk, addtbl], axis=1)
            tbl_ratio_fk = tbl_ratio_fk[tbl_ratio_fk.fyear >= (sample_start - 1)]
            init_size_tbl = len(tbl_ratio_fk)
            tbl_ratio_fk = tbl_ratio_fk[tbl_ratio_fk.at_last.isna() == False]
            tbl_ratio_fk = tbl_ratio_fk.reset_index(drop=True)
            drop_for_missing = init_size_tbl / len(tbl_ratio_fk) - 1
            print(str(round(drop_for_missing * 100, 2)) + '% of observations dropped due to ' + \
                  'missing data from last year')

            # Map the raw fundamentals into features >>> 3n(n-1) features for n attributes
            # this means 1518 features for 23 attributes
            t1 = datetime.now()
            red_tbl_fk = tbl_ratio_fk.iloc[:, -46:]
            mapped_X = np.array([])
            progress_unit2 = len(red_tbl_fk) // 100
            for m in range(0, len(red_tbl_fk)):
                if np.mod(m, progress_unit2) == 0:
                    print(str(int(m // progress_unit2 * 100 / 100)) + '% feature processing completed')
                features = np.array([])
                for i in range(0, 23):
                    for j in range(i + 1, 23):
                        ui1 = red_tbl_fk.iloc[m, i]
                        ui1 = (ui1 == 0).astype(int) * 1e-4 + ui1
                        ui2 = red_tbl_fk.iloc[m, 23 + i]
                        ui2 = (ui2 == 0).astype(int) * 1e-4 + ui2
                        uj1 = red_tbl_fk.iloc[m, j]
                        uj1 = (uj1 == 0).astype(int) * 1e-4 + uj1
                        uj2 = red_tbl_fk.iloc[m, 23 + j]
                        uj2 = (uj2 == 0).astype(int) * 1e-4 + uj2
                        features_new = np.array([ui1 / uj1, uj1 / ui1, uj2 / ui2, ui2 / uj2, (ui1 * uj2) / (uj1 * ui2), \
                                                 (uj1 * ui2) / (ui1 * uj2)])
                        features = np.append(features, features_new)
                if mapped_X.shape[-1] == 0:
                    mapped_X = np.append(mapped_X, features.T)
                else:
                    mapped_X = np.vstack((mapped_X, features.T))

            dt = round((datetime.now() - t1).total_seconds() / 60, 3)
            print('feature processing completed ...')
            print('elapsed time ' + str(dt) + ' mins')
            if record_matrix == True:
                DB_Dict = {'matrix': mapped_X, 'lagged_Data': tbl_ratio_fk}
                fl_name = 'features_fk.pkl'
                pickle.dump(DB_Dict, open(fl_name, 'w+b'))

    def FK_23(self, C_FN=30, C_FP=1):

        """
        This code generates the results for SVM-FK23 in Table 5.

        Methodological choices:
                    - 23 raw accounting items from Cecchini et al. (2010)
                    - 10-fold validation
                    - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cross_val = self.cv
        case_window = self.sa
        fraud_df = self.df
        write = self.w

        dict_db = pickle.load(open('features_fk.pkl', 'r+b'))
        tbl_ratio_fk = dict_db['lagged_Data']
        mapped_X = dict_db['matrix']
        red_tbl_fk = tbl_ratio_fk.iloc[:, -46:]
        print('pickle file loaded successfully ...')

        idx_CV = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear >= sample_start,
                                             tbl_ratio_fk.fyear < start_OOS_year)].index
        Y_CV = tbl_ratio_fk.AAER_DUMMY[idx_CV]

        X_CV = mapped_X[idx_CV, :]
        idx_real = np.where(np.logical_and(np.isnan(X_CV).any(axis=1) == False, \
                                           np.isinf(X_CV).any(axis=1) == False))[0]
        X_CV = X_CV[idx_real, :]

        Y_CV = Y_CV.iloc[idx_real]
        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        print('Grid search hyperparameter optimisation started for SVM-FK')
        t1=datetime.now()

        pipe_svm = Pipeline([('scale', StandardScaler()),\
                    ('base_mdl_svm',SVC(shrinking=False,\
                            probability=False,random_state=0,cache_size=1000,\
                                tol=X_CV.shape[-1]*1e-3))])

        param_grid_svm = {'base_mdl_svm__kernel': ['rbf', 'linear', 'poly'], \
                          'base_mdl_svm__C': [0.001, 0.01, 0.1, 1, 10, 100], \
                          'base_mdl_svm__gamma': [0.0001, 0.001, 0.01, 0.1], \
                          'base_mdl_svm__class_weight': [{0: 1/x, 1: 1} for x in range(10,501,10)]}

        clf_svm_fk = GridSearchCV(pipe_svm, param_grid_svm,scoring='roc_auc',\
                            n_jobs=None,cv=k_fold,refit=False)
        clf_svm_fk.fit(X_CV, Y_CV)
        opt_params_svm_fk=clf_svm_fk.best_params_
        gamma_opt = opt_params_svm_fk['base_mdl_svm__gamma']
        cw_opt = opt_params_svm_fk['base_mdl_svm__class_weight']
        c_opt = opt_params_svm_fk['base_mdl_svm__C']
        kernel_opt = opt_params_svm_fk['base_mdl_svm__kernel']
        score_svm=clf_svm_fk.best_score_

        t2=datetime.now()
        dt=t2-t1
        print('SVM CV finished after '+str(dt.total_seconds())+' sec')
        print('SVM: The optimal C+/C- ratio is ' + str(cw_opt) + ', gamma is' + str(
            gamma_opt) + ', C is' + str(c_opt) + ',score is' + str(score_svm) + \
              ', kernel is' + str(kernel_opt))


        t000 = datetime.now()
        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)  # (2001,2010+1,1)
        roc_ratio = np.zeros(len(range_oos))
        ndcg_ratio = np.zeros(len(range_oos))
        sensitivity_ratio = np.zeros(len(range_oos))
        specificity_ratio = np.zeros(len(range_oos))
        precision_ratio = np.zeros(len(range_oos))
        ecm_ratio = np.zeros(len(range_oos))
        f1_ratio = np.zeros(len(range_oos))


        m = 0

        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            idx_IS = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear < yr - OOS_gap, \
                                                 tbl_ratio_fk.fyear >= year_start_IS)].index
            tbl_year_IS = tbl_ratio_fk.loc[idx_IS, :]
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = tbl_ratio_fk.loc[tbl_ratio_fk.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]

            X = mapped_X[idx_IS, :]
            idx_real = np.where(np.logical_and(np.isnan(X).any(axis=1) == False, \
                                               np.isinf(X).any(axis=1) == False))[0]
            X = X[idx_real, :]
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y = tbl_ratio_fk.AAER_DUMMY[idx_IS]
            Y = Y.iloc[idx_real]

            X_OOS = mapped_X[tbl_year_OOS.index, :]
            idx_real_OOS = np.where(np.logical_and(np.isnan(X_OOS).any(axis=1) == False, \
                                                   np.isinf(X_OOS).any(axis=1) == False))[0]
            X_OOS = X_OOS[idx_real_OOS, :]
            X_OOS = (X_OOS - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y_OOS = tbl_year_OOS.AAER_DUMMY
            Y_OOS = Y_OOS.iloc[idx_real_OOS]
            Y_OOS = Y_OOS.reset_index(drop=True)

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            t01 = datetime.now()

            svm_fk = SVC(class_weight=cw_opt, kernel=kernel_opt,\
                         C=c_opt, gamma=gamma_opt,\
                         shrinking=False, \
                         probability=False, random_state=0, cache_size=1000, \
                         tol=X.shape[-1] * 1e-3)
            clf_svm_fk = svm_fk.fit(X, Y)
            predicted_test = clf_svm_fk.decision_function(X_OOS)
            predicted_test[predicted_test >= 1] = 1 + np.log(predicted_test[predicted_test >= 1])
            predicted_test = np.exp(predicted_test) / (1 + np.exp(predicted_test))
            roc_ratio[m] = roc_auc_score(Y_OOS, predicted_test)

            cutoff_ratio = np.percentile(predicted_test, 99)
            labels_ratio = (predicted_test >= cutoff_ratio).astype(int)
            sensitivity_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_ratio[m] = np.sum(np.logical_and(labels_ratio == 0, Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(labels_ratio)
            ndcg_ratio[m] = ndcg_k(Y_OOS.to_numpy(), predicted_test, 99)

            FN2 = np.sum(np.logical_and(predicted_test < cutoff_ratio, \
                                       Y_OOS == 1))
            FP2 = np.sum(np.logical_and(predicted_test >= cutoff_ratio, \
                                       Y_OOS == 0))
            ecm_ratio[m] = C_FN * P_f * FN2 / n_P + C_FP * P_nf * FP2 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        f1_ratio = 2 * (precision_ratio * sensitivity_ratio) / (precision_ratio + sensitivity_ratio + 1e-8)

        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['FK23']

        perf_tbl_general['Roc'] = [str(np.round(
            np.mean(roc_ratio) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_ratio) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_ratio) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_ratio) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ratio) * 100, 2)) + '%)']

        perf_tbl_general['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_ratio) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_ratio) * 100, 2)) + '%)']

        perf_tbl_general['NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_ratio) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ratio) * 100, 2)) + '%)']

        perf_tbl_general['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_ratio) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ratio) * 100, 2)) + '%)']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_FK23.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)

        t001 = datetime.now()
        dt00 = t001 - t000
        print('MC analysis is completed after ' + str(dt00.total_seconds()) + ' seconds')

    def FK23_temporal(self, C_FN=30, C_FP=1):
        """
        This code generates the results for SVM-FK23 in Table 8.

        Methodological choices:
                            - 23 raw accounting items from Cecchini et al. (2010)
                            - temporal validation
                            - Bertomeu et al. (2021)'s serial fraud treatment
        """
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        import pickle

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        temp_year = self.cv_t_y
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cross_val = self.cv
        case_window = self.sa  # expanding or rolling window
        fraud_df = self.df
        write = self.w

        dict_db = pickle.load(open('features_fk.pkl', 'r+b'))
        tbl_ratio_fk = dict_db['lagged_Data']
        mapped_X = dict_db['matrix']
        red_tbl_fk = tbl_ratio_fk.iloc[:, -46:]
        print('pickle file loaded successfully ...')

        idx_CV = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear >= sample_start,
                                             tbl_ratio_fk.fyear < start_OOS_year)].index
        Y_CV = tbl_ratio_fk.AAER_DUMMY[idx_CV]

        X_CV = mapped_X[idx_CV, :]
        idx_real = np.where(np.logical_and(np.isnan(X_CV).any(axis=1) == False, \
                                           np.isinf(X_CV).any(axis=1) == False))[0]
        X_CV = X_CV[idx_real, :]

        Y_CV = Y_CV.iloc[idx_real]
        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        print('Grid search hyperparameter optimisation started for SVM-FK')
        t1=datetime.now()
        cutoff_temporal=2001-temp_year
        idx_CV_train=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear>=1991,
                                            tbl_ratio_fk.fyear<cutoff_temporal)].index
        X_CV_train=mapped_X[idx_CV_train]
        Y_CV_train=tbl_ratio_fk.AAER_DUMMY[idx_CV_train]
        idx_CV_test=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear>=cutoff_temporal,
                                            tbl_ratio_fk.fyear<2001)].index
        mean_CV=np.mean(X_CV_train)
        std_CV=np.std(X_CV_train)
        X_CV_train=(X_CV_train-mean_CV)/std_CV

        X_CV_test=mapped_X[idx_CV_test]
        Y_CV_test=tbl_ratio_fk.AAER_DUMMY[idx_CV_test]

        X_CV_test=(X_CV_test-mean_CV)/std_CV

        C = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma = [0.0001, 0.001, 0.01, 0.1]
        kernel = ['rbf', 'linear', 'poly']
        class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]

        param_grid_svm={'kernel': kernel,'class_weight': class_weight,'C': C, 'gamma': gamma}

        temp_svm={'kernel':[],'class_weight':[],'C':[],'gamma':[],'score':[]}

        for k in param_grid_svm['kernel']:
            for w in param_grid_svm['class_weight']:
                for c in param_grid_svm['C']:
                    for g in param_grid_svm['gamma']:
                        temp_svm['kernel'].append(k)
                        temp_svm['class_weight'].append(w)
                        temp_svm['C'].append(c)
                        temp_svm['gamma'].append(g)

                        base_mdl_svm=SVC(shrinking=False,\
                                        probability=False,\
                                        kernel=k, C=c, gamma=g,\
                                        class_weight=w,\
                                        random_state=0,max_iter=-1,\
                                        tol=X_CV.shape[-1]*1e-3).fit(X_CV_train,Y_CV_train)
                        predicted_test_svc=base_mdl_svm.decision_function(X_CV_test)

                        predicted_test_svc=np.exp(predicted_test_svc)/(1+np.exp(predicted_test_svc))

                        temp_svm['score'].append(roc_auc_score(Y_CV_test,predicted_test_svc))

        idx_opt_svm=temp_svm['score'].index(np.max(temp_svm['score']))

        kernel_opt=temp_svm['kernel'][idx_opt_svm]
        cw_opt=temp_svm['class_weight'][idx_opt_svm]
        C_opt=temp_svm['C'][idx_opt_svm]
        gamma_opt= temp_svm['gamma'][idx_opt_svm]
        score_svm=temp_svm['score'][idx_opt_svm]
        opt_params_svm_fk={'class_weight':cw_opt,'kernel':kernel_opt,\
                           'C':C_opt, 'gamma':gamma_opt}
        print(opt_params_svm_fk)
        t2=datetime.now()
        dt=t2-t1
        print('SVM Temporal validation finished after '+str(dt.total_seconds())+' sec')
        print('SVM: The optimal class weight is ' +str(cw_opt) + ' ,kernel is ' + str(kernel_opt) + \
              ' ,C is ' + str(C_opt) + ' ,and gamma is ' +str(gamma_opt))

        t000 = datetime.now()

        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)  # (2001,2010+1,1)
        roc_ratio = np.zeros(len(range_oos))
        ndcg_ratio = np.zeros(len(range_oos))
        sensitivity_ratio = np.zeros(len(range_oos))
        specificity_ratio = np.zeros(len(range_oos))
        precision_ratio = np.zeros(len(range_oos))
        ecm_ratio = np.zeros(len(range_oos))
        f1_ratio = np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            idx_IS = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear < yr - OOS_gap, \
                                                 tbl_ratio_fk.fyear >= year_start_IS)].index
            tbl_year_IS = tbl_ratio_fk.loc[idx_IS, :]
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = tbl_ratio_fk.loc[tbl_ratio_fk.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]

            X = mapped_X[idx_IS, :]
            idx_real = np.where(np.logical_and(np.isnan(X).any(axis=1) == False, \
                                               np.isinf(X).any(axis=1) == False))[0]
            X = X[idx_real, :]
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y = tbl_ratio_fk.AAER_DUMMY[idx_IS]
            Y = Y.iloc[idx_real]

            X_OOS = mapped_X[tbl_year_OOS.index, :]
            idx_real_OOS = np.where(np.logical_and(np.isnan(X_OOS).any(axis=1) == False, \
                                                   np.isinf(X_OOS).any(axis=1) == False))[0]
            X_OOS = X_OOS[idx_real_OOS, :]
            X_OOS = (X_OOS - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y_OOS = tbl_year_OOS.AAER_DUMMY
            Y_OOS = Y_OOS.iloc[idx_real_OOS]
            Y_OOS = Y_OOS.reset_index(drop=True)

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            t01 = datetime.now()

            svm_fk = SVC(class_weight=opt_params_svm_fk['class_weight'], kernel=opt_params_svm_fk['kernel'], \
                         C=opt_params_svm_fk['C'], gamma=opt_params_svm_fk['gamma'],\
                         shrinking=False, \
                         probability=False, random_state=0, cache_size=1000, \
                         tol=X.shape[-1] * 1e-3)
            clf_svm_fk = svm_fk.fit(X, Y)
            predicted_test = clf_svm_fk.decision_function(X_OOS)
            predicted_test[predicted_test >= 1] = 1 + np.log(predicted_test[predicted_test >= 1])
            predicted_test = np.exp(predicted_test) / (1 + np.exp(predicted_test))
            roc_ratio[m] = roc_auc_score(Y_OOS, predicted_test)

            cutoff_ratio = np.percentile(predicted_test, 99)
            labels_ratio = (predicted_test >= cutoff_ratio).astype(int)
            sensitivity_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_ratio[m] = np.sum(np.logical_and(labels_ratio == 0, Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(labels_ratio)
            ndcg_ratio[m] = ndcg_k(Y_OOS.to_numpy(), predicted_test, 99)
            FN = np.sum(np.logical_and(predicted_test < cutoff_ratio, \
                                       Y_OOS == 1))
            FP = np.sum(np.logical_and(predicted_test >= cutoff_ratio, \
                                       Y_OOS == 0))
            ecm_ratio[m] = C_FN * P_f * FN / n_P + C_FP * P_nf * FP / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        f1_ratio = 2 * (precision_ratio * sensitivity_ratio) / (precision_ratio + sensitivity_ratio + 1e-8)

        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['FK23_temporal']
        perf_tbl_general['Roc'] = [str(np.round(
            np.mean(roc_ratio) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_ratio) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_ratio) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_ratio) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ratio) * 100, 2)) + '%)']

        perf_tbl_general['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_ratio) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_ratio) * 100, 2)) + '%)']

        perf_tbl_general['NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_ratio) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ratio) * 100, 2)) + '%)']

        perf_tbl_general['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_ratio) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ratio) * 100, 2)) + '%)']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_FK23_temporal.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)

        t001 = datetime.now()
        dt00 = t001 - t000
        print('MC analysis is completed after ' + str(dt00.total_seconds()) + ' seconds')


    def FK23_biased(self, C_FN=30, C_FP=1):

        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cross_val = self.cv
        case_window = self.sa
        fraud_df = self.df
        write = self.w

        dict_db = pickle.load(open('features_fk.pkl', 'r+b'))
        tbl_ratio_fk = dict_db['lagged_Data']
        mapped_X = dict_db['matrix']
        red_tbl_fk = tbl_ratio_fk.iloc[:, -46:]
        print('pickle file loaded successfully ...')

        idx_CV = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear >= sample_start,
                                             tbl_ratio_fk.fyear < start_OOS_year)].index
        Y_CV = tbl_ratio_fk.AAER_DUMMY[idx_CV]

        X_CV = mapped_X[idx_CV, :]

        idx_real = np.where(np.logical_and(np.isnan(X_CV).any(axis=1) == False, \
                                           np.isinf(X_CV).any(axis=1) == False))[0]
        X_CV = X_CV[idx_real, :]

        Y_CV = Y_CV.iloc[idx_real]
        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        print('Grid search hyperparameter optimisation started for SVM-FK')
        t1=datetime.now()

        pipe_svm = Pipeline([('scale', StandardScaler()),\
                            ('base_mdl_svm',SVC(shrinking=False,\
                                probability=False,random_state=0,cache_size=1000,\
                            tol=X_CV.shape[-1]*1e-3))])

        C = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma = [0.0001, 0.001, 0.01, 0.1]
        kernel = ['rbf', 'linear', 'poly']
        class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]

        param_grid_svm = {'base_mdl_svm__kernel': kernel, \
                          'base_mdl_svm__C': C, \
                          'base_mdl_svm__class_weight': class_weight,
                          'base_mdl_svm__gamma':gamma}

        clf_svm_fk = GridSearchCV(pipe_svm, param_grid_svm,scoring='roc_auc',\
        n_jobs=None,cv=k_fold,refit=False)
        clf_svm_fk.fit(X_CV, Y_CV)
        opt_params_svm_fk=clf_svm_fk.best_params_
        cw_opt = opt_params_svm_fk['base_mdl_svm__class_weight']
        c_opt = opt_params_svm_fk['base_mdl_svm__C']
        kernel_opt = opt_params_svm_fk['base_mdl_svm__kernel']
        gamma_opt = opt_params_svm_fk['base_mdl_svm__gamma']
        score_svm=clf_svm_fk.best_score_

        t2=datetime.now()
        dt=t2-t1
        print('SVM CV finished after '+str(dt.total_seconds())+' sec')
        print('SVM: The optimal class weight is ' + str(cw_opt) + \
              ', C is ' + str(c_opt) + ',score is' + str(score_svm) + \
              ', kernel is ' + str(kernel_opt) + ' ,gamma is ' + str(gamma_opt))

        t000 = datetime.now()
        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)
        roc_ratio = np.zeros(len(range_oos))
        ndcg_ratio = np.zeros(len(range_oos))
        sensitivity_ratio = np.zeros(len(range_oos))
        specificity_ratio = np.zeros(len(range_oos))
        precision_ratio = np.zeros(len(range_oos))
        ecm_ratio = np.zeros(len(range_oos))
        f1_ratio = np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            idx_IS = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear < yr - OOS_gap, \
                                                 tbl_ratio_fk.fyear >= year_start_IS)].index
            tbl_year_IS = tbl_ratio_fk.loc[idx_IS, :]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            tbl_year_OOS = tbl_ratio_fk.loc[tbl_ratio_fk.fyear == yr]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = mapped_X[idx_IS, :]
            idx_real = np.where(np.logical_and(np.isnan(X).any(axis=1) == False, \
                                               np.isinf(X).any(axis=1) == False))[0]
            X = X[idx_real, :]
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y = tbl_ratio_fk.AAER_DUMMY[idx_IS]
            Y = Y.iloc[idx_real]

            X_OOS = mapped_X[tbl_year_OOS.index, :]
            idx_real_OOS = np.where(np.logical_and(np.isnan(X_OOS).any(axis=1) == False, \
                                                   np.isinf(X_OOS).any(axis=1) == False))[0]
            X_OOS = X_OOS[idx_real_OOS, :]
            X_OOS = (X_OOS - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y_OOS = tbl_year_OOS.AAER_DUMMY
            Y_OOS = Y_OOS.iloc[idx_real_OOS]
            Y_OOS = Y_OOS.reset_index(drop=True)

            misstate_test = np.unique(tbl_year_OOS[tbl_year_OOS['AAER_DUMMY'] == 1]['gvkey'])
            Y[np.isin(tbl_year_IS['gvkey'], misstate_test)] = 0

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            t01 = datetime.now()

            svm_fk = SVC(class_weight=cw_opt, kernel=kernel_opt, C=c_opt, gamma=gamma_opt, shrinking=False, \
                         probability=False, random_state=0, cache_size=1000, \
                         tol=X.shape[-1] * 1e-3)
            clf_svm_fk = svm_fk.fit(X, Y)
            predicted_test = clf_svm_fk.decision_function(X_OOS)
            predicted_test[predicted_test >= 1] = 1 + np.log(predicted_test[predicted_test >= 1])
            predicted_test = np.exp(predicted_test) / (1 + np.exp(predicted_test))
            roc_ratio[m] = roc_auc_score(Y_OOS, predicted_test)

            cutoff_ratio = np.percentile(predicted_test, 99)
            labels_ratio = (predicted_test >= cutoff_ratio).astype(int)
            sensitivity_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_ratio[m] = np.sum(np.logical_and(labels_ratio == 0, Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(labels_ratio)
            ndcg_ratio[m] = ndcg_k(Y_OOS.to_numpy(), predicted_test, 99)

            FN = np.sum(np.logical_and(predicted_test < cutoff_ratio, \
                                       Y_OOS == 1))
            FP = np.sum(np.logical_and(predicted_test >= cutoff_ratio, \
                                       Y_OOS == 0))

            ecm_ratio[m] = C_FN * P_f * FN / n_P + C_FP * P_nf * FP / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        f1_ratio = 2 * (precision_ratio * sensitivity_ratio) / (precision_ratio + sensitivity_ratio + 1e-8)

        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['FK23_kfold']
        perf_tbl_general['Roc'] = [str(np.round(
            np.mean(roc_ratio) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_ratio) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_ratio) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_ratio) * 100, 2)) + '%)']

        perf_tbl_general['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_ratio) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ratio) * 100, 2)) + '%)']

        perf_tbl_general['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_ratio) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_ratio) * 100, 2)) + '%)']

        perf_tbl_general['NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_ratio) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ratio) * 100, 2)) + '%)']

        perf_tbl_general['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_ratio) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ratio) * 100, 2)) + '%)']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_FK23_biased.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)

        t001 = datetime.now()
        dt00 = t001 - t000
        print('MC analysis is completed after ' + str(dt00.total_seconds()) + ' seconds')

    def analyse_forward(self, C_FN=30, C_FP=1):
        """
        This code produces the results for Table 10.
        For more details on the metrics we use, please see Section 7.1 of the paper.

        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from imblearn.ensemble import RUSBoostClassifier
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        import pickle

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cross_val = self.cv
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        fyears_available = np.unique(fraud_df.fyear)

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]

        tbl_fk_svm = reduced_tbl.copy()

        tbl_fk_svm.pop('act')
        tbl_fk_svm.pop('ap')
        tbl_fk_svm.pop('ppegt')
        tbl_fk_svm.pop('dltis')
        tbl_fk_svm.pop('sstk')

        # RUSBOost-28
        n_opt_rus = 400
        r_opt_rus = 0.02

        #SVM
        opt_params_svm = {'class_weight': {0: 0.01, 1: 1}, 'kernel': 'rbf', 'gamma': 0.001, 'C': 10}
        cw_opt = opt_params_svm['class_weight']
        kernel_opt = opt_params_svm['kernel']
        c_opt = opt_params_svm['C']
        gamma_opt = opt_params_svm['gamma']
        score_svm = 0.6979458124832295

        #Logit
        score_lr = 0.6944198770970713

        #LogitBoost
        opt_params_ada = {'learning_rate': .1, 'n_estimators': 80}
        score_ada = 0.6949275223759799

        #SVM-FK23
        opt_params_svm_fk = {'class_weight': {0: 0.02, 1: 1}, 'kernel': 'linear', 'C': 1}

        if isfile('features_fk.pkl') == True:
            dict_db = pickle.load(open('features_fk.pkl', 'r+b'))
            tbl_ratio_fk = dict_db['lagged_Data']
            mapped_X = dict_db['matrix']
            red_tbl_fk = tbl_ratio_fk.iloc[:, -46:]
            print('pickle file for SVM-FK loaded successfully ...')
        else:
            raise NameError('The pickle file for the financial kernel missing. Rerun the SVM FK procedure first...')

        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)

        sensitivity_OOS_svm1 = np.zeros(len(range_oos))
        specificity_OOS_svm1 = np.zeros(len(range_oos))
        precision_svm1 = np.zeros(len(range_oos))
        ecm_svm1 = np.zeros(len(range_oos))

        sensitivity_OOS_lr1 = np.zeros(len(range_oos))
        specificity_OOS_lr1 = np.zeros(len(range_oos))
        precision_lr1 = np.zeros(len(range_oos))
        ecm_lr1 = np.zeros(len(range_oos))

        sensitivity_OOS_ada1 = np.zeros(len(range_oos))
        specificity_OOS_ada1 = np.zeros(len(range_oos))
        precision_ada1 = np.zeros(len(range_oos))
        ecm_ada1 = np.zeros(len(range_oos))

        sensitivity_OOS_fused1 = np.zeros(len(range_oos))
        specificity_OOS_fused1 = np.zeros(len(range_oos))
        precision_fused1 = np.zeros(len(range_oos))
        ecm_fused1 = np.zeros(len(range_oos))

        precision_svm_fk1 = np.zeros(len(range_oos))
        sensitivity_svm_fk1 = np.zeros(len(range_oos))
        specificity_svm_fk1 = np.zeros(len(range_oos))
        ecm_svm_fk1 = np.zeros(len(range_oos))

        precision_rus = np.zeros(len(range_oos))
        sensitivity_rus = np.zeros(len(range_oos))
        specificity_rus = np.zeros(len(range_oos))
        ecm_rus = np.zeros(len(range_oos))

        C_FN = 30
        C_FP = 1
        m = 0


        for yr in range_oos:
            t1 = datetime.now()
            year_start_IS = sample_start

            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]
            X = tbl_year_IS.iloc[:, -11:]

            mean_vals = np.mean(X)
            std_vals = np.std(X)
            X = (X - mean_vals) / std_vals

            Y = tbl_year_IS.AAER_DUMMY

            # Setting the IS for RUS
            tbl_year_IS_rus = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                             reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS_rus = tbl_year_IS_rus.reset_index(drop=True)
            misstate_firms_rus = np.unique(tbl_year_IS_rus.gvkey[tbl_year_IS_rus.AAER_DUMMY == 1])
            tbl_year_OOS_rus = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                              reduced_tbl.fyear < yr + OOS_period)]
            X_rus = tbl_year_IS_rus.iloc[:, -39:-11]
            X_rus = X_rus.apply(lambda x: x / np.linalg.norm(x))

            Y_rus = tbl_year_IS_rus.AAER_DUMMY

            # Setting the IS for SVM-FK
            idx_IS_FK = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear < yr, \
                                                    tbl_ratio_fk.fyear >= year_start_IS)].index

            X_FK = mapped_X[idx_IS_FK, :]
            idx_real = np.where(np.logical_and(np.isnan(X_FK).any(axis=1) == False, \
                                               np.isinf(X_FK).any(axis=1) == False))[0]
            X_FK = X_FK[idx_real, :]
            X_FK = (X_FK - np.mean(X_FK, axis=0)) / np.std(X_FK, axis=0)
            Y_FK = tbl_ratio_fk.AAER_DUMMY[idx_IS_FK]
            Y_FK = Y_FK.iloc[idx_real]

            ok_index = np.zeros(tbl_year_OOS.shape[0])
            for s in range(0, tbl_year_OOS.shape[0]):
                if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                    ok_index[s] = True

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS = (X_OOS - mean_vals) / std_vals

            Y_OOS = tbl_year_OOS.AAER_DUMMY

            # Setting the OOS for RUS
            ok_index_rus = np.zeros(tbl_year_OOS_rus.shape[0])
            for s in range(0, tbl_year_OOS_rus.shape[0]):
                if not tbl_year_OOS_rus.iloc[s, 1] in misstate_firms_rus:
                    ok_index_rus[s] = True

            tbl_year_OOS_rus = tbl_year_OOS_rus.iloc[ok_index_rus == True, :]
            tbl_year_OOS_rus = tbl_year_OOS_rus.reset_index(drop=True)

            X_rus_OOS = tbl_year_OOS_rus.iloc[:, -39:-11]
            X_rus_OOS = X_rus_OOS.apply(lambda x: x / np.linalg.norm(x))

            Y_OOS_rus = tbl_year_OOS_rus.AAER_DUMMY

            # Setting the OOS for SVM-FK
            tbl_fk_OOS = tbl_ratio_fk.loc[np.logical_and(tbl_ratio_fk.fyear >= yr, \
                                                         tbl_ratio_fk.fyear < yr + OOS_period)]

            ok_index_fk = np.zeros(tbl_fk_OOS.shape[0])
            for s in range(0, tbl_fk_OOS.shape[0]):
                if not tbl_fk_OOS.iloc[s, 1] in misstate_firms:
                    ok_index_fk[s] = True

            X_OOS_FK = mapped_X[tbl_fk_OOS.index, :]
            idx_real_OOS_FK = np.where(np.logical_and(np.isnan(X_OOS_FK).any(axis=1) == False, \
                                                      np.isinf(X_OOS_FK).any(axis=1) == False))[0]
            X_OOS_FK = X_OOS_FK[idx_real_OOS_FK, :]
            X_OOS_FK = (X_OOS_FK - np.mean(X_FK, axis=0)) / np.std(X_FK, axis=0)
            Y_OOS_FK = tbl_year_OOS.AAER_DUMMY
            Y_OOS_FK = Y_OOS_FK.iloc[idx_real_OOS_FK]
            Y_OOS_FK = Y_OOS_FK.reset_index(drop=True)
            tbl_fk_OOS = tbl_fk_OOS.reset_index(drop=True)


            # 2002-2018
            tbl_forward = reduced_tbl.loc[reduced_tbl.fyear >= yr + OOS_period]
            # adjust serial frauds
            ok_index_forward = np.zeros(tbl_forward.shape[0])
            for s in range(0, tbl_forward.shape[0]):
                if not tbl_forward.iloc[s, 1] in misstate_firms:
                    ok_index_forward[s] = True

            tbl_forward = tbl_forward.iloc[ok_index_forward == True, :]
            tbl_forward = tbl_forward.reset_index(drop=True)
            # identify companies that are fraudulent in forward year e.g.2002-2018
            forward_misstatement = tbl_forward.loc[tbl_forward.AAER_DUMMY == 1]
            forward_misstatement = forward_misstatement.reset_index(drop=True)

            forward_misstate_firms = np.unique(forward_misstatement['gvkey'])

            # SVM-FK23
            clf_svm_fk = SVC(class_weight=opt_params_svm_fk['class_weight'], kernel=opt_params_svm_fk['kernel'], \
                             C=opt_params_svm_fk['C'], shrinking=False, \
                             probability=False, random_state=0, cache_size=1000, \
                             tol=X_FK.shape[-1] * 1e-3)
            clf_svm_fk = clf_svm_fk.fit(X_FK, Y_FK)
            pred_test_svm_fk = clf_svm_fk.decision_function(X_OOS_FK)
            pred_test_svm_fk[pred_test_svm_fk >= 1] = 1 + np.log(pred_test_svm_fk[pred_test_svm_fk >= 1])
            probs_oos_fraud_svm_fk = np.exp(pred_test_svm_fk) / (1 + np.exp(pred_test_svm_fk))

            cutoff_OOS_svm_fk = np.percentile(probs_oos_fraud_svm_fk, 99)
            # identify companies labelled as fraud (pass the 99% threshold) in OOS period
            idx_top_1fk = np.where(probs_oos_fraud_svm_fk >= cutoff_OOS_svm_fk)[0]
            firms_top1fk = tbl_fk_OOS['gvkey'][idx_top_1fk].values
            idx_nonfraud_fk = np.where(probs_oos_fraud_svm_fk < cutoff_OOS_svm_fk)[0]
            firms_nonfraud_fk = tbl_fk_OOS['gvkey'][idx_nonfraud_fk].values

            TP_svm_fk1 = 0
            for frm in firms_top1fk:
                if frm in forward_misstate_firms:
                    TP_svm_fk1 += 1

            TN_svm_fk1 = 0
            for frm in firms_nonfraud_fk:
                if frm not in forward_misstate_firms:
                    TN_svm_fk1 += 1

            FP_svm_fk1 = 0
            for frm in firms_top1fk:
                if frm not in forward_misstate_firms:
                    FP_svm_fk1 += 1

            FN_svm_fk1 = 0
            for frm in firms_nonfraud_fk:
                if frm in forward_misstate_firms:
                    FN_svm_fk1 += 1

            precision_svm_fk1[m] = TP_svm_fk1 / (TP_svm_fk1 + FP_svm_fk1)
            sensitivity_svm_fk1[m] = TP_svm_fk1 / (TP_svm_fk1 + FN_svm_fk1)
            specificity_svm_fk1[m] = TN_svm_fk1 / (TN_svm_fk1 + FP_svm_fk1)
            ecm_svm_fk1[m] = (FP_svm_fk1 * C_FP + FN_svm_fk1 * C_FN) / (
                        TP_svm_fk1 + TN_svm_fk1 + FP_svm_fk1 + FN_svm_fk1)

            # RUSBoost-28
            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=n_opt_rus, \
                                              learning_rate=r_opt_rus, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X_rus, Y_rus)
            probs_oos_fraud_rus = clf_rusboost.predict_proba(X_rus_OOS)[:, -1]

            cutoff_OOS_rus = np.percentile(probs_oos_fraud_rus, 99)

            idx_top1rus = np.where(probs_oos_fraud_rus >= cutoff_OOS_rus)[0]
            firms_top1rus = tbl_year_OOS_rus['gvkey'][idx_top1rus].values
            idx_nonfraud_rus = np.where(probs_oos_fraud_rus < cutoff_OOS_rus)[0]
            firms_nonfraud_rus = tbl_year_OOS_rus['gvkey'][idx_nonfraud_rus].values

            TP_rus = 0
            for frm in firms_top1rus:
                if frm in forward_misstate_firms:
                    TP_rus += 1

            TN_rus = 0
            for frm in firms_nonfraud_rus:
                if frm not in forward_misstate_firms:
                    TN_rus += 1

            FP_rus = 0
            for frm in firms_top1rus:
                if frm not in forward_misstate_firms:
                    FP_rus += 1

            FN_rus = 0
            for frm in firms_nonfraud_rus:
                if frm in forward_misstate_firms:
                    FN_rus += 1

            precision_rus[m] = TP_rus / (TP_rus + FP_rus)
            sensitivity_rus[m] = TP_rus / (TP_rus + FN_rus)
            specificity_rus[m] = TN_rus / (TN_rus + FP_rus)
            ecm_rus[m] = (FP_rus * C_FP + FN_rus * C_FN) / (TP_rus + TN_rus + FP_rus + FN_rus)

            # SVM
            clf_svm = SVC(class_weight=cw_opt, kernel=kernel_opt, C=c_opt, gamma=gamma_opt, shrinking=False, \
                          probability=False, random_state=0, max_iter=-1, \
                          tol=X.shape[-1] * 1e-3)

            clf_svm = clf_svm.fit(X, Y)

            pred_test_svm = clf_svm.decision_function(X_OOS)
            probs_oos_fraud_svm = np.exp(pred_test_svm) / (1 + np.exp(pred_test_svm))

            cutoff_OOS_svm = np.percentile(probs_oos_fraud_svm, 99)
            idx_top1svm = np.where(probs_oos_fraud_svm >= cutoff_OOS_svm)[0]
            firms_top1svm = tbl_year_OOS['gvkey'][idx_top1svm].values
            idx_nonfraud_svm = np.where(probs_oos_fraud_svm < cutoff_OOS_svm)[0]
            firms_nonfraud_svm = tbl_year_OOS['gvkey'][idx_nonfraud_svm].values

            TP_svm = 0
            for frm in firms_top1svm:
                if frm in forward_misstate_firms:
                    TP_svm += 1

            TN_svm = 0
            for frm in firms_nonfraud_svm:
                if frm not in forward_misstate_firms:
                    TN_svm += 1

            FP_svm = 0
            for frm in firms_top1svm:
                if frm not in forward_misstate_firms:
                    FP_svm += 1

            FN_svm = 0
            for frm in firms_nonfraud_svm:
                if frm in forward_misstate_firms:
                    FN_svm += 1

            precision_svm1[m] = TP_svm / (TP_svm + FP_svm)
            sensitivity_OOS_svm1[m] = TP_svm / (TP_svm + FN_svm)
            specificity_OOS_svm1[m] = TN_svm / (TN_svm + FP_svm)
            ecm_svm1[m] = (FP_svm * C_FP + FN_svm * C_FN) / (TP_svm + TN_svm + FP_svm + FN_svm)

            # Logit
            clf_lr = Logit(Y, X)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS)

            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)

            idx_top1lr = np.where(probs_oos_fraud_lr >= cutoff_OOS_lr)[0]
            firms_top1lr = tbl_year_OOS['gvkey'][idx_top1lr].values
            idx_nonfraud_lr = np.where(probs_oos_fraud_lr < cutoff_OOS_lr)[0]
            firms_nonfraud_lr = tbl_year_OOS['gvkey'][idx_nonfraud_lr].values

            TP_lr = 0
            for frm in firms_top1lr:
                if frm in forward_misstate_firms:
                    TP_lr += 1

            TN_lr = 0
            for frm in firms_nonfraud_lr:
                if frm not in forward_misstate_firms:
                    TN_lr += 1

            FP_lr = 0
            for frm in firms_top1lr:
                if frm not in forward_misstate_firms:
                    FP_lr += 1

            FN_lr = 0
            for frm in firms_nonfraud_lr:
                if frm in forward_misstate_firms:
                    FN_lr += 1

            precision_lr1[m] = TP_lr / (TP_lr + FP_lr)
            sensitivity_OOS_lr1[m] = TP_lr / (TP_lr + FN_lr)
            specificity_OOS_lr1[m] = TN_lr / (TN_lr + FP_lr)
            ecm_lr1[m] = (FP_lr * C_FP + FN_lr * C_FN) / (TP_lr + TN_lr + FP_lr + FN_lr)

            # LogitBoost
            base_tree = LogisticRegression(random_state=0)
            clf_ada = AdaBoostClassifier(n_estimators=opt_params_ada['n_estimators'], \
                                         learning_rate=opt_params_ada['learning_rate'], \
                                         estimator=base_tree, random_state=0)
            clf_ada = clf_ada.fit(X, Y)
            probs_oos_fraud_ada = clf_ada.predict_proba(X_OOS)[:, -1]

            cutoff_OOS_ada = np.percentile(probs_oos_fraud_ada, 99)

            idx_top1ada = np.where(probs_oos_fraud_ada >= cutoff_OOS_ada)[0]
            firms_top1ada = tbl_year_OOS['gvkey'][idx_top1ada].values
            idx_nonfraud_ada = np.where(probs_oos_fraud_ada < cutoff_OOS_ada)[0]
            firms_nonfraud_ada = tbl_year_OOS['gvkey'][idx_nonfraud_ada].values

            TP_ada = 0
            for frm in firms_top1ada:
                if frm in forward_misstate_firms:
                    TP_ada += 1

            TN_ada = 0
            for frm in firms_nonfraud_ada:
                if frm not in forward_misstate_firms:
                    TN_ada += 1

            FP_ada = 0
            for frm in firms_top1ada:
                if frm not in forward_misstate_firms:
                    FP_ada += 1

            FN_ada = 0
            for frm in firms_nonfraud_ada:
                if frm in forward_misstate_firms:
                    FN_ada += 1

            precision_ada1[m] = TP_ada / (TP_ada + FP_ada)
            sensitivity_OOS_ada1[m] = TP_ada / (TP_ada + FN_ada)
            specificity_OOS_ada1[m] = TN_ada / (TN_ada + FP_ada)
            ecm_ada1[m] = (FP_ada * C_FP + FN_ada * C_FN) / (TP_ada + TN_ada + FP_ada + FN_ada)

            # Fused approach
            weight_ser = np.array([score_svm, score_lr, score_ada])
            weight_ser = weight_ser / np.sum(weight_ser)

            probs_oos_fraud_svm = (1 + np.exp(-1 * probs_oos_fraud_svm)) ** -1

            probs_oos_fraud_lr = (1 + np.exp(-1 * probs_oos_fraud_lr)) ** -1

            probs_oos_fraud_ada = (1 + np.exp(-1 * probs_oos_fraud_ada)) ** -1

            clf_fused = np.dot(np.array([probs_oos_fraud_svm, \
                                         probs_oos_fraud_lr, \
                                         probs_oos_fraud_ada]).T, weight_ser)

            probs_oos_fraud_fused = clf_fused

            cutoff_OOS_fused = np.percentile(probs_oos_fraud_fused, 99)

            idx_top1fused = np.where(probs_oos_fraud_fused >= cutoff_OOS_fused)[0]
            firms_top1fused = tbl_year_OOS['gvkey'][idx_top1fused].values
            idx_nonfraud_fused = np.where(probs_oos_fraud_fused < cutoff_OOS_fused)[0]
            firms_nonfraud_fused = tbl_year_OOS['gvkey'][idx_nonfraud_fused].values

            TP_fused = 0
            for frm in firms_top1fused:
                if frm in forward_misstate_firms:
                    TP_fused += 1

            TN_fused = 0
            for frm in firms_nonfraud_fused:
                if frm not in forward_misstate_firms:
                    TN_fused += 1

            FP_fused = 0
            for frm in firms_top1fused:
                if frm not in forward_misstate_firms:
                    FP_fused += 1

            FN_fused = 0
            for frm in firms_nonfraud_fused:
                if frm in forward_misstate_firms:
                    FN_fused += 1

            precision_fused1[m] = TP_fused / (TP_fused + FP_fused)
            sensitivity_OOS_fused1[m] = TP_fused / (TP_fused + FN_fused)
            specificity_OOS_fused1[m] = TN_fused / (TN_fused + FP_fused)
            ecm_fused1[m] = (FP_fused * C_FP + FN_fused * C_FN) / (TP_fused + TN_fused + FP_fused + FN_fused)

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        f1_score_svm1 = 2 * (precision_svm1 * sensitivity_OOS_svm1) / \
                        (precision_svm1 + sensitivity_OOS_svm1 + 1e-8)

        f1_score_lr1 = 2 * (precision_lr1 * sensitivity_OOS_lr1) / \
                       (precision_lr1 + sensitivity_OOS_lr1 + 1e-8)

        f1_score_ada1 = 2 * (precision_ada1 * sensitivity_OOS_ada1) / \
                        (precision_ada1 + sensitivity_OOS_ada1 + 1e-8)

        f1_score_fused1 = 2 * (precision_fused1 * sensitivity_OOS_fused1) / \
                          (precision_fused1 + sensitivity_OOS_fused1 + 1e-8)

        f1_score_svm_fk1 = 2 * (precision_svm_fk1 * sensitivity_svm_fk1) / \
                           (precision_svm_fk1 + sensitivity_svm_fk1 + 1e-8)

        f1_score_rus = 2 * (precision_rus * sensitivity_rus) / \
                       (precision_rus + sensitivity_rus + 1e-8)

        # create performance table now
        forward_tbl = pd.DataFrame()
        forward_tbl['models'] = ['SVM', 'LR', 'LogitBoost', 'FUSED', 'SVM-FK', 'RUSBoost']

        forward_tbl['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_svm1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(sensitivity_OOS_svm1) * 100, 2)) + '%)',
                                              str(np.round(
                                                  np.mean(sensitivity_OOS_lr1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(sensitivity_OOS_lr1) * 100, 2)) + '%)',
                                              str(np.round(
                                                  np.mean(sensitivity_OOS_ada1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(sensitivity_OOS_ada1) * 100, 2)) + '%)',
                                              str(np.round(
                                                  np.mean(sensitivity_OOS_fused1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(sensitivity_OOS_fused1) * 100, 2)) + '%)',
                                              str(np.round(
                                                  np.mean(sensitivity_svm_fk1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(sensitivity_svm_fk1) * 100, 2)) + '%)',
                                              str(np.round(
                                                  np.mean(sensitivity_rus) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(sensitivity_rus) * 100, 2)) + '%)'
                                              ]


        forward_tbl['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_svm1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(specificity_OOS_svm1) * 100, 2)) + '%)',
                                              str(np.round(
                                                  np.mean(specificity_OOS_lr1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(specificity_OOS_lr1) * 100, 2)) + '%)',
                                              str(np.round(
                                                  np.mean(specificity_OOS_ada1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(specificity_OOS_ada1) * 100, 2)) + '%)',
                                              str(np.round(np.mean(specificity_OOS_fused1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(specificity_OOS_fused1) * 100, 2)) + '%)',
                                              str(np.round(np.mean(specificity_svm_fk1) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(specificity_svm_fk1) * 100, 2)) + '%)',
                                              str(np.round(np.mean(specificity_rus) * 100, 2)) + '% (' + \
                                              str(np.round(np.std(specificity_rus) * 100, 2)) + '%)'
                                              ]


        forward_tbl['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(precision_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(precision_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(precision_ada1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(precision_fused1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(precision_fused1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(precision_svm_fk1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(precision_svm_fk1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(precision_rus) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(precision_rus) * 100, 2)) + '%)'
                                            ]

        forward_tbl['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(f1_score_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(f1_score_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(f1_score_ada1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(f1_score_fused1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(f1_score_fused1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(f1_score_svm_fk1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(f1_score_svm_fk1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(f1_score_rus) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(f1_score_rus) * 100, 2)) + '%)'
                                           ]

        forward_tbl['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1) * 100, 2)) + '% (' + \
                                      str(np.round(np.std(ecm_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1) * 100, 2)) + '% (' + \
                                      str(np.round(np.std(ecm_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1) * 100, 2)) + '% (' + \
                                      str(np.round(np.std(ecm_ada1) * 100, 2)) + '%)',
                                      str(np.round(
                                          np.mean(ecm_fused1) * 100, 2)) + '% (' + \
                                      str(np.round(np.std(ecm_fused1) * 100, 2)) + '%)',
                                      str(np.round(
                                          np.mean(ecm_svm_fk1) * 100, 2)) + '% (' + \
                                      str(np.round(np.std(ecm_svm_fk1) * 100, 2)) + '%)',
                                      str(np.round(
                                          np.mean(ecm_rus) * 100, 2)) + '% (' + \
                                      str(np.round(np.std(ecm_rus) * 100, 2)) + '%)'
                                      ]


        lbl_perf_tbl = 'forward_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       ',OOS=' + str(OOS_period) + ',serial=' + str(
            adjust_serial) + '_forward_prediction.csv'

        if write == True:
            forward_tbl.to_csv(lbl_perf_tbl, index=False)
        print(forward_tbl)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


    def compare_ada(self, C_FN=30, C_FP=1):
        """
        This code uses 11 financial ratios to compare performance of an AdaBoost with
        decision tree learners with AdaBoost with Logistic Regression

        Predictive models:
            – Adaptive Boosting with Decision Tree (AdaBoost-Tree)
            – Adaptive Boosting with Logistic Regression (LogitBoost)

        Methodological choices:
            - 11 financial ratios from Dechow et al.(2011)
            - 10-fold validation
            - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score
        from sklearn.tree import DecisionTreeClassifier
        from datetime import datetime
        from extra_codes import ndcg_k
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cross_val = self.cv
        case_window = self.sa
        fraud_df = self.df
        write = self.w

        fyears_available = np.unique(fraud_df.fyear)

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, -14:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]

        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -11:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        # redo cross-validation if you wish
        if cross_val is True:

            # optimise AdaBoost with logistic regression (Ada-LR)
            print('Grid search hyperparameter optimisation started for AdaBoost')
            t1 = datetime.now()

            base_lr = LogisticRegression(random_state=0, solver='newton-cg')

            pipe_ada = Pipeline([('scale', StandardScaler()), \
                                 ('base_mdl_ada', AdaBoostClassifier(estimator=base_lr, random_state=0))])

            estimators = list(range(10, 3001, 10))
            learning_rates = [x / 1000 for x in range(10, 1001, 10)]

            param_grid_ada = {'base_mdl_ada__n_estimators': estimators, \
                              'base_mdl_ada__learning_rate': learning_rates}

            best_perf_ada = 0

            clf_ada_lr = GridSearchCV(pipe_ada, param_grid_ada, scoring='roc_auc', \
                                      n_jobs=-1, cv=k_fold, refit=False)
            clf_ada_lr.fit(X_CV, Y_CV)
            score_ada_lr = clf_ada_lr.best_score_
            if score_ada_lr >= best_perf_ada:
                best_perf_ada = score_ada_lr
                opt_params_ada_lr = clf_ada_lr.best_params_

            t2 = datetime.now()
            dt = t2 - t1
            print('AdaBoost-LR CV finished after ' + str(dt.total_seconds()) + ' sec')

            print('AdaBoost-LR: The optimal number of estimators is ' + \
                  str(opt_params_ada_lr['base_mdl_ada__n_estimators']) + ', and learning rate ' + \
                  str(opt_params_ada_lr['base_mdl_ada__learning_rate']))

            # optimise AdaBoost with tree learners (Ada-Tree): this is the basic model
            t1 = datetime.now()

            best_perf_ada_tree = 0

            base_tree = DecisionTreeClassifier(min_samples_leaf=5)

            pipe_ada_tree = Pipeline([('scale', StandardScaler()), \
                                      ('base_mdl_ada_tree',
                                       AdaBoostClassifier(estimator=base_tree, random_state=0))])

            estimators = list(range(10, 3001, 10))
            learning_rates = [x / 1000 for x in range(10, 1001, 10)]
            param_grid_ada_tree = {'base_mdl_ada_tree__n_estimators': estimators, \
                                   'base_mdl_ada_tree__learning_rate': learning_rates}

            clf_ada_tree = GridSearchCV(pipe_ada_tree, param_grid_ada_tree, scoring='roc_auc', \
                                        n_jobs=-1, cv=k_fold, refit=False)
            clf_ada_tree.fit(X_CV, Y_CV)
            score_ada_tree = clf_ada_tree.best_score_
            if score_ada_tree > best_perf_ada_tree:
                best_perf_ada_tree = score_ada_tree
                opt_params_ada_tree = clf_ada_tree.best_params_

            t2 = datetime.now()
            dt = t2 - t1
            print('AdaBoost-Tree CV finished after ' + str(dt.total_seconds()) + ' sec')

            print('AdaBoost-Tree: The optimal number of estimators is ' + \
                  str(opt_params_ada_tree['base_mdl_ada_tree__n_estimators']) + ', and learning rate ' + \
                  str(opt_params_ada_tree['base_mdl_ada_tree__learning_rate']))

            print('Hyperparameter optimisation finished successfully.\nStarting the main analysis now...')

        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)

        roc_ada_tree = np.zeros(len(range_oos))
        specificity_ada_tree = np.zeros(len(range_oos))
        sensitivity_OOS_ada_tree = np.zeros(len(range_oos))
        precision_ada_tree = np.zeros(len(range_oos))
        sensitivity_OOS_ada_tree1 = np.zeros(len(range_oos))
        specificity_OOS_ada_tree1 = np.zeros(len(range_oos))
        precision_ada_tree1 = np.zeros(len(range_oos))
        ndcg_ada_tree1 = np.zeros(len(range_oos))
        ecm_ada_tree1 = np.zeros(len(range_oos))

        roc_ada_lr = np.zeros(len(range_oos))
        specificity_ada_lr = np.zeros(len(range_oos))
        sensitivity_OOS_ada_lr = np.zeros(len(range_oos))
        precision_ada_lr = np.zeros(len(range_oos))
        sensitivity_OOS_ada_lr1 = np.zeros(len(range_oos))
        specificity_OOS_ada_lr1 = np.zeros(len(range_oos))
        precision_ada_lr1 = np.zeros(len(range_oos))
        ndcg_ada_lr1 = np.zeros(len(range_oos))
        ecm_ada_lr1 = np.zeros(len(range_oos))

        m = 0
        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period

            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True


            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -11:]
            mean_vals = np.mean(X)
            std_vals = np.std(X)
            X = (X - mean_vals) / std_vals
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS = (X_OOS - mean_vals) / std_vals

            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            # Adaptive Boosting with logistic regression for weak learners (LogitBoost)
            base_lr = LogisticRegression(random_state=0, solver='newton-cg')
            clf_ada_lr = AdaBoostClassifier(n_estimators=opt_params_ada_lr['base_mdl_ada__n_estimators'], \
                                            learning_rate=opt_params_ada_lr['base_mdl_ada__learning_rate'], \
                                            estimator=base_lr, random_state=0)
            clf_ada_lr = clf_ada_lr.fit(X, Y)
            probs_oos_fraud_ada_lr = clf_ada_lr.predict_proba(X_OOS)[:, -1]

            labels_ada_lr = clf_ada_lr.predict(X_OOS)

            roc_ada_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada_lr)
            specificity_ada_lr[m] = np.sum(np.logical_and(labels_ada_lr == 0, Y_OOS == 0)) / \
                                    np.sum(Y_OOS == 0)
            if np.sum(labels_ada_lr) > 0:
                sensitivity_OOS_ada_lr[m] = np.sum(np.logical_and(labels_ada_lr == 1, \
                                                                  Y_OOS == 1)) / np.sum(Y_OOS)
                precision_ada_lr[m] = np.sum(np.logical_and(labels_ada_lr == 1, Y_OOS == 1)) / np.sum(labels_ada_lr)

            cutoff_OOS_ada_lr = np.percentile(probs_oos_fraud_ada_lr, 99)
            sensitivity_OOS_ada_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_ada_lr >= cutoff_OOS_ada_lr, \
                                                               Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_ada_lr < cutoff_OOS_ada_lr, \
                                                               Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_ada_lr >= cutoff_OOS_ada_lr, \
                                                         Y_OOS == 1)) / np.sum(
                probs_oos_fraud_ada_lr >= cutoff_OOS_ada_lr)
            ndcg_ada_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada_lr, 99)

            FN_ada_lr1 = np.sum(np.logical_and(probs_oos_fraud_ada_lr < cutoff_OOS_ada_lr, \
                                               Y_OOS == 1))
            FP_ada_lr1 = np.sum(np.logical_and(probs_oos_fraud_ada_lr >= cutoff_OOS_ada_lr, \
                                               Y_OOS == 0))

            ecm_ada_lr1[m] = C_FN * P_f * FN_ada_lr1 / n_P + C_FP * P_nf * FP_ada_lr1 / n_N

            # Adaptive Boosting with decision trees as weak learners (AdaBoost)
            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            clf_ada_tree = AdaBoostClassifier(n_estimators=opt_params_ada_tree['base_mdl_ada_tree__n_estimators'], \
                                              learning_rate=opt_params_ada_tree['base_mdl_ada_tree__learning_rate'], \
                                              estimator=base_tree, random_state=0)
            clf_ada_tree = clf_ada_tree.fit(X, Y)
            probs_oos_fraud_ada_tree = clf_ada_tree.predict_proba(X_OOS)[:, -1]

            labels_ada_tree = clf_ada_tree.predict(X_OOS)

            roc_ada_tree[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada_tree)
            specificity_ada_tree[m] = np.sum(np.logical_and(labels_ada_tree == 0, Y_OOS == 0)) / \
                                      np.sum(Y_OOS == 0)
            if np.sum(labels_ada_tree) > 0:
                sensitivity_OOS_ada_tree[m] = np.sum(np.logical_and(labels_ada_tree == 1, \
                                                                    Y_OOS == 1)) / np.sum(Y_OOS)
                precision_ada_tree[m] = np.sum(np.logical_and(labels_ada_tree == 1, Y_OOS == 1)) / np.sum(
                    labels_ada_tree)

            cutoff_OOS_ada_tree = np.percentile(probs_oos_fraud_ada_tree, 99)
            sensitivity_OOS_ada_tree1[m] = np.sum(np.logical_and(probs_oos_fraud_ada_tree >= cutoff_OOS_ada_tree, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada_tree1[m] = np.sum(np.logical_and(probs_oos_fraud_ada_tree < cutoff_OOS_ada_tree, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada_tree1[m] = np.sum(np.logical_and(probs_oos_fraud_ada_tree >= cutoff_OOS_ada_tree, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_ada_tree >= cutoff_OOS_ada_tree)
            ndcg_ada_tree1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada_tree, 99)

            FN_ada_tree1 = np.sum(np.logical_and(probs_oos_fraud_ada_tree < cutoff_OOS_ada_tree, \
                                                 Y_OOS == 1))
            FP_ada_tree1 = np.sum(np.logical_and(probs_oos_fraud_ada_tree >= cutoff_OOS_ada_tree, \
                                                 Y_OOS == 0))

            ecm_ada_tree1[m] = C_FN * P_f * FN_ada_tree1 / n_P + C_FP * P_nf * FP_ada_tree1 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['AdaBoost', 'LogitBoost']
        perf_tbl_general['Roc'] = [np.mean(roc_ada_tree), np.mean(roc_ada_lr)]

        perf_tbl_general['Sensitivity @ 1 Prc'] = [np.mean(sensitivity_OOS_ada_tree1), \
                                                   np.mean(sensitivity_OOS_ada_lr1)]

        perf_tbl_general['Specificity @ 1 Prc'] = [np.mean(specificity_OOS_ada_tree1), \
                                                   np.mean(specificity_OOS_ada_lr1)]

        perf_tbl_general['Precision @ 1 Prc'] = [np.mean(precision_ada_tree1), \
                                                 np.mean(precision_ada_lr1)]

        perf_tbl_general['F1 Score @ 1 Prc'] = 2 * (perf_tbl_general['Precision @ 1 Prc'] * \
                                                    perf_tbl_general['Sensitivity @ 1 Prc']) / \
                                               ((perf_tbl_general['Precision @ 1 Prc'] + \
                                                 perf_tbl_general['Sensitivity @ 1 Prc']))
        perf_tbl_general['NDCG @ 1 Prc'] = [np.mean(ndcg_ada_tree1), \
                                            np.mean(ndcg_ada_lr1)]

        perf_tbl_general['ECM @ 1 Prc'] = [np.mean(ecm_ada_tree1), \
                                           np.mean(ecm_ada_lr1)]

        lbl_perf_tbl = 'Compare_Ada_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + '_with_LogitBoost.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')

    def ratio_2003(self, C_FN=30, C_FP=1):
        """
        This code is the same as ratio_analyse, but the period is shorted to 2003-2008,
        which is the period in Bao et al. (2020).
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, -14:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]

        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -11:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        # redo cross-validation if you wish
        if cv_type == 'kfold':
            if cross_val == True:

                # optimise LogitBoost
                print('Grid search hyperparameter optimisation started for AdaBoost')
                t1 = datetime.now()

                best_perf_ada = 0

                base_lr = LogisticRegression(random_state=0, solver='newton-cg')

                pipe_ada = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_ada', AdaBoostClassifier(estimator=base_lr, random_state=0))])

                estimators = list(range(10, 3001, 10))
                learning_rates = [x / 1000 for x in range(10, 1001, 10)]
                param_grid_ada = {'base_mdl_ada__n_estimators': estimators, \
                                  'base_mdl_ada__learning_rate': learning_rates}

                clf_ada = GridSearchCV(pipe_ada, param_grid_ada, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_ada.fit(X_CV, Y_CV)
                score_ada = clf_ada.best_score_
                if score_ada >= best_perf_ada:
                    best_perf_ada = score_ada
                    opt_params_ada = clf_ada.best_params_


                print('LogitBoost: The optimal number of estimators is ' + \
                      str(opt_params_ada['base_mdl_ada__n_estimators']) + ', and learning rate is ' + \
                      str(opt_params_ada['base_mdl_ada__learning_rate']))

                print('Computing CV ROC for LR ...')
                t1 = datetime.now()
                score_lr = []
                for m in range(0, k_fold):
                    train_sample, test_sample = train_test_split(Y_CV, test_size=1 /
                                                                                 k_fold, shuffle=False, random_state=m)
                    X_train = X_CV.iloc[train_sample.index]
                    X_train = add_constant(X_train)
                    Y_train = train_sample
                    X_test = X_CV.iloc[test_sample.index]
                    X_test = add_constant(X_test)
                    Y_test = test_sample

                    logit_model = Logit(Y_train, X_train)
                    logit_model = logit_model.fit(disp=0)
                    pred_LR_CV = logit_model.predict(X_test)
                    score_lr.append(roc_auc_score(Y_test, pred_LR_CV))

                score_lr = np.mean(score_lr)

                t2 = datetime.now()
                dt = t2 - t1
                print('LogitBoost CV finished after ' + str(dt.total_seconds()) + ' sec')


                # optimize SVM
                print('Grid search hyperparameter optimisation started for SVM')
                t1 = datetime.now()

                pipe_svm = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_svm', SVC(shrinking=False, \
                                                          probability=False, random_state=0, max_iter=-1, \
                                                          tol=X_CV.shape[-1] * 1e-3))])
                C = [0.001, 0.01, 0.1, 1, 10, 100]
                gamma = [0.0001, 0.001, 0.01, 0.1]
                kernel = ['rbf', 'linear', 'poly']
                class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]

                param_grid_svm = {'base_mdl_svm__kernel': kernel,
                                  'base_mdl_svm__class_weight': class_weight,
                                  'base_mdl_svm__C': C,
                                  'base_mdl_svm__gamma': gamma}

                clf_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_svm.fit(X_CV, Y_CV)
                opt_params_svm = clf_svm.best_params_
                kernel_opt = opt_params_svm['base_mdl_svm__kernel']
                cw_opt = opt_params_svm['base_mdl_svm__class_weight']
                gamma_opt = opt_params_svm['base_mdl_svm__gamma']
                c_opt = opt_params_svm['base_mdl_svm__C']
                score_svm = clf_svm.best_score_

                t2 = datetime.now()
                dt = t2 - t1
                print('SVM CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('SVM: The optimal class weight is ' + str(cw_opt) + ', gamma is ' + str(
                    gamma_opt) + ', C is ' + str(c_opt) + ',score is ' + str(score_svm) +\
                      ', kernel is ' + str(kernel_opt))


        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)

        roc_svm = np.zeros(len(range_oos))
        sensitivity_OOS_svm1 = np.zeros(len(range_oos))
        specificity_OOS_svm1 = np.zeros(len(range_oos))
        precision_svm1 = np.zeros(len(range_oos))
        ndcg_svm1 = np.zeros(len(range_oos))
        ecm_svm1 = np.zeros(len(range_oos))

        roc_lr = np.zeros(len(range_oos))
        sensitivity_OOS_lr1 = np.zeros(len(range_oos))
        specificity_OOS_lr1 = np.zeros(len(range_oos))
        precision_lr1 = np.zeros(len(range_oos))
        ndcg_lr1 = np.zeros(len(range_oos))
        ecm_lr1 = np.zeros(len(range_oos))

        roc_ada = np.zeros(len(range_oos))
        sensitivity_OOS_ada1 = np.zeros(len(range_oos))
        specificity_OOS_ada1 = np.zeros(len(range_oos))
        precision_ada1 = np.zeros(len(range_oos))
        ndcg_ada1 = np.zeros(len(range_oos))
        ecm_ada1 = np.zeros(len(range_oos))

        roc_fused = np.zeros(len(range_oos))
        sensitivity_OOS_fused1 = np.zeros(len(range_oos))
        specificity_OOS_fused1 = np.zeros(len(range_oos))
        precision_fused1 = np.zeros(len(range_oos))
        ndcg_fused1 = np.zeros(len(range_oos))
        ecm_fused1 = np.zeros(len(range_oos))

        m = 0
        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -11:]
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS = (X_OOS - mean) / std
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            # SVM
            clf_svm = SVC(class_weight=opt_params_svm['base_mdl_svm__class_weight'],
                          kernel=opt_params_svm['base_mdl_svm__kernel'], \
                          C=opt_params_svm['base_mdl_svm__C'], gamma=opt_params_svm['base_mdl_svm__gamma'],\
                          shrinking=False, \
                          probability=False, random_state=0, max_iter=-1, \
                          tol=X.shape[-1] * 1e-3)

            clf_svm = clf_svm.fit(X, Y)

            pred_test_svm = clf_svm.decision_function(X_OOS)
            probs_oos_fraud_svm = np.exp(pred_test_svm) / (1 + np.exp(pred_test_svm))

            roc_svm[m] = roc_auc_score(Y_OOS, probs_oos_fraud_svm)

            cutoff_OOS_svm = np.percentile(probs_oos_fraud_svm, 99)
            sensitivity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_svm >= cutoff_OOS_svm)
            ndcg_svm1[m] = ndcg_k(Y_OOS, probs_oos_fraud_svm, 99)

            FN_svm1 = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                            Y_OOS == 1))
            FP_svm1 = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                            Y_OOS == 0))

            ecm_svm1[m] = C_FN * P_f * FN_svm1 / n_P + C_FP * P_nf * FP_svm1 / n_N

            # Logistic Regression – Dechow et al (2011)
            X_lr = add_constant(X)
            X_OOS_lr = add_constant(X_OOS)
            clf_lr = Logit(Y, X_lr)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS_lr)

            roc_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_lr)

            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)
            sensitivity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                           Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                                           Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                     Y_OOS == 1)) / np.sum(probs_oos_fraud_lr >= cutoff_OOS_lr)
            ndcg_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_lr, 99)

            FN_lr1 = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                           Y_OOS == 1))
            FP_lr1 = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                           Y_OOS == 0))

            ecm_lr1[m] = C_FN * P_f * FN_lr1 / n_P + C_FP * P_nf * FP_lr1 / n_N

            # LogitBoost
            base_lr = LogisticRegression(random_state=0, solver='newton-cg')

            clf_ada = AdaBoostClassifier(n_estimators=opt_params_ada['base_mdl_ada__n_estimators'], \
                                         learning_rate=opt_params_ada['base_mdl_ada__learning_rate'], \
                                         estimator=base_lr, random_state=0)
            clf_ada = clf_ada.fit(X, Y)
            probs_oos_fraud_ada = clf_ada.predict_proba(X_OOS)[:, -1]

            labels_ada = clf_ada.predict(X_OOS)

            roc_ada[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada)
            cutoff_OOS_ada = np.percentile(probs_oos_fraud_ada, 99)
            sensitivity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_ada >= cutoff_OOS_ada)
            ndcg_ada1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada, 99)

            FN_ada1 = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                            Y_OOS == 1))
            FP_ada1 = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                            Y_OOS == 0))

            ecm_ada1[m] = C_FN * P_f * FN_ada1 / n_P + C_FP * P_nf * FP_ada1 / n_N


            # Fused approach
            weight_ser = np.array([score_svm, score_lr, score_ada])
            weight_ser = weight_ser / np.sum(weight_ser)
            probs_oos_fraud_svm = (1 + np.exp(-1 * probs_oos_fraud_svm)) ** -1

            probs_oos_fraud_lr = (1 + np.exp(-1 * probs_oos_fraud_lr)) ** -1

            probs_oos_fraud_ada = (1 + np.exp(-1 * probs_oos_fraud_ada)) ** -1

            clf_fused = np.dot(np.array([probs_oos_fraud_svm, \
                                         probs_oos_fraud_lr, probs_oos_fraud_ada]).T, weight_ser)

            probs_oos_fraud_fused = clf_fused

            roc_fused[m] = roc_auc_score(Y_OOS, probs_oos_fraud_fused)

            cutoff_OOS_fused = np.percentile(probs_oos_fraud_fused, 99)
            sensitivity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                              Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                                              Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                        Y_OOS == 1)) / np.sum(probs_oos_fraud_fused >= cutoff_OOS_fused)
            ndcg_fused1[m] = ndcg_k(Y_OOS, probs_oos_fraud_fused, 99)

            FN_fused1 = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                              Y_OOS == 1))
            FP_fused1 = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                              Y_OOS == 0))

            ecm_fused1[m] = C_FN * P_f * FN_fused1 / n_P + C_FP * P_nf * FP_fused1 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        f1_score_svm1 = 2 * (precision_svm1 * sensitivity_OOS_svm1) / \
                        (precision_svm1 + sensitivity_OOS_svm1 + 1e-8)

        f1_score_lr1 = 2 * (precision_lr1 * sensitivity_OOS_lr1) / \
                       (precision_lr1 + sensitivity_OOS_lr1 + 1e-8)

        f1_score_ada1 = 2 * (precision_ada1 * sensitivity_OOS_ada1) / \
                        (precision_ada1 + sensitivity_OOS_ada1 + 1e-8)

        f1_score_fused1 = 2 * (precision_fused1 * sensitivity_OOS_fused1) / \
                          (precision_fused1 + sensitivity_OOS_fused1 + 1e-8)

        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['SVM', 'LR', 'LogitBoost', 'FUSED']
        perf_tbl_general['Roc'] = [str(np.round(
            np.mean(roc_svm[2:8]) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr[2:8]) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada[2:8]) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada[2:8]) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused[2:8]) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_svm1[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_svm1[2:8]) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_lr1[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_lr1[2:8]) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_ada1[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_ada1[2:8]) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_fused1[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_fused1[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_svm1[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_svm1[2:8]) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_lr1[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_lr1[2:8]) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_ada1[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_ada1[2:8]) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_OOS_fused1[2:8]) * 100,
                                                                2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_fused1[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1[2:8]) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1[2:8]) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_lr1[2:8]) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1[2:8]) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_ada1[2:8]) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1[2:8]) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1[2:8]) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1[2:8]) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1[2:8]) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1[2:8]) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1[2:8]) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1[2:8]) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1[2:8]) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1[2:8]) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1[2:8]) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1[2:8]) * 100, 2)) + '%)']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_ratio_0308.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')

    def RUS28_2003(self, C_FN=30, C_FP=1):
        """
        This code is the same as RUS_28, but the period is shorted to 2003-2008.
        """

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k

        t0 = datetime.now()
        IS_period=self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-14]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        range_oos = range(start_OOS_year, end_OOS_year + 1)
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= sample_start)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -28:]
        X_CV = X_CV.apply(lambda x: x / np.linalg.norm(x))
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        # optimize RUSBoost number of estimators
        if cv_type == 'kfold':
            if cross_val == True:

                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1 = datetime.now()

                n_estimators = list(range(10, 3001, 10))
                learning_rate = list(x / 1000 for x in range(10, 1001, 10))
                param_grid_rusboost = {'n_estimators': n_estimators,
                                       'learning_rate': learning_rate}

                base_tree = DecisionTreeClassifier(min_samples_leaf=5)
                bao_RUSboost = RUSBoostClassifier(estimator=base_tree, \
                                                  sampling_strategy=1, random_state=0)
                clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_rus.fit(X_CV, Y_CV)
                opt_params_rus = clf_rus.best_params_
                n_opt_rus = opt_params_rus['n_estimators']
                r_opt_rus = opt_params_rus['learning_rate']
                score_rus = clf_rus.best_score_
                t2 = datetime.now()
                dt = t2 - t1
                print('RUSBoost CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('RUSBoost: The optimal number of estimators is ' + str(n_opt_rus) + 'learning rate is ' + str(
                    r_opt_rus))


        roc_rusboost = np.zeros(len(range_oos))
        specificity_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost = np.zeros(len(range_oos))
        precision_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1 = np.zeros(len(range_oos))
        specificity_OOS_rusboost1 = np.zeros(len(range_oos))
        precision_rusboost1 = np.zeros(len(range_oos))
        ndcg_rusboost1 = np.zeros(len(range_oos))
        ecm_rusboost1 = np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()

            year_start_IS = sample_start
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[reduced_tbl.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -28:]
            X = X.apply(lambda x: x / np.linalg.norm(x))
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -28:]
            X_OOS = X_OOS.apply(lambda x: x / np.linalg.norm(x))
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)
            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=n_opt_rus, \
                                              learning_rate=r_opt_rus, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X, Y)

            probs_oos_fraud_rusboost = clf_rusboost.predict_proba(X_OOS)[:, -1]

            labels_rusboost = clf_rusboost.predict(X_OOS)

            roc_rusboost[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rusboost)
            specificity_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 0, Y_OOS == 0)) / \
                                      np.sum(Y_OOS == 0)

            sensitivity_OOS_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, \
                                                                Y_OOS == 1)) / np.sum(Y_OOS)
            precision_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, Y_OOS == 1)) / np.sum(labels_rusboost)

            cutoff_OOS_rusboost = np.percentile(probs_oos_fraud_rusboost, 99)
            sensitivity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_rusboost >= cutoff_OOS_rusboost)
            ndcg_rusboost1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rusboost, 99)

            FN_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                 Y_OOS == 1))
            FP_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                 Y_OOS == 0))

            ecm_rusboost1[m] = C_FN * P_f * FN_rusboost1 / n_P + C_FP * P_nf * FP_rusboost1 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['RUSBoost-28_bao_2003']

        perf_tbl_general['Roc'] = str(np.round(
            np.mean(roc_rusboost[2:8]) * 100, 2)) + '% (' + \
                                  str(np.round(np.std(roc_rusboost[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['Sensitivity @ 1 Prc'] = str(np.round(
            np.mean(sensitivity_OOS_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(sensitivity_OOS_rusboost1[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['Specificity @ 1 Prc'] = str(np.round(
            np.mean(specificity_OOS_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(specificity_OOS_rusboost1[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['Precision @ 1 Prc'] = str(np.round(
            np.mean(precision_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(precision_rusboost1[2:8]) * 100, 2)) + '%)'

        f1_score_rusboost1 = 2 * (precision_rusboost1 * sensitivity_OOS_rusboost1) / \
                             (precision_rusboost1 + sensitivity_OOS_rusboost1 + 1e-8)

        perf_tbl_general['F1 Score @ 1 Prc'] = str(np.round(
            np.mean(f1_score_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                               str(np.round(np.std(f1_score_rusboost1[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['NDCG @ 1 Prc'] = str(np.round(
            np.mean(ndcg_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ndcg_rusboost1[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['ECM @ 1 Prc'] = str(np.round(
            np.mean(ecm_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                          str(np.round(np.std(ecm_rusboost1[2:8]) * 100, 2)) + '%)'

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_RUS_0308.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


    def FK23_2003(self, C_FN=30, C_FP=1):

        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cross_val = self.cv
        case_window = self.sa
        fraud_df = self.df
        write = self.w

        dict_db = pickle.load(open('features_fk.pkl', 'r+b'))
        tbl_ratio_fk = dict_db['lagged_Data']
        mapped_X = dict_db['matrix']
        red_tbl_fk = tbl_ratio_fk.iloc[:, -46:]
        print('pickle file loaded successfully ...')

        idx_CV = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear >= sample_start,
                                             tbl_ratio_fk.fyear < start_OOS_year)].index
        Y_CV = tbl_ratio_fk.AAER_DUMMY[idx_CV]

        X_CV = mapped_X[idx_CV, :]
        idx_real = np.where(np.logical_and(np.isnan(X_CV).any(axis=1) == False, \
                                           np.isinf(X_CV).any(axis=1) == False))[0]
        X_CV = X_CV[idx_real, :]

        Y_CV = Y_CV.iloc[idx_real]
        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        print('Grid search hyperparameter optimisation started for SVM-FK')
        t1 = datetime.now()

        pipe_svm = Pipeline([('scale', StandardScaler()), \
                             ('base_mdl_svm', SVC(shrinking=False, \
                                                  probability=False, random_state=0, cache_size=1000, \
                                                  tol=X_CV.shape[-1] * 1e-3))])

        C = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma = [0.0001, 0.001, 0.01, 0.1]
        kernel = ['rbf', 'linear', 'poly']
        class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]

        param_grid_svm = {'base_mdl_svm__kernel': kernel, \
                          'base_mdl_svm__C': C, \
                          'base_mdl_svm__class_weight': class_weight,
                          'base_mdl_svm__gamma': gamma}

        clf_svm_fk = GridSearchCV(pipe_svm, param_grid_svm, scoring='roc_auc', \
                                  n_jobs=None, cv=k_fold, refit=False)
        clf_svm_fk.fit(X_CV, Y_CV)
        opt_params_svm_fk = clf_svm_fk.best_params_
        cw_opt = opt_params_svm_fk['base_mdl_svm__class_weight']
        c_opt = opt_params_svm_fk['base_mdl_svm__C']
        kernel_opt = opt_params_svm_fk['base_mdl_svm__kernel']
        gamma_opt = opt_params_svm_fk['base_mdl_svm__gamma']
        score_svm = clf_svm_fk.best_score_

        t2 = datetime.now()
        dt = t2 - t1
        print('SVM CV finished after ' + str(dt.total_seconds()) + ' sec')
        print('SVM: The optimal class weight is ' + str(cw_opt) + \
              ', C is ' + str(c_opt) + ',score is' + str(score_svm) + \
              ', kernel is ' + str(kernel_opt) + ' ,gamma is ' + str(gamma_opt))


        t000 = datetime.now()

        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)  # (2001,2010+1,1)
        roc_ratio = np.zeros(len(range_oos))
        ndcg_ratio = np.zeros(len(range_oos))
        sensitivity_ratio = np.zeros(len(range_oos))
        specificity_ratio = np.zeros(len(range_oos))
        precision_ratio = np.zeros(len(range_oos))
        ecm_ratio = np.zeros(len(range_oos))
        f1_ratio = np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            idx_IS = tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear < yr - OOS_gap, \
                                                 tbl_ratio_fk.fyear >= year_start_IS)].index
            tbl_year_IS = tbl_ratio_fk.loc[idx_IS, :]
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = tbl_ratio_fk.loc[tbl_ratio_fk.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]

            X = mapped_X[idx_IS, :]
            idx_real = np.where(np.logical_and(np.isnan(X).any(axis=1) == False, \
                                               np.isinf(X).any(axis=1) == False))[0]
            X = X[idx_real, :]
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y = tbl_ratio_fk.AAER_DUMMY[idx_IS]
            Y = Y.iloc[idx_real]

            X_OOS = mapped_X[tbl_year_OOS.index, :]
            idx_real_OOS = np.where(np.logical_and(np.isnan(X_OOS).any(axis=1) == False, \
                                                   np.isinf(X_OOS).any(axis=1) == False))[0]
            X_OOS = X_OOS[idx_real_OOS, :]
            X_OOS = (X_OOS - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y_OOS = tbl_year_OOS.AAER_DUMMY
            Y_OOS = Y_OOS.iloc[idx_real_OOS]
            Y_OOS = Y_OOS.reset_index(drop=True)

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            t01 = datetime.now()

            svm_fk = SVC(class_weight=cw_opt, kernel=kernel_opt, C=c_opt, gamma=gamma_opt, shrinking=False, \
                         probability=False, random_state=0, cache_size=1000, \
                         tol=X.shape[-1] * 1e-3)
            clf_svm_fk = svm_fk.fit(X, Y)
            predicted_test = clf_svm_fk.decision_function(X_OOS)
            predicted_test[predicted_test >= 1] = 1 + np.log(predicted_test[predicted_test >= 1])
            predicted_test = np.exp(predicted_test) / (1 + np.exp(predicted_test))
            roc_ratio[m] = roc_auc_score(Y_OOS, predicted_test)

            cutoff_ratio = np.percentile(predicted_test, 99)
            labels_ratio = (predicted_test >= cutoff_ratio).astype(int)
            sensitivity_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_ratio[m] = np.sum(np.logical_and(labels_ratio == 0, Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ratio[m] = np.sum(np.logical_and(labels_ratio == 1, Y_OOS == 1)) / np.sum(labels_ratio)
            ndcg_ratio[m] = ndcg_k(Y_OOS.to_numpy(), predicted_test, 99)

            FN = np.sum(np.logical_and(predicted_test < cutoff_ratio, \
                                       Y_OOS == 1))
            FP = np.sum(np.logical_and(predicted_test >= cutoff_ratio, \
                                       Y_OOS == 0))

            ecm_ratio[m] = C_FN * P_f * FN / n_P + C_FP * P_nf * FP / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        f1_ratio = 2 * (precision_ratio * sensitivity_ratio) / (precision_ratio + sensitivity_ratio + 1e-8)

        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['FK23_2003']
        perf_tbl_general['Roc'] = [str(np.round(
            np.mean(roc_ratio[2:8]) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ratio[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_ratio[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_ratio[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_ratio[2:8]) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_ratio[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_ratio[2:8]) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ratio[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_ratio[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_ratio[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_ratio[2:8]) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ratio[2:8]) * 100, 2)) + '%)']

        perf_tbl_general['ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_ratio[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ratio[2:8]) * 100, 2)) + '%)']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_FK23_0308.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)

        t001 = datetime.now()
        dt00 = t001 - t000
        print('MC analysis is completed after ' + str(dt00.total_seconds()) + ' seconds')

    def AppendixE_excluding_missing_values(self, C_FN=30, C_FP=1):
        """
        This code replicates Bao et al. (2020)'s methodological choices and produces Panel B of Appendix E.
        WARNING!:
        1) For Panel A of Appendix E where missing values are included, please change the dataset used
        from "FraudDB2020.csv" to "FraudDB2020_including_missing_values.csv"!
        2) Remember to set OOS_gap = 1 in order to set up a two-year gap!
        """

        from sklearn.tree import DecisionTreeClassifier
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from statsmodels.tools import add_constant
        from statsmodels.discrete.discrete_model import Logit

        t0 = datetime.now()
        IS_period=self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, 9:-14]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        range_oos = range(start_OOS_year, end_OOS_year + 1)
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= sample_start)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)

        X_CV = tbl_year_IS_CV.iloc[:, -28:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        roc_rusboost = np.zeros(len(range_oos))
        specificity_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost = np.zeros(len(range_oos))
        precision_rusboost = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1 = np.zeros(len(range_oos))
        specificity_OOS_rusboost1 = np.zeros(len(range_oos))
        precision_rusboost1 = np.zeros(len(range_oos))
        ndcg_rusboost1 = np.zeros(len(range_oos))
        ecm_rusboost1 = np.zeros(len(range_oos))

        roc_lr=np.zeros(len(range_oos))
        sensitivity_OOS_lr1=np.zeros(len(range_oos))
        specificity_OOS_lr1=np.zeros(len(range_oos))
        precision_lr1=np.zeros(len(range_oos))
        ndcg_lr1=np.zeros(len(range_oos))
        ecm_lr1=np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()

            year_start_IS = sample_start
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            tbl_year_OOS = reduced_tbl.loc[reduced_tbl.fyear == yr]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -28:]
            X = X.apply(lambda x: x / np.linalg.norm(x))
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -28:]
            X_OOS = X_OOS.apply(lambda x: x / np.linalg.norm(x))
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            X_lr = tbl_year_IS.iloc[:, -28:]
            mean = np.mean(X_lr)
            std = np.std(X_lr)
            X_lr = (X_lr - mean) / std

            X_OOS_lr = tbl_year_OOS.iloc[:, -28:]
            X_OOS_lr = (X_OOS_lr - mean) / std

            misstate_test = np.unique(tbl_year_OOS[tbl_year_OOS['AAER_DUMMY'] == 1]['gvkey'])
            Y[np.isin(tbl_year_IS['gvkey'], misstate_test)] = 0

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            # Logistic Regression – Dechow et al (2011)
            X_lr = add_constant(X_lr)
            X_OOS_lr = add_constant(X_OOS_lr)
            clf_lr = Logit(Y, X_lr)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS_lr)

            roc_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_lr)

            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)
            sensitivity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                           Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                                           Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                     Y_OOS == 1)) / np.sum(probs_oos_fraud_lr >= cutoff_OOS_lr)
            ndcg_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_lr, 99)

            FN_lr1 = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                           Y_OOS == 1))
            FP_lr1 = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                           Y_OOS == 0))

            ecm_lr1[m] = C_FN * P_f * FN_lr1 / n_P + C_FP * P_nf * FP_lr1 / n_N

            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=300, \
                                              learning_rate=0.1, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X, Y)

            probs_oos_fraud_rusboost = clf_rusboost.predict_proba(X_OOS)[:, -1]

            labels_rusboost = clf_rusboost.predict(X_OOS)

            roc_rusboost[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rusboost)
            specificity_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 0, Y_OOS == 0)) / \
                                      np.sum(Y_OOS == 0)

            sensitivity_OOS_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, \
                                                                Y_OOS == 1)) / np.sum(Y_OOS)
            precision_rusboost[m] = np.sum(np.logical_and(labels_rusboost == 1, Y_OOS == 1)) / np.sum(labels_rusboost)

            cutoff_OOS_rusboost = np.percentile(probs_oos_fraud_rusboost, 99)
            sensitivity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_rusboost >= cutoff_OOS_rusboost)
            ndcg_rusboost1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rusboost, 99)

            FN_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                 Y_OOS == 1))
            FP_rusboost1 = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                 Y_OOS == 0))

            ecm_rusboost1[m] = C_FN * P_f * FN_rusboost1 / n_P + C_FP * P_nf * FP_rusboost1 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1

        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['RUSBoost-28_2003', 'Logit-28_2003']

        perf_tbl_general['Roc'] = str(np.round(
            np.mean(roc_rusboost[2:8]) * 100, 2)) + '% (' + \
                                  str(np.round(np.std(roc_rusboost[2:8]) * 100, 2)) + '%)', \
        str(np.round(np.mean(roc_lr[2:8]) * 100, 2)) + '% (' + \
        str(np.round(np.std(roc_lr[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['Sensitivity @ 1 Prc'] = str(np.round(
            np.mean(sensitivity_OOS_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(sensitivity_OOS_rusboost1[2:8]) * 100, 2)) + '%)', \
                                                  str(np.round(np.mean(sensitivity_OOS_lr1[2:8]) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(sensitivity_OOS_lr1[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['Specificity @ 1 Prc'] = str(np.round(
            np.mean(specificity_OOS_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(specificity_OOS_rusboost1[2:8]) * 100, 2)) + '%)', \
                                                  str(np.round(np.mean(specificity_OOS_lr1[2:8]) * 100, 2)) + '% (' + \
                                                  str(np.round(np.std(specificity_OOS_lr1[2:8]) * 100, 2)) + '%)'


        perf_tbl_general['Precision @ 1 Prc'] = str(np.round(
            np.mean(precision_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(precision_rusboost1[2:8]) * 100, 2)) + '%)', \
                                                str(np.round(np.mean(precision_lr1[2:8]) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(precision_lr1[2:8]) * 100, 2)) + '%)'

        f1_score_rusboost1 = 2 * (precision_rusboost1 * sensitivity_OOS_rusboost1) / \
                             (precision_rusboost1 + sensitivity_OOS_rusboost1 + 1e-8)

        f1_score_lr1=2*(precision_lr1*sensitivity_OOS_lr1)/\
            (precision_lr1+sensitivity_OOS_lr1+1e-8)

        perf_tbl_general['F1 Score @ 1 Prc'] = str(np.round(
            np.mean(f1_score_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                               str(np.round(np.std(f1_score_rusboost1[2:8]) * 100, 2)) + '%)', \
                                               str(np.round(np.mean(f1_score_lr1[2:8]) * 100, 2)) + '% (' + \
                                               str(np.round(np.std(f1_score_lr1[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['NDCG @ 1 Prc'] = str(np.round(
            np.mean(ndcg_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ndcg_rusboost1[2:8]) * 100, 2)) + '%)', \
                                           str(np.round(np.mean(ndcg_lr1[2:8]) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ndcg_lr1[2:8]) * 100, 2)) + '%)'

        perf_tbl_general['ECM @ 1 Prc'] = str(np.round(
            np.mean(ecm_rusboost1[2:8]) * 100, 2)) + '% (' + \
                                          str(np.round(np.std(ecm_rusboost1[2:8]) * 100, 2)) + '%)', \
                                          str(np.round(np.mean(ecm_lr1[2:8]) * 100, 2)) + '% (' + \
                                          str(np.round(np.std(ecm_lr1[2:8]) * 100, 2)) + '%)'

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + 'AppendixE_excluding_missing_values.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')

    def ratio_23_AppendixF(self, C_FN=30, C_FP=1):
        """
        This code is the same as ratio_analyse, but variables used here is the 23 raw accounting items that
        derive the 11 financial ratios in Dechow et al. (2011).

        Methodological choices:
            - 23 raw accounting items that derive the 11 financial ratios in Dechow et al. (2011)
            - 10-fold validation
            - Bertomeu et al. (2021)'s serial fraud treatment
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        variables_to_select = ['act', 'che', 'lct', 'dlc', 'txp', 'at', 'ivao',
                               'lt', 'dltt', 'ivst', 'pstk', 'rect', 'invt',
                               'ppegt', 'sale', 'cogs', 'ap', 'ib', 'sstk',
                               'dltis', 'ceq', 'prcc_f', 'csho']
        reduced_tbl_2 = fraud_df.loc[:, variables_to_select]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]  # 1991
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]  # 2010

        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -23:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        # redo cross-validation if you wish
        if cv_type == 'kfold':
            if cross_val == True:

                # optimise LogitBoost
                print('Grid search hyperparameter optimisation started for AdaBoost')
                t1 = datetime.now()

                best_perf_ada = 0

                base_lr = LogisticRegression(random_state=0, solver='newton-cg')
                estimators = list(range(10, 3001, 10))
                learning_rates = [x/1000 for x in range(10,1001,10)]
                pipe_ada = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_ada', AdaBoostClassifier(estimator=base_lr, random_state=0))])
                param_grid_ada = {'base_mdl_ada__n_estimators': estimators, \
                                  'base_mdl_ada__learning_rate': learning_rates}

                clf_ada = GridSearchCV(pipe_ada, param_grid_ada, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_ada.fit(X_CV, Y_CV)
                score_ada = clf_ada.best_score_
                if score_ada >= best_perf_ada:
                    best_perf_ada = score_ada
                    opt_params_ada = clf_ada.best_params_


                print('LogitBoost: The optimal number of estimators is ' + \
                      str(opt_params_ada['base_mdl_ada__n_estimators']) + ', and learning rate is ' + \
                      str(opt_params_ada['base_mdl_ada__learning_rate']) + ', and score is ' + \
                      str(score_ada))

                print('Computing CV ROC for LR ...')
                t1 = datetime.now()
                score_lr = []
                for m in range(0, k_fold):
                    train_sample, test_sample = train_test_split(Y_CV, test_size=1 /
                                                                                 k_fold, shuffle=False, random_state=m)
                    X_train = X_CV.iloc[train_sample.index]
                    X_train = add_constant(X_train)
                    Y_train = train_sample
                    X_test = X_CV.iloc[test_sample.index]
                    X_test = add_constant(X_test)
                    Y_test = test_sample

                    logit_model = Logit(Y_train, X_train)
                    logit_model = logit_model.fit(disp=0)
                    pred_LR_CV = logit_model.predict(X_test)
                    score_lr.append(roc_auc_score(Y_test, pred_LR_CV))

                score_lr = np.mean(score_lr)
                print('Logit: The optimal score is ' + str(score_lr))

                t2 = datetime.now()
                dt = t2 - t1
                print('LogitBoost CV finished after ' + str(dt.total_seconds()) + ' sec')


                # optimize SVM
                print('Grid search hyperparameter optimisation started for SVM')
                t1 = datetime.now()

                pipe_svm = Pipeline([('scale', StandardScaler()), \
                                     ('base_mdl_svm', SVC(shrinking=False, \
                                                          probability=False, random_state=0, max_iter=-1, \
                                                          tol=X_CV.shape[-1] * 1e-3))])
                C = [0.001, 0.01, 0.1, 1, 10, 100]
                gamma = [0.0001, 0.001, 0.01, 0.1]
                kernel = ['rbf', 'linear', 'poly']
                class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]
                param_grid_svm = {'base_mdl_svm__kernel': kernel,\
                                  'base_mdl_svm__C': C, \
                                  'base_mdl_svm__gamma': gamma,\
                                  'base_mdl_svm__class_weight':class_weight }


                clf_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_svm.fit(X_CV, Y_CV)
                opt_params_svm = clf_svm.best_params_
                gamma_opt = opt_params_svm['base_mdl_svm__gamma']
                cw_opt = opt_params_svm['base_mdl_svm__class_weight']
                c_opt = opt_params_svm['base_mdl_svm__C']
                kernel_opt = opt_params_svm['base_mdl_svm__kernel']
                score_svm = clf_svm.best_score_

                t2 = datetime.now()
                dt = t2 - t1
                print('SVM CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('SVM: The optimal C+/C- ratio is ' + str(cw_opt) + ', gamma is ' + str(
                    gamma_opt) + ', C is' + str(c_opt) + ',score is ' + str(score_svm) +\
                      ', kernel is ' + str(kernel_opt))


        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)

        roc_svm = np.zeros(len(range_oos))
        roc_svm_training = np.zeros(len(range_oos))
        sensitivity_OOS_svm1 = np.zeros(len(range_oos))
        sensitivity_svm1_training = np.zeros(len(range_oos))
        specificity_OOS_svm1 = np.zeros(len(range_oos))
        specificity_svm1_training = np.zeros(len(range_oos))
        precision_svm1 = np.zeros(len(range_oos))
        precision_svm1_training = np.zeros(len(range_oos))
        ndcg_svm1 = np.zeros(len(range_oos))
        ndcg_svm1_training = np.zeros(len(range_oos))
        ecm_svm1 = np.zeros(len(range_oos))
        ecm_svm1_training = np.zeros(len(range_oos))


        roc_lr = np.zeros(len(range_oos))
        roc_lr_training = np.zeros(len(range_oos))
        sensitivity_OOS_lr1 = np.zeros(len(range_oos))
        sensitivity_lr1_training = np.zeros(len(range_oos))
        specificity_OOS_lr1 = np.zeros(len(range_oos))
        specificity_lr1_training = np.zeros(len(range_oos))
        precision_lr1 = np.zeros(len(range_oos))
        precision_lr1_training = np.zeros(len(range_oos))
        ndcg_lr1 = np.zeros(len(range_oos))
        ndcg_lr1_training = np.zeros(len(range_oos))
        ecm_lr1 = np.zeros(len(range_oos))
        ecm_lr1_training = np.zeros(len(range_oos))


        roc_ada = np.zeros(len(range_oos))
        roc_ada_training = np.zeros(len(range_oos))
        sensitivity_OOS_ada1 = np.zeros(len(range_oos))
        sensitivity_ada1_training = np.zeros(len(range_oos))
        specificity_OOS_ada1 = np.zeros(len(range_oos))
        specificity_ada1_training = np.zeros(len(range_oos))
        precision_ada1 = np.zeros(len(range_oos))
        precision_ada1_training = np.zeros(len(range_oos))
        ndcg_ada1 = np.zeros(len(range_oos))
        ndcg_ada1_training = np.zeros(len(range_oos))
        ecm_ada1 = np.zeros(len(range_oos))
        ecm_ada1_training = np.zeros(len(range_oos))


        roc_fused = np.zeros(len(range_oos))
        roc_fused_training = np.zeros(len(range_oos))
        sensitivity_OOS_fused1 = np.zeros(len(range_oos))
        sensitivity_fused1_training = np.zeros(len(range_oos))
        specificity_OOS_fused1 = np.zeros(len(range_oos))
        specificity_fused1_training = np.zeros(len(range_oos))
        precision_fused1 = np.zeros(len(range_oos))
        precision_fused1_training = np.zeros(len(range_oos))
        ndcg_fused1 = np.zeros(len(range_oos))
        ndcg_fused1_training = np.zeros(len(range_oos))
        ecm_fused1 = np.zeros(len(range_oos))
        ecm_fused1_training = np.zeros(len(range_oos))


        m = 0
        for yr in range_oos:
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start
            else:
                year_start_IS = yr - IS_period
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]

            print(f'before dropping the number of observations is: {len(tbl_year_OOS)}')

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True


            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)
            print(f'after dropping the number of observations is: {len(tbl_year_OOS)}')

            X = tbl_year_IS.iloc[:, -23:]
            print(X.columns)
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            Y = tbl_year_IS.AAER_DUMMY

            X_OOS = tbl_year_OOS.iloc[:, -23:]
            X_OOS = (X_OOS - mean) / std
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            n_P_training = np.sum(Y == 1)
            n_N_training = np.sum(Y == 0)

            # SVM
            clf_svm = SVC(class_weight=opt_params_svm['base_mdl_svm__class_weight'],
                          kernel=opt_params_svm['base_mdl_svm__kernel'], \
                          C=opt_params_svm['base_mdl_svm__C'],\
                          gamma=opt_params_svm['base_mdl_svm__gamma'], shrinking=False, \
                          probability=False, random_state=0, max_iter=-1, \
                          tol=X.shape[-1] * 1e-3)

            clf_svm = clf_svm.fit(X, Y)

            #performance on training sample- svm
            pred_train_svm = clf_svm.decision_function(X)
            probs_fraud_svm = np.exp(pred_train_svm) / (1 + np.exp(pred_train_svm))
            cutoff_svm = np.percentile(probs_fraud_svm, 99)
            roc_svm_training[m] = roc_auc_score(Y, probs_fraud_svm)
            sensitivity_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                                            Y == 1)) / np.sum(Y)
            specificity_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm < cutoff_svm, \
                                                            Y == 0)) / np.sum(Y == 0)
            precision_svm1_training[m] = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                                      Y == 1)) / np.sum(probs_fraud_svm >= cutoff_svm)
            ndcg_svm1_training[m] = ndcg_k(Y, probs_fraud_svm, 99)


            FN_svm3 = np.sum(np.logical_and(probs_fraud_svm < cutoff_svm, \
                                            Y == 1))
            FP_svm3 = np.sum(np.logical_and(probs_fraud_svm >= cutoff_svm, \
                                            Y == 0))
            ecm_svm1_training[m] = C_FN * P_f * FN_svm3 / n_P_training + C_FP * P_nf * FP_svm3 / n_N_training


            #performance on testing sample- svm
            pred_test_svm = clf_svm.decision_function(X_OOS)
            probs_oos_fraud_svm = np.exp(pred_test_svm) / (1 + np.exp(pred_test_svm))

            roc_svm[m] = roc_auc_score(Y_OOS, probs_oos_fraud_svm)

            cutoff_OOS_svm = np.percentile(probs_oos_fraud_svm, 99)
            sensitivity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_svm1[m] = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_svm >= cutoff_OOS_svm)
            ndcg_svm1[m] = ndcg_k(Y_OOS, probs_oos_fraud_svm, 99)

            FN_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm < cutoff_OOS_svm, \
                                            Y_OOS == 1))
            FP_svm2 = np.sum(np.logical_and(probs_oos_fraud_svm >= cutoff_OOS_svm, \
                                            Y_OOS == 0))

            ecm_svm1[m] = C_FN * P_f * FN_svm2 / n_P + C_FP * P_nf * FP_svm2 / n_N

            # Logistic Regression – Dechow et al (2011)
            X_lr = add_constant(X)
            X_OOS_lr = add_constant(X_OOS)
            clf_lr = Logit(Y, X_lr)
            clf_lr = clf_lr.fit(disp=0)
            probs_oos_fraud_lr = clf_lr.predict(X_OOS_lr)

            #performance on training sample- logit
            probs_fraud_lr = clf_lr.predict(X_lr)
            cutoff_lr = np.percentile(probs_fraud_lr, 99)
            roc_lr_training[m] = roc_auc_score(Y, probs_fraud_lr)
            sensitivity_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                                           Y == 1)) / np.sum(Y)
            specificity_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr < cutoff_lr, \
                                                           Y == 0)) / np.sum(Y == 0)
            precision_lr1_training[m] = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                                     Y == 1)) / np.sum(probs_fraud_lr >= cutoff_lr)
            ndcg_lr1_training[m] = ndcg_k(Y, probs_fraud_lr, 99)
            FN_lr3 = np.sum(np.logical_and(probs_fraud_lr < cutoff_lr, \
                                           Y == 1))
            FP_lr3 = np.sum(np.logical_and(probs_fraud_lr >= cutoff_lr, \
                                           Y == 0))
            ecm_lr1_training[m] = C_FN * P_f * FN_lr3 / n_P_training + C_FP * P_nf * FP_lr3 / n_N_training


            #performance on testing sample- logit
            roc_lr[m] = roc_auc_score(Y_OOS, probs_oos_fraud_lr)
            cutoff_OOS_lr = np.percentile(probs_oos_fraud_lr, 99)
            sensitivity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                           Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                                           Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_lr1[m] = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                                     Y_OOS == 1)) / np.sum(probs_oos_fraud_lr >= cutoff_OOS_lr)
            ndcg_lr1[m] = ndcg_k(Y_OOS, probs_oos_fraud_lr, 99)
            FN_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr < cutoff_OOS_lr, \
                                           Y_OOS == 1))
            FP_lr2 = np.sum(np.logical_and(probs_oos_fraud_lr >= cutoff_OOS_lr, \
                                           Y_OOS == 0))
            ecm_lr1[m] = C_FN * P_f * FN_lr2 / n_P + C_FP * P_nf * FP_lr2 / n_N


            # LogitBoost
            base_lr = LogisticRegression(random_state=0, solver='newton-cg')

            clf_ada = AdaBoostClassifier(n_estimators=opt_params_ada['base_mdl_ada__n_estimators'], \
                                         learning_rate=opt_params_ada['base_mdl_ada__learning_rate'], \
                                         estimator=base_lr, random_state=0)
            clf_ada = clf_ada.fit(X, Y)
            probs_oos_fraud_ada = clf_ada.predict_proba(X_OOS)[:, -1]

            labels_ada = clf_ada.predict(X_OOS)

            # performance on training sample- LogitBoost
            probs_fraud_ada = clf_ada.predict_proba(X)[:, -1]
            cutoff_ada = np.percentile(probs_fraud_ada, 99)
            roc_ada_training[m] = roc_auc_score(Y, probs_fraud_ada)
            sensitivity_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                                            Y == 1)) / np.sum(Y)
            specificity_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada < cutoff_ada, \
                                                            Y == 0)) / np.sum(Y == 0)
            precision_ada1_training[m] = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                                      Y == 1)) / np.sum(probs_fraud_ada >= cutoff_ada)
            ndcg_ada1_training[m] = ndcg_k(Y, probs_fraud_ada, 99)
            FN_ada3 = np.sum(np.logical_and(probs_fraud_ada < cutoff_ada, \
                                            Y == 1))
            FP_ada3 = np.sum(np.logical_and(probs_fraud_ada >= cutoff_ada, \
                                            Y == 0))
            ecm_ada1_training[m] = C_FN * P_f * FN_ada3 / n_P_training + C_FP * P_nf * FP_ada3 / n_N_training

            # performance on testing sample- LogitBoost
            roc_ada[m] = roc_auc_score(Y_OOS, probs_oos_fraud_ada)
            cutoff_OOS_ada = np.percentile(probs_oos_fraud_ada, 99)
            sensitivity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_ada1[m] = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_ada >= cutoff_OOS_ada)
            ndcg_ada1[m] = ndcg_k(Y_OOS, probs_oos_fraud_ada, 99)
            FN_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada < cutoff_OOS_ada, \
                                            Y_OOS == 1))
            FP_ada2 = np.sum(np.logical_and(probs_oos_fraud_ada >= cutoff_OOS_ada, \
                                            Y_OOS == 0))
            ecm_ada1[m] = C_FN * P_f * FN_ada2 / n_P + C_FP * P_nf * FP_ada2 / n_N


            # Fused approach
            weight_ser = np.array([score_svm, score_lr, score_ada])
            weight_ser = weight_ser / np.sum(weight_ser)

            # performance on training sample- Fused
            probs_fraud_svm_fused = (1 + np.exp(-1 * probs_fraud_svm)) ** -1

            probs_fraud_lr_fused = (1 + np.exp(-1 * probs_fraud_lr)) ** -1

            probs_fraud_ada_fused = (1 + np.exp(-1 * probs_fraud_ada)) ** -1
            clf_fused_training = np.dot(np.array([probs_fraud_svm_fused, \
                                         probs_fraud_lr_fused, probs_fraud_ada_fused]).T, weight_ser)

            probs_fraud_fused = clf_fused_training
            cutoff_fused = np.percentile(probs_fraud_fused, 99)
            roc_fused_training[m] = roc_auc_score(Y, probs_fraud_fused)
            sensitivity_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                                              Y == 1)) / np.sum(Y)
            specificity_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused < cutoff_fused, \
                                                              Y == 0)) / np.sum(Y == 0)
            precision_fused1_training[m] = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                                        Y == 1)) / np.sum(probs_fraud_fused >= cutoff_fused)
            ndcg_fused1_training[m] = ndcg_k(Y, probs_fraud_fused, 99)
            FN_fused3 = np.sum(np.logical_and(probs_fraud_fused < cutoff_fused, \
                                              Y == 1))
            FP_fused3 = np.sum(np.logical_and(probs_fraud_fused >= cutoff_fused, \
                                              Y == 0))

            ecm_fused1_training[m] = C_FN * P_f * FN_fused3 / n_P_training + C_FP * P_nf * FP_fused3 / n_N_training

            # performance on testing sample- Fused
            probs_oos_fraud_svm = (1 + np.exp(-1 * probs_oos_fraud_svm)) ** -1
            probs_oos_fraud_lr = (1 + np.exp(-1 * probs_oos_fraud_lr)) ** -1
            probs_oos_fraud_ada = (1 + np.exp(-1 * probs_oos_fraud_ada)) ** -1
            clf_fused = np.dot(np.array([probs_oos_fraud_svm, \
                                         probs_oos_fraud_lr, probs_oos_fraud_ada]).T, weight_ser)
            probs_oos_fraud_fused = clf_fused
            roc_fused[m] = roc_auc_score(Y_OOS, probs_oos_fraud_fused)
            cutoff_OOS_fused = np.percentile(probs_oos_fraud_fused, 99)
            sensitivity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                              Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                                              Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_fused1[m] = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                                        Y_OOS == 1)) / np.sum(probs_oos_fraud_fused >= cutoff_OOS_fused)
            ndcg_fused1[m] = ndcg_k(Y_OOS, probs_oos_fraud_fused, 99)
            FN_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused < cutoff_OOS_fused, \
                                              Y_OOS == 1))
            FP_fused2 = np.sum(np.logical_and(probs_oos_fraud_fused >= cutoff_OOS_fused, \
                                              Y_OOS == 0))

            ecm_fused1[m] = C_FN * P_f * FN_fused2 / n_P + C_FP * P_nf * FP_fused2 / n_N

            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1


        f1_score_svm1_training = 2 * (precision_svm1_training * sensitivity_svm1_training) / \
                        (precision_svm1_training + sensitivity_svm1_training + 1e-8)

        f1_score_lr1_training = 2 * (precision_lr1_training * sensitivity_lr1_training) / \
                       (precision_lr1_training + sensitivity_lr1_training + 1e-8)

        f1_score_ada1_training = 2 * (precision_ada1_training * sensitivity_ada1_training) / \
                        (precision_ada1_training + sensitivity_ada1_training + 1e-8)

        f1_score_fused1_training = 2 * (precision_fused1_training * sensitivity_fused1_training) / \
                          (precision_fused1_training + sensitivity_fused1_training + 1e-8)


        f1_score_svm1 = 2 * (precision_svm1 * sensitivity_OOS_svm1) / \
                        (precision_svm1 + sensitivity_OOS_svm1 + 1e-8)

        f1_score_lr1 = 2 * (precision_lr1 * sensitivity_OOS_lr1) / \
                       (precision_lr1 + sensitivity_OOS_lr1 + 1e-8)

        f1_score_ada1 = 2 * (precision_ada1 * sensitivity_OOS_ada1) / \
                        (precision_ada1 + sensitivity_OOS_ada1 + 1e-8)

        f1_score_fused1 = 2 * (precision_fused1 * sensitivity_OOS_fused1) / \
                          (precision_fused1 + sensitivity_OOS_fused1 + 1e-8)

        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['SVM', 'LR', 'LogitBoost', 'FUSED']

        perf_tbl_general['Training Roc'] = [str(np.round(
            np.mean(roc_svm_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm_training) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr_training) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada_training) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Roc'] = [str(np.round(
            np.mean(roc_svm) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_svm) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_lr) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_lr) * 100, 2)) + '%)', str(np.round(
            np.mean(roc_ada) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_ada) * 100, 2)) + '%)',
                                   str(np.round(
                                       np.mean(roc_fused) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_fused) * 100, 2)) + '%)']

        gap_roc_svm = roc_svm - roc_svm_training
        gap_roc_lr = roc_lr - roc_lr_training
        gap_roc_ada = roc_ada - roc_ada_training
        gap_roc_fused = roc_fused - roc_fused_training

        mean_gap_roc_svm = np.round(np.mean(gap_roc_svm) * 100, 2)
        mean_gap_roc_lr = np.round(np.mean(gap_roc_lr) * 100, 2)
        mean_gap_roc_ada = np.round(np.mean(gap_roc_ada) * 100, 2)
        mean_gap_roc_fused = np.round(np.mean(gap_roc_fused) * 100, 2)

        perf_tbl_general['Gap Roc'] = [str(mean_gap_roc_svm) + '%', str(mean_gap_roc_lr) + '%', str(mean_gap_roc_ada) + '%', str(mean_gap_roc_fused) + '%']

        perf_tbl_general['Training Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_svm1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_svm1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_lr1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_lr1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_ada1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_ada1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_fused1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(sensitivity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_fused1) * 100, 2)) + '%)']

        gap_sensitivity_svm = sensitivity_OOS_svm1 - sensitivity_svm1_training
        gap_sensitivity_lr = sensitivity_OOS_lr1 - sensitivity_lr1_training
        gap_sensitivity_ada = sensitivity_OOS_ada1 - sensitivity_ada1_training
        gap_sensitivity_fused = sensitivity_OOS_fused1 - sensitivity_fused1_training


        mean_gap_sensitivity_svm = np.round(np.mean(gap_sensitivity_svm) * 100, 2)
        mean_gap_sensitivity_lr = np.round(np.mean(gap_sensitivity_lr) * 100, 2)
        mean_gap_sensitivity_ada = np.round(np.mean(gap_sensitivity_ada) * 100, 2)
        mean_gap_sensitivity_fused = np.round(np.mean(gap_sensitivity_fused) * 100, 2)

        perf_tbl_general['Gap Sensitivity'] = [str(mean_gap_sensitivity_svm) + '%',\
                                               str(mean_gap_sensitivity_lr) + '%', \
                                               str(mean_gap_sensitivity_ada) + '%',\
                                       str(mean_gap_sensitivity_fused) + '%']

        perf_tbl_general['Training Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_svm1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_svm1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_lr1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_lr1_training) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_ada1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_ada1_training) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_fused1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_fused1_training) * 100, 2)) + '%)']


        perf_tbl_general['Testing Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_svm1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_svm1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_lr1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_lr1) * 100, 2)) + '%)',
                                                   str(np.round(
                                                       np.mean(specificity_OOS_ada1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_ada1) * 100, 2)) + '%)',
                                                   str(np.round(np.mean(specificity_OOS_fused1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_fused1) * 100, 2)) + '%)']


        gap_specificity_svm = specificity_OOS_svm1 - specificity_svm1_training
        gap_specificity_lr = specificity_OOS_lr1 - specificity_lr1_training
        gap_specificity_ada = specificity_OOS_ada1 - specificity_ada1_training
        gap_specificity_fused = specificity_OOS_fused1 - specificity_fused1_training

        mean_gap_specificity_svm = np.round(np.mean(gap_specificity_svm) * 100, 2)
        mean_gap_specificity_lr = np.round(np.mean(gap_specificity_lr) * 100, 2)
        mean_gap_specificity_ada = np.round(np.mean(gap_specificity_ada) * 100, 2)
        mean_gap_specificity_fused = np.round(np.mean(gap_specificity_fused) * 100, 2)

        perf_tbl_general['Gap Specificity'] = [str(mean_gap_specificity_svm) + '%', str(mean_gap_specificity_lr) + '%', str(mean_gap_specificity_ada) + '%',\
                                       str(mean_gap_specificity_fused) + '%']


        perf_tbl_general['Training Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1_training) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_svm1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_lr1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(precision_ada1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_ada1) * 100, 2)) + '%)',
                                                 str(np.round(
                                                     np.mean(precision_fused1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_fused1) * 100, 2)) + '%)']

        gap_precision_svm = precision_svm1 - precision_svm1_training
        gap_precision_lr = precision_lr1 - precision_lr1_training
        gap_precision_ada = precision_ada1 - precision_ada1_training
        gap_precision_fused = precision_fused1 - precision_fused1_training


        mean_gap_precision_svm = np.round(np.mean(gap_precision_svm) * 100, 2)
        mean_gap_precision_lr = np.round(np.mean(gap_precision_lr) * 100, 2)
        mean_gap_precision_ada = np.round(np.mean(gap_precision_ada) * 100, 2)
        mean_gap_precision_fused = np.round(np.mean(gap_precision_fused) * 100, 2)

        perf_tbl_general['Gap Precision'] = [str(mean_gap_precision_svm) + '%', str(mean_gap_precision_lr) + '%', str(mean_gap_precision_ada) + '%',\
                                       str(mean_gap_precision_fused) + '%']


        perf_tbl_general['Training F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1_training) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_svm1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_lr1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(f1_score_ada1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_ada1) * 100, 2)) + '%)',
                                                str(np.round(
                                                    np.mean(f1_score_fused1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_fused1) * 100, 2)) + '%)']


        gap_f1_score_svm = f1_score_svm1 - f1_score_svm1_training
        gap_f1_score_lr = f1_score_lr1 - f1_score_lr1_training
        gap_f1_score_ada = f1_score_ada1 - f1_score_ada1_training
        gap_f1_score_fused = f1_score_fused1 - f1_score_fused1_training


        mean_gap_f1_score_svm = np.round(np.mean(gap_f1_score_svm) * 100, 2)
        mean_gap_f1_score_lr = np.round(np.mean(gap_f1_score_lr) * 100, 2)
        mean_gap_f1_score_ada = np.round(np.mean(gap_f1_score_ada) * 100, 2)
        mean_gap_f1_score_fused = np.round(np.mean(gap_f1_score_fused) * 100, 2)

        perf_tbl_general['Gap F1 Score'] = [str(mean_gap_f1_score_svm) + '%', str(mean_gap_f1_score_lr) + '%', str(mean_gap_f1_score_ada) + '%',\
                                       str(mean_gap_f1_score_fused) + '%']

        perf_tbl_general['Training NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1_training) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_svm1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_lr1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ndcg_ada1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_ada1) * 100, 2)) + '%)',
                                            str(np.round(
                                                np.mean(ndcg_fused1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_fused1) * 100, 2)) + '%)']

        gap_ndcg_svm = ndcg_svm1 - ndcg_svm1_training
        gap_ndcg_lr = ndcg_lr1 - ndcg_lr1_training
        gap_ndcg_ada = ndcg_ada1 - ndcg_ada1_training
        gap_ndcg_fused = ndcg_fused1 - ndcg_fused1_training

        mean_gap_ndcg_svm = np.round(np.mean(gap_ndcg_svm) * 100, 2)
        mean_gap_ndcg_lr = np.round(np.mean(gap_ndcg_lr) * 100, 2)
        mean_gap_ndcg_ada = np.round(np.mean(gap_ndcg_ada) * 100, 2)
        mean_gap_ndcg_fused = np.round(np.mean(gap_ndcg_fused) * 100, 2)

        perf_tbl_general['Gap NDCG'] = [str(mean_gap_ndcg_svm) + '%', str(mean_gap_ndcg_lr) + '%', str(mean_gap_ndcg_ada) + '%',\
                                       str(mean_gap_ndcg_fused) + '%']

        perf_tbl_general['Training ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1_training) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1_training) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_svm1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_svm1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_lr1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_lr1) * 100, 2)) + '%)', str(np.round(
            np.mean(ecm_ada1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_ada1) * 100, 2)) + '%)',
                                           str(np.round(
                                               np.mean(ecm_fused1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_fused1) * 100, 2)) + '%)']

        gap_ecm_svm = ecm_svm1 - ecm_svm1_training
        gap_ecm_lr = ecm_lr1 - ecm_lr1_training
        gap_ecm_ada = ecm_ada1 - ecm_ada1_training
        gap_ecm_fused = ecm_fused1 - ecm_fused1_training

        mean_gap_ecm_svm = np.round(np.mean(gap_ecm_svm) * 100, 2)
        mean_gap_ecm_lr = np.round(np.mean(gap_ecm_lr) * 100, 2)
        mean_gap_ecm_ada = np.round(np.mean(gap_ecm_ada) * 100, 2)
        mean_gap_ecm_fused = np.round(np.mean(gap_ecm_fused) * 100, 2)

        perf_tbl_general['Gap ECM'] = [str(mean_gap_ecm_svm) + '%', str(mean_gap_ecm_lr) + '%', str(mean_gap_ecm_ada) + '%',\
                                       str(mean_gap_ecm_fused) + '%']


        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_ratio_23_with_gap_AppendixF.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')



    def RUS_23_AppendixF(self, C_FN=30, C_FP=1):
        """
        This code uses RUSBoost model of Bao et al. (2020) with the 23 raw accounting items that
        derive the 11 financial ratios in Dechow et al. (2011). Designed for Appendix F.
        """

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = datetime.now()
        IS_period=self.ip
        k_fold = self.cv_k
        OOS_period = self.op
        OOS_gap = self.og
        start_OOS_year = self.ts[0]
        end_OOS_year = self.ts[-1]
        sample_start = self.ss
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        variables_to_select = ['act', 'che', 'lct', 'dlc', 'txp', 'at', 'ivao',
                               'lt', 'dltt', 'ivst', 'pstk', 'rect', 'invt',
                               'ppegt', 'sale', 'cogs', 'ap', 'ib', 'sstk',
                               'dltis', 'ceq', 'prcc_f', 'csho']
        reduced_tbl_2 = fraud_df.loc[:, variables_to_select]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl.reset_index(drop=True)

        range_oos = range(start_OOS_year, end_OOS_year + 1)
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= sample_start)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -23:]
        X_CV = X_CV.apply(lambda x: x/np.linalg.norm(x))
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        # optimize RUSBoost number of estimators
        if cv_type == 'kfold':
            if cross_val == True:

                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1 = datetime.now()
                n_estimators = list(range(10, 3001, 10))
                learning_rate = list(x / 1000 for x in range(10, 1001, 10))

                base_tree = DecisionTreeClassifier(min_samples_leaf=5)
                bao_RUSboost = RUSBoostClassifier(estimator=base_tree, \
                                                  sampling_strategy=1, random_state=0)
                param_grid_rusboost = {'base_rusboost__n_estimators': n_estimators,
                                       'base_rusboost__learning_rate': learning_rate}

                clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)
                clf_rus.fit(X_CV, Y_CV)
                opt_params_rus = clf_rus.best_params_
                n_opt_rus = opt_params_rus['base_rusboost__n_estimators']
                r_opt_rus = opt_params_rus['base_rusboost__learning_rate']
                score_rus = clf_rus.best_score_
                t2 = datetime.now()
                dt = t2 - t1
                print('RUSBoost CV finished after ' + str(dt.total_seconds()) + ' sec')
                print('RUSBoost: The optimal number of estimators is ' + str(n_opt_rus) + 'learning rate is ' + str(
                    r_opt_rus))

        roc_rusboost = np.zeros(len(range_oos))
        roc_rusboost_training = np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1 = np.zeros(len(range_oos))
        sensitivity_rusboost1_training = np.zeros(len(range_oos))
        specificity_OOS_rusboost1 = np.zeros(len(range_oos))
        specificity_rusboost1_training = np.zeros(len(range_oos))
        precision_rusboost1 = np.zeros(len(range_oos))
        precision_rusboost1_training = np.zeros(len(range_oos))
        ndcg_rusboost1 = np.zeros(len(range_oos))
        ndcg_rusboost1_training = np.zeros(len(range_oos))
        ecm_rusboost1 = np.zeros(len(range_oos))
        ecm_rusboost1_training = np.zeros(len(range_oos))

        m = 0

        for yr in range_oos:
            t1 = datetime.now()

            year_start_IS = sample_start
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)
            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            tbl_year_OOS = reduced_tbl.loc[reduced_tbl.fyear == yr]

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True

            else:
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)

            X = tbl_year_IS.iloc[:, -23:]
            X_rus = X.apply(lambda x: x / np.linalg.norm(x))
            Y = tbl_year_IS.AAER_DUMMY
            X_OOS = tbl_year_OOS.iloc[:, -23:]
            X_OOS_rus = X_OOS.apply(lambda x: x / np.linalg.norm(x))
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            n_P_training = np.sum(Y == 1)
            n_N_training = np.sum(Y == 0)


            base_tree = DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost = RUSBoostClassifier(estimator=base_tree, n_estimators=n_opt_rus, \
                                              learning_rate=r_opt_rus, sampling_strategy=1, random_state=0)
            clf_rusboost = bao_RUSboost.fit(X_rus, Y)

            # performance on training sample- rusboost
            probs_fraud_rusboost = clf_rusboost.predict_proba(X_rus)[:, -1]
            roc_rusboost_training[m] = roc_auc_score(Y, probs_fraud_rusboost)
            cutoff_rusboost = np.percentile(probs_fraud_rusboost, 99)
            sensitivity_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                                      Y == 1)) / np.sum(Y)
            specificity_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost < cutoff_rusboost, \
                                                                      Y == 0)) / np.sum(Y == 0)
            precision_rusboost1_training[m] = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                                    Y == 1)) / np.sum(
                probs_fraud_rusboost >= cutoff_rusboost)
            ndcg_rusboost1_training[m] = ndcg_k(Y, probs_fraud_rusboost, 99)

            FN_rusboost3 = np.sum(np.logical_and(probs_fraud_rusboost < cutoff_rusboost, \
                                                 Y == 1))
            FP_rusboost3 = np.sum(np.logical_and(probs_fraud_rusboost >= cutoff_rusboost, \
                                                 Y == 0))

            ecm_rusboost1_training[m] = C_FN * P_f * FN_rusboost3 / n_P_training + C_FP * P_nf * FP_rusboost3 / n_N_training

            # performance on testing sample- rusboost
            probs_oos_fraud_rusboost = clf_rusboost.predict_proba(X_OOS_rus)[:, -1]

            labels_rusboost = clf_rusboost.predict(X_OOS_rus)

            roc_rusboost[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rusboost)

            cutoff_OOS_rusboost = np.percentile(probs_oos_fraud_rusboost, 99)
            sensitivity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                                 Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                                 Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rusboost1[m] = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                           Y_OOS == 1)) / np.sum(
                probs_oos_fraud_rusboost >= cutoff_OOS_rusboost)
            ndcg_rusboost1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rusboost, 99)
            FN_rusboost2 = np.sum(np.logical_and(probs_oos_fraud_rusboost < cutoff_OOS_rusboost, \
                                                 Y_OOS == 1))
            FP_rusboost2 = np.sum(np.logical_and(probs_oos_fraud_rusboost >= cutoff_OOS_rusboost, \
                                                 Y_OOS == 0))

            ecm_rusboost1[m] = C_FN * P_f * FN_rusboost2 / n_P + C_FP * P_nf * FP_rusboost2 / n_N


            t2 = datetime.now()
            dt = t2 - t1
            print('analysis finished for OOS period ' + str(yr) + ' after ' + str(dt.total_seconds()) + ' sec')
            m += 1


        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['RUSBoost-23']


        perf_tbl_general['Training Roc'] = [str(np.round(
            np.mean(roc_rusboost_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(roc_rusboost_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Roc'] = [str(np.round(
            np.mean(roc_rusboost) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(roc_rusboost) * 100, 2)) + '%)']

        gap_roc_rusboost = roc_rusboost - roc_rusboost_training
        mean_gap_roc_rusboost = np.round(np.mean(gap_roc_rusboost) * 100, 2)
        perf_tbl_general['Gap Roc'] = [str(mean_gap_roc_rusboost) + '%']

        perf_tbl_general['Training Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_rusboost1_training) * 100, 2)) + '% (' + \
                                                            str(np.round(np.std(sensitivity_rusboost1_training) * 100,
                                                                         2)) + '%)']

        perf_tbl_general['Testing Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                           str(np.round(np.std(sensitivity_OOS_rusboost1) * 100,
                                                                        2)) + '%)']

        gap_sensitivity_rusboost = sensitivity_OOS_rusboost1 - sensitivity_rusboost1_training
        mean_gap_sensitivity_rusboost = np.round(np.mean(gap_sensitivity_rusboost) * 100, 2)
        perf_tbl_general['Gap Sensitivity'] = [str(mean_gap_sensitivity_rusboost) + '%']

        perf_tbl_general[('Training Specificity @ 1 Prc')] = [str(np.round(
            np.mean(specificity_rusboost1_training) * 100, 2)) + '% (' + \
                                                              str(np.round(np.std(specificity_rusboost1_training) * 100,
                                                                           2)) + '%)']

        perf_tbl_general['Testing Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_rusboost1) * 100, 2)) + '% (' + \
                                                           str(np.round(np.std(specificity_OOS_rusboost1) * 100,
                                                                        2)) + '%)']

        gap_specificity_rusboost = specificity_OOS_rusboost1 - specificity_rusboost1_training
        mean_gap_specificity_rusboost = np.round(np.mean(gap_specificity_rusboost) * 100, 2)
        perf_tbl_general['Gap Specificity'] = [str(mean_gap_specificity_rusboost) + '%']

        perf_tbl_general['Training Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rusboost1_training) * 100, 2)) + '% (' + \
                                                          str(np.round(np.std(precision_rusboost1_training) * 100,
                                                                       2)) + '%)']

        perf_tbl_general['Testing Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rusboost1) * 100, 2)) + '% (' + \
                                                         str(np.round(np.std(precision_rusboost1) * 100, 2)) + '%)']

        gap_precision_rusboost = precision_rusboost1 - precision_rusboost1_training
        mean_gap_precision_rusboost = np.round(np.mean(gap_precision_rusboost) * 100, 2)
        perf_tbl_general['Gap Precision'] = [str(mean_gap_precision_rusboost) + '%']

        f1_score_rusboost1_training = 2 * (precision_rusboost1_training * sensitivity_rusboost1_training) / \
                                      (precision_rusboost1_training + sensitivity_rusboost1_training + 1e-8)
        f1_score_rusboost1 = 2 * (precision_rusboost1 * sensitivity_OOS_rusboost1) / \
                             (precision_rusboost1 + sensitivity_OOS_rusboost1 + 1e-8)


        perf_tbl_general['Training F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rusboost1_training) * 100, 2)) + '% (' + \
                                                         str(np.round(np.std(f1_score_rusboost1_training) * 100,
                                                                      2)) + '%)']

        perf_tbl_general['Testing F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rusboost1) * 100, 2)) + '% (' + \
                                                        str(np.round(np.std(f1_score_rusboost1) * 100, 2)) + '%)']

        gap_f1_score_rusboost = f1_score_rusboost1 - f1_score_rusboost1_training
        mean_gap_f1_score_rusboost = np.round(np.mean(gap_f1_score_rusboost) * 100, 2)
        perf_tbl_general['Gap F1 Score'] = [str(mean_gap_f1_score_rusboost) + '%']

        perf_tbl_general['Training NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rusboost1_training) * 100, 2)) + '% (' + \
                                                     str(np.round(np.std(ndcg_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rusboost1) * 100, 2)) + '% (' + \
                                                    str(np.round(np.std(ndcg_rusboost1) * 100, 2)) + '%)']

        gap_ndcg_rusboost = ndcg_rusboost1 - ndcg_rusboost1_training
        mean_gap_ndcg_rusboost = np.round(np.mean(gap_ndcg_rusboost) * 100, 2)
        perf_tbl_general['Gap NDCG'] = [str(mean_gap_ndcg_rusboost) + '%']

        perf_tbl_general['Training ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rusboost1_training) * 100, 2)) + '% (' + \
                                                    str(np.round(np.std(ecm_rusboost1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rusboost1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(ecm_rusboost1) * 100, 2)) + '%)']

        gap_ecm_rusboost = ecm_rusboost1 - ecm_rusboost1_training
        mean_gap_ecm_rusboost = np.round(np.mean(gap_ecm_rusboost) * 100, 2)
        perf_tbl_general['Gap ECM'] = [str(mean_gap_ecm_rusboost) + '%']

        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_RUS_23_with_gap.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


a = ML_Fraud(sample_start = 1991,test_sample = range (2001,2011),OOS_per = 1,OOS_gap = 0,sampling = "expanding",adjust_serial = True,
            cv_flag = True,cv_k = 10,write = True,IS_per = 10)
a.raw_analyse()