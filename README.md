# ML_FruadDetection
This repository contains the codes to replicate the results of the paper entitled "A Machine Learning Approach to Detect Accounting Fraud".


Fraud_index_2001-2018:

Our findings of the paper reveal a significant negative association between the fraud index, derived from LogitBoost probabilities, and firm valuation. It is provided in this repository for future research.


Datasets:

1) FraudDB2020.csv:
This is the main dataset after excluding all missing values for both raw accounting numbers and financial ratios, used for all tests except for Panel A of Appendix E.

2) FraudDB2020_including_missing_values.csv:
This is the dataset including all missing values for both raw accounting numbers and financial ratios, used only for Panel A of Appendix E.


Replication of Tables:

1) Table 5. Performance of fraud detection models:
   Use the modules ratio_analyse, RUS_28, FK_23 (make sure you've run FK23_pickle beforehand to generate the financial kernel for SVMFK-23, you need to do it only once).

2) Table 6 and 7. The effects of collinearity among fraud predictors and Fitting test for each set of predictors:
   Use the modules ratio_analyse, raw_analyse, RUS_11, RUS_28.

3) Table 8. The effects of the validation approach used for hyperparameter tuning:
   Use the modules ratio_analyse, ratio_temporal, RUS_28, RUS28_temporal, FK_23, FK23_temporal.

4) Table 9. The effects of the approach used for serial fraud treatment:
   Use the modules ratio_analyse, ratio_biased, RUS_28, RUS28_biased, FK_23, FK23_biased.

5) Table 10. Performance of fraud detection models to predict fraud:
   Use the module analyse_forward.

6) Table 11. The effect of fraud index on firm performance:
   Use the do file Table 11_Firm_Valuation.do.

7) Appendix B. Additional evidence on the effectiveness of LogitBoost:
   Use the module compare_ada.

8) Appendix C. The number of firm-year observations before and after serial frauds treatment:
   Use the module ratio_analyse. You can check the number of drops per year in print.

9) Appendix D. Performance of ML models to detect fraud over the period 2003-2008:
   Use the modules ratio_2003, RUS28_2003, FK23_2003.

10) Appendix E: Additional performance of RUSBoost and Logit Models:
    For Panel B, simply use the module AppendixE_excluding_missing_values.
    For Panel A, change the dataset used from "FraudDB2020.csv" to "FraudDB2020_including_missing_values.csv" and set OOS_gap = 1 to set up a two-year gap.

11) Appendix F. Replicating the main results using 23 raw accounting numbers:
    Use the modules ratio_analyse, ratio_23_AppendixF, RUS_11, RUS_23_AppendixF.



   
