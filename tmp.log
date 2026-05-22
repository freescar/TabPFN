$python tabpfn_simple_classification_bw09_high_conf_plus.py > tmp.log

GPU: NVIDIA L20
OUTPUT_DIR=./results/two_routes_calibrated_high_conf
DATA_PATH=/ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/
split train_ratio=0.8, val_ratio=0.95
reference_slot_ids=[2, 3, 4, 5, 12, 13, 20, 21, 22, 23]
use_residual_compensation=True
include_slot_as_feature=True
target_hc_acc=95.0, max_hc_severe=0.0, min_cal_hc_n=10
Selected routes: C_loop_count, F_delta_run_trend
Rule selection: scan on CAL, apply fixed selected rule to TEST
Found 36 file(s).

[1/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB01_CHA1_1011_1229.parquet
loaded shape=(3053, 882)

==================================================================================================================================
Dataset: EPLBAB01_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(3053, 883), sort_time=0.012s
label out-of-range run_value=9/3053, policy=clip
split: train=[0,2442), cal=[2442,2900), test=[2900,3053)
split sizes: train=2442, cal=458, test=153

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           74   3.030303          4  0.873362           0  0.000000
    3          375  15.356265         64 13.973799           9  5.882353
    4         1143  46.805897        228 49.781659          53 34.640523
    5          685  28.050778        150 32.751092          75 49.019608
    6          145   5.937756         12  2.620087          16 10.457516
    7           20   0.819001          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.088726
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   3          2.250294        1.861619      -0.045642       3.963217       4.008859        2.250294                 False
       6 188          0.856054        0.803988      -0.932841       3.037609       3.970450        0.856054                 False
       8 217          0.653465        0.569718      -1.706532       2.946407       4.652939        0.653465                 False
       9   3          0.871891        1.434749       0.341377       2.246692       1.905314        0.871891                 False
      10 189          1.261143        1.130194      -1.369106       3.413761       4.782867        1.261143                 False
      14 188         -0.434623       -0.548434      -2.121087       1.377360       3.498447       -0.434623                 False
      15   1          8.119522        8.119522       8.119522       8.119522       0.000000       -0.088726                  True
      16 215         -0.494835       -0.250148      -2.587283       1.989338       4.576621       -0.494835                 False
      17   2          0.469096        0.469096       0.215578       0.722615       0.507037       -0.088726                  True
      18 190         -1.135084       -1.081823      -2.929996       0.958911       3.888906       -1.135084                 False
      24 217         -1.443253       -1.390762      -4.256424       1.003792       5.260216       -1.443253                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=879 -> miss=874 -> var=721 -> final=120
  C_loop_count: fit_time=2.86s pred_time=2.43s

  [CAL FULL C_loop_count]
    n=266 | Acc=61.65% | Within1=99.25% | Severe(|d|>=2)=0.75% | MAE=0.3910 | RMSE=0.6372 | Penalty=0.4060 | MeanDiff=0.0902

  [TEST FULL C_loop_count]
    n=89 | Acc=56.18% | Within1=97.75% | Severe(|d|>=2)=2.25% | MAE=0.4607 | RMSE=0.7111 | Penalty=0.5056 | MeanDiff=0.1910

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=909 -> miss=904 -> var=751 -> final=120
  F_delta_run_trend: fit_time=2.27s pred_time=1.85s

  [CAL FULL F_delta_run_trend]
    n=265 | Acc=65.28% | Within1=99.25% | Severe(|d|>=2)=0.75% | MAE=0.3547 | RMSE=0.6081 | Penalty=0.3698 | MeanDiff=0.0075

  [TEST FULL F_delta_run_trend]
    n=88 | Acc=67.05% | Within1=98.86% | Severe(|d|>=2)=1.14% | MAE=0.3409 | RMSE=0.6030 | Penalty=0.3636 | MeanDiff=0.0227

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         265     65.283019         0.754717           88      67.045455          98.863636          1.136364           0.363636              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         266     61.654135         0.751880           89      56.179775          97.752809          2.247191           0.505618              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000             0.000000               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000             0.000000               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000             0.000000               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000             0.000000               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.511278         83.333333             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               4.135338         81.818182             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               4.135338         81.818182             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.511278         83.333333             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB01_CHA1_1011_1229_parquet_slot_delta_prior.csv

[2/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB01_CHA2_1011_1229.parquet
loaded shape=(3338, 866)

==================================================================================================================================
Dataset: EPLBAB01_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(3338, 867), sort_time=0.009s
label out-of-range run_value=0/3338, policy=clip
split: train=[0,2670), cal=[2670,3171), test=[3171,3338)
split sizes: train=2670, cal=501, test=167

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           93   3.483146         11  2.195609           5  2.994012
    3          537  20.112360        108 21.556886          53 31.736527
    4         1239  46.404494        213 42.514970          90 53.892216
    5          709  26.554307        156 31.137725          18 10.778443
    6           90   3.370787         11  2.195609           1  0.598802
    7            2   0.074906          2  0.399202           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.208458
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 210         -0.903151       -0.805545      -3.432093       1.366345       4.798438       -0.903151                 False
       6   3          0.536034       -0.742621      -2.487572       1.641657       4.129229        0.536034                 False
       7 188          0.414762        0.874748      -1.367278       3.178883       4.546160        0.414762                 False
       9 217          0.860149        0.993644      -1.285374       3.177967       4.463341        0.860149                 False
      10   3          1.292175        2.539608      -0.160141       4.615641       4.775782        1.292175                 False
      11 189         -0.321243       -0.104300      -2.777843       2.035007       4.812851       -0.321243                 False
      14   2          1.460539        1.460539       0.218902       2.702176       2.483274       -0.208458                  True
      15 189         -0.716888       -0.522202      -2.757332       1.759438       4.516769       -0.716888                 False
      16   1          1.907291        1.907291       1.907291       1.907291       0.000000       -0.208458                  True
      17 216         -0.954577       -0.850549      -3.188485       1.627406       4.815891       -0.954577                 False
      18   2         -1.264225       -1.264225      -2.693474       0.165024       2.858498       -0.208458                  True
      19 190         -1.452330       -1.227300      -3.145462       0.765998       3.911460       -1.452330                 False
      25 217          0.479092        0.640906      -1.603683       3.580875       5.184559        0.479092                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=863 -> miss=838 -> var=691 -> final=120
  C_loop_count: fit_time=2.24s pred_time=1.78s

  [CAL FULL C_loop_count]
    n=304 | Acc=62.83% | Within1=99.34% | Severe(|d|>=2)=0.66% | MAE=0.3783 | RMSE=0.6257 | Penalty=0.3914 | MeanDiff=0.0493

  [TEST FULL C_loop_count]
    n=103 | Acc=64.08% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3592 | RMSE=0.5994 | Penalty=0.3592 | MeanDiff=0.0097

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=893 -> miss=868 -> var=721 -> final=120
  F_delta_run_trend: fit_time=2.07s pred_time=1.72s

  [CAL FULL F_delta_run_trend]
    n=301 | Acc=67.77% | Within1=99.67% | Severe(|d|>=2)=0.33% | MAE=0.3256 | RMSE=0.5764 | Penalty=0.3322 | MeanDiff=0.0465

  [TEST FULL F_delta_run_trend]
    n=102 | Acc=64.71% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3529 | RMSE=0.5941 | Penalty=0.3529 | MeanDiff=-0.0980

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         301     67.774086         0.332226          102      64.705882         100.000000          0.000000           0.352941              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         304     62.828947         0.657895          103      64.077670         100.000000          0.000000           0.359223              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               6.578947         90.000000             0.000000               2               1.941748        100.000000            100.000000             0.000000              0.000000                  True      0.200000               1.500000        999.000000                0.000000         4.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               6.578947         90.000000             0.000000               2               1.941748        100.000000            100.000000             0.000000              0.000000                  True      0.200000               1.500000        999.000000                0.000000         4.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.289474         90.000000             0.000000               5               4.854369         80.000000            100.000000             0.000000              0.200000                  True      0.000000               0.750000        999.000000                0.000000         4.000000            4.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143             0.000000               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143             0.000000               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143             0.000000               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143             0.000000               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.289474         90.000000             0.000000               3               2.912621         33.333333            100.000000             0.000000              0.666667                  True      0.200000               0.750000        999.000000                0.000000       999.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB01_CHA2_1011_1229_parquet_slot_delta_prior.csv

[3/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB01_CHB1_1011_1229.parquet
loaded shape=(3108, 801)

==================================================================================================================================
Dataset: EPLBAB01_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(3108, 802), sort_time=0.005s
label out-of-range run_value=1/3108, policy=clip
split: train=[0,2486), cal=[2486,2952), test=[2952,3108)
split sizes: train=2486, cal=466, test=156

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2            6   0.241352          2  0.429185           2  1.282051
    3          152   6.114240         56 12.017167          21 13.461538
    4          982  39.501207        237 50.858369          61 39.102564
    5         1077  43.322607        160 34.334764          60 38.461538
    6          258  10.378117         11  2.360515          12  7.692308
    7           11   0.442478          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.244745
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   3          1.151672        2.749014      -0.665770       5.365127       6.030896        1.151672                 False
       6 223          0.978285        0.828136      -1.667080       3.069958       4.737038        0.978285                 False
       7   3         -2.443466       -0.406338      -2.703857       0.872618       3.576475       -2.443466                 False
       8 186          0.028179        0.262735      -1.823328       2.774693       4.598020        0.028179                 False
      10 223          0.865673        0.519314      -1.307746       2.687925       3.995671        0.865673                 False
      11   3         -2.475834       -2.133170      -3.164150      -1.273521       1.890629       -2.475834                 False
      14 223         -0.680920       -0.649185      -2.534143       1.600023       4.134167       -0.680920                 False
      15   2         -1.822906       -1.822906      -1.829901      -1.815912       0.013988       -0.244745                  True
      16 189          0.052589       -0.094940      -2.154594       2.022804       4.177399        0.052589                 False
      18 223         -1.013031       -1.604574      -3.204199       1.216664       4.420863       -1.013031                 False
      19   2         -4.767457       -4.767457      -5.001576      -4.533338       0.468238       -0.244745                  True
      24 188         -1.490345       -1.678224      -3.913422       0.434273       4.347694       -1.490345                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=798 -> miss=753 -> var=616 -> final=120
  C_loop_count: fit_time=2.00s pred_time=1.74s

  [CAL FULL C_loop_count]
    n=269 | Acc=62.08% | Within1=99.26% | Severe(|d|>=2)=0.74% | MAE=0.3866 | RMSE=0.6336 | Penalty=0.4015 | MeanDiff=0.0892

  [TEST FULL C_loop_count]
    n=94 | Acc=62.77% | Within1=98.94% | Severe(|d|>=2)=1.06% | MAE=0.3830 | RMSE=0.6358 | Penalty=0.4043 | MeanDiff=0.1277

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=828 -> miss=783 -> var=646 -> final=120
  F_delta_run_trend: fit_time=2.08s pred_time=1.85s

  [CAL FULL F_delta_run_trend]
    n=269 | Acc=68.77% | Within1=98.88% | Severe(|d|>=2)=1.12% | MAE=0.3234 | RMSE=0.5880 | Penalty=0.3457 | MeanDiff=0.1004

  [TEST FULL F_delta_run_trend]
    n=94 | Acc=62.77% | Within1=98.94% | Severe(|d|>=2)=1.06% | MAE=0.3830 | RMSE=0.6358 | Penalty=0.4043 | MeanDiff=0.1064

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         269     62.081784         0.743494           94      62.765957          98.936170          1.063830           0.404255              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         269     68.773234         1.115242           94      62.765957          98.936170          1.063830           0.404255              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32              11.895911         84.375000             0.000000              11              11.702128         63.636364            100.000000             0.000000              0.363636                 False      0.300000               0.250000        999.000000                0.000000         3.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               5.204461         85.714286             0.000000              12              12.765957         58.333333            100.000000             0.000000              0.416667                 False      0.400000               1.000000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.832714         92.307692             0.000000              12              12.765957         58.333333            100.000000             0.000000              0.416667                 False      0.400000               1.000000        999.000000                0.000000         2.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32              11.895911         87.500000             0.000000              19              20.212766         52.631579            100.000000             0.000000              0.473684                 False      0.000000               0.250000        999.000000                0.000000         2.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              12               4.460967         83.333333             0.000000              10              10.638298         50.000000            100.000000             0.000000              0.500000                 False      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              12               4.460967         83.333333             0.000000              10              10.638298         50.000000            100.000000             0.000000              0.500000                 False      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.460967         91.666667             0.000000               8               8.510638         37.500000            100.000000             0.000000              0.625000                  True      0.400000               1.000000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.832714         84.615385             0.000000               8               8.510638         37.500000            100.000000             0.000000              0.625000                  True      0.400000               1.000000        999.000000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB01_CHB1_1011_1229_parquet_slot_delta_prior.csv

[4/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB01_CHB2_1011_1229.parquet
loaded shape=(3361, 803)

==================================================================================================================================
Dataset: EPLBAB01_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(3361, 804), sort_time=0.012s
label out-of-range run_value=0/3361, policy=clip
split: train=[0,2688), cal=[2688,3192), test=[3192,3361)
split sizes: train=2688, cal=504, test=169

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           76   2.827381          1  0.198413           2  1.183432
    3          742  27.604167         94 18.650794          36 21.301775
    4         1385  51.525298        281 55.753968          95 56.213018
    5          459  17.075893        124 24.603175          33 19.526627
    6           25   0.930060          4  0.793651           2  1.183432
    7            1   0.037202          0  0.000000           1  0.591716
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.157761
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 177         -0.401052       -0.031828      -1.978703       2.027164       4.005867       -0.401052                 False
       7 223          0.548061        0.637763      -1.754282       2.791701       4.545983        0.548061                 False
       8   3          0.030972       -2.157885      -3.579220       0.357878       3.937098        0.030972                 False
       9 187          0.596348        0.785897      -1.268729       2.700301       3.969030        0.596348                 False
      11 223         -0.305643       -0.161666      -2.094830       2.135708       4.230537       -0.305643                 False
      14   1         -2.326008       -2.326008      -2.326008      -2.326008       0.000000       -0.157761                  True
      15 223         -0.298960       -0.598655      -2.500325       1.439692       3.940017       -0.298960                 False
      16   2         -3.120295       -3.120295      -3.178590      -3.061999       0.116590       -0.157761                  True
      17 189         -0.450386       -0.610353      -3.104605       1.770218       4.874823       -0.450386                 False
      19 223         -0.936794       -0.921022      -3.150686       1.241709       4.392395       -0.936794                 False
      24   1         -1.972832       -1.972832      -1.972832      -1.972832       0.000000       -0.157761                  True
      25 189          0.345793        0.370688      -2.221649       2.557503       4.779152        0.345793                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=800 -> miss=755 -> var=608 -> final=120
  C_loop_count: fit_time=2.12s pred_time=1.78s

  [CAL FULL C_loop_count]
    n=309 | Acc=58.25% | Within1=99.03% | Severe(|d|>=2)=0.97% | MAE=0.4272 | RMSE=0.6683 | Penalty=0.4466 | MeanDiff=0.0906

  [TEST FULL C_loop_count]
    n=104 | Acc=58.65% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.4135 | RMSE=0.6430 | Penalty=0.4135 | MeanDiff=0.0096

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=830 -> miss=785 -> var=638 -> final=120
  F_delta_run_trend: fit_time=2.07s pred_time=1.88s

  [CAL FULL F_delta_run_trend]
    n=309 | Acc=62.14% | Within1=99.03% | Severe(|d|>=2)=0.97% | MAE=0.3883 | RMSE=0.6386 | Penalty=0.4078 | MeanDiff=0.0259

  [TEST FULL F_delta_run_trend]
    n=104 | Acc=67.31% | Within1=98.08% | Severe(|d|>=2)=1.92% | MAE=0.3462 | RMSE=0.6202 | Penalty=0.3846 | MeanDiff=0.0192

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         309     62.135922         0.970874          104      67.307692          98.076923          1.923077           0.384615              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         309     58.252427         0.970874          104      58.653846         100.000000          0.000000           0.413462              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              22               7.119741         81.818182             0.000000               6               5.769231         66.666667            100.000000             0.000000              0.333333                  True      0.200000               0.250000          1.500000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               7.443366         78.260870             0.000000               6               5.769231         66.666667            100.000000             0.000000              0.333333                  True      0.200000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               7.443366         78.260870             0.000000               6               5.769231         66.666667            100.000000             0.000000              0.333333                  True      0.200000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              46              14.886731         78.260870             0.000000              14              13.461538         64.285714            100.000000             0.000000              0.357143                 False      0.100000               0.250000        999.000000                0.000000       999.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              25               8.090615         80.000000             0.000000               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.500000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               4.530744         85.714286             0.000000               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.500000        999.000000                0.500000       999.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.883495         83.333333             0.000000               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.500000        999.000000                0.500000       999.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              26               8.414239         80.769231             0.000000               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True      0.200000               0.500000        999.000000                0.000000         3.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB01_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB01_CHB2_1011_1229_parquet_slot_delta_prior.csv

[5/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB02_CHA1_1011_1229.parquet
loaded shape=(2886, 862)

==================================================================================================================================
Dataset: EPLBAB02_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(2886, 863), sort_time=0.010s
label out-of-range run_value=0/2886, policy=clip
split: train=[0,2308), cal=[2308,2741), test=[2741,2886)
split sizes: train=2308, cal=433, test=145

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           28   1.213172          5  1.154734           1  0.689655
    3          477  20.667244        118 27.251732          39 26.896552
    4         1271  55.069324        248 57.274827          84 57.931034
    5          503  21.793761         61 14.087760          20 13.793103
    6           29   1.256499          1  0.230947           1  0.689655
    7            0   0.000000          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=0.095469
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   6          2.062696        3.002542       0.715117       6.604730       5.889614        2.062696                 False
       6 196          0.769361        0.931293      -1.085365       3.088140       4.173504        0.769361                 False
       7   3         -0.461422       -0.819786      -1.076292      -0.384098       0.692194       -0.461422                 False
       8 185          1.117563        0.793059      -1.150570       3.026194       4.176764        1.117563                 False
       9   1          6.480337        6.480337       6.480337       6.480337       0.000000        0.095469                  True
      10 195          1.111462        1.042108      -1.306917       3.030840       4.337757        1.111462                 False
      11   3         -1.266171       -2.374971      -3.334213      -0.861328       2.472885       -1.266171                 False
      14 193         -0.312733       -0.270173      -2.328537       2.017284       4.345821       -0.312733                 False
      15   3         -3.707401       -2.755530      -4.227835      -1.759161       2.468674       -3.707401                 False
      16 188          0.476896        0.316734      -1.592048       2.130696       3.722744        0.476896                 False
      17   1          3.532358        3.532358       3.532358       3.532358       0.000000        0.095469                  True
      18 186         -1.013975       -1.116626      -3.424417       0.679898       4.104315       -1.013975                 False
      19   2         -2.356554       -2.356554      -3.558888      -1.154220       2.404669        0.095469                  True
      24 190         -0.915614       -0.893372      -3.495386       1.902606       5.397992       -0.915614                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=859 -> miss=816 -> var=666 -> final=120
  C_loop_count: fit_time=2.23s pred_time=1.73s

  [CAL FULL C_loop_count]
    n=251 | Acc=67.33% | Within1=98.80% | Severe(|d|>=2)=1.20% | MAE=0.3386 | RMSE=0.6021 | Penalty=0.3625 | MeanDiff=-0.0916

  [TEST FULL C_loop_count]
    n=82 | Acc=60.98% | Within1=98.78% | Severe(|d|>=2)=1.22% | MAE=0.4024 | RMSE=0.6533 | Penalty=0.4268 | MeanDiff=-0.0854

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=889 -> miss=846 -> var=696 -> final=120
  F_delta_run_trend: fit_time=2.41s pred_time=1.73s

  [CAL FULL F_delta_run_trend]
    n=251 | Acc=70.12% | Within1=99.60% | Severe(|d|>=2)=0.40% | MAE=0.3028 | RMSE=0.5575 | Penalty=0.3108 | MeanDiff=-0.0159

  [TEST FULL F_delta_run_trend]
    n=82 | Acc=67.07% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3293 | RMSE=0.5738 | Penalty=0.3293 | MeanDiff=-0.0854

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         251     70.119522         0.398406           82      67.073171         100.000000          0.000000           0.329268              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         251     67.330677         1.195219           82      60.975610          98.780488          1.219512           0.426829              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842             0.000000               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842             0.000000               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842             0.000000               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842             0.000000               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               7.968127         90.000000             0.000000               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               6.772908         94.117647             0.000000               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               6.772908         94.117647             0.000000               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               7.968127         90.000000             0.000000               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB02_CHA1_1011_1229_parquet_slot_delta_prior.csv

[6/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB02_CHA2_1011_1229.parquet
loaded shape=(3143, 845)

==================================================================================================================================
Dataset: EPLBAB02_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(3143, 846), sort_time=0.012s
label out-of-range run_value=0/3143, policy=clip
split: train=[0,2514), cal=[2514,2985), test=[2985,3143)
split sizes: train=2514, cal=471, test=158

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           10   0.397772          0  0.000000           0  0.000000
    3          217   8.631663         37  7.855626           8  5.063291
    4         1249  49.681782        267 56.687898          77 48.734177
    5          932  37.072395        154 32.696391          69 43.670886
    6          106   4.216388         13  2.760085           4  2.531646
    7            0   0.000000          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.544838
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 187         -1.130001       -1.296900      -3.322796       0.750126       4.072922       -1.130001                 False
       6   1          2.421272        2.421272       2.421272       2.421272       0.000000       -0.544838                  True
       7 196          0.581343        0.343346      -1.928093       2.607998       4.536091        0.581343                 False
       8   3          1.966267        0.985817      -0.335072       2.796930       3.132002        1.966267                 False
       9 186          0.571304        0.606645      -1.452024       2.628162       4.080186        0.571304                 False
      10   1          9.075756        9.075756       9.075756       9.075756       0.000000       -0.544838                  True
      11 196         -0.580238       -0.615870      -2.736484       1.567831       4.304315       -0.580238                 False
      14   1          0.932816        0.932816       0.932816       0.932816       0.000000       -0.544838                  True
      15 194         -0.979822       -1.108772      -3.350970       0.703968       4.054938       -0.979822                 False
      16   3          3.617424        3.059892       2.521797       3.876753       1.354956        3.617424                 False
      17 190         -0.737635       -0.739558      -2.771885       1.094700       3.866585       -0.737635                 False
      18   1          1.716034        1.716034       1.716034       1.716034       0.000000       -0.544838                  True
      19 187         -1.493019       -1.349986      -3.639056       0.976452       4.615508       -1.493019                 False
      24   1          0.554745        0.554745       0.554745       0.554745       0.000000       -0.544838                  True
      25 190          0.151688        0.208213      -2.278113       3.266128       5.544240        0.151688                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=842 -> miss=780 -> var=629 -> final=120
  C_loop_count: fit_time=2.09s pred_time=1.74s

  [CAL FULL C_loop_count]
    n=289 | Acc=66.78% | Within1=98.27% | Severe(|d|>=2)=1.73% | MAE=0.3495 | RMSE=0.6197 | Penalty=0.3841 | MeanDiff=-0.1073

  [TEST FULL C_loop_count]
    n=95 | Acc=64.21% | Within1=95.79% | Severe(|d|>=2)=4.21% | MAE=0.4000 | RMSE=0.6959 | Penalty=0.4842 | MeanDiff=-0.0211

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=872 -> miss=810 -> var=659 -> final=120
  F_delta_run_trend: fit_time=2.04s pred_time=1.74s

  [CAL FULL F_delta_run_trend]
    n=289 | Acc=68.51% | Within1=98.96% | Severe(|d|>=2)=1.04% | MAE=0.3253 | RMSE=0.5882 | Penalty=0.3460 | MeanDiff=-0.0277

  [TEST FULL F_delta_run_trend]
    n=95 | Acc=64.21% | Within1=96.84% | Severe(|d|>=2)=3.16% | MAE=0.3895 | RMSE=0.6728 | Penalty=0.4526 | MeanDiff=0.0316

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         289     66.782007         1.730104           95      64.210526          95.789474          4.210526           0.484211              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         289     68.512111         1.038062           95      64.210526          96.842105          3.157895           0.452632              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               5.190311         86.666667             0.000000               7               7.368421         85.714286            100.000000             0.000000              0.142857                  True      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               4.844291         85.714286             0.000000               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               5.536332         93.750000             0.000000               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True      0.400000               0.250000        999.000000                0.000000         4.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               5.882353         94.117647             0.000000               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True      0.400000               0.250000        999.000000                0.000000       999.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               5.882353         88.235294             0.000000               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True      0.400000               0.250000        999.000000                0.500000         4.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.498270         92.307692             0.000000              11              11.578947         72.727273             90.909091             9.090909              0.545455                 False      0.300000               0.500000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               5.190311         93.333333             0.000000              12              12.631579         66.666667            100.000000             0.000000              0.333333                 False      0.300000               0.250000          2.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.152249         91.666667             0.000000               9               9.473684         66.666667             88.888889            11.111111              0.666667                  True      0.300000               0.500000        999.000000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB02_CHA2_1011_1229_parquet_slot_delta_prior.csv

[7/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB02_CHB1_1011_1229.parquet
loaded shape=(2691, 782)

==================================================================================================================================
Dataset: EPLBAB02_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(2691, 783), sort_time=0.011s
label out-of-range run_value=0/2691, policy=clip
split: train=[0,2152), cal=[2152,2556), test=[2556,2691)
split sizes: train=2152, cal=404, test=135

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           12   0.557621          1  0.247525           0  0.000000
    3          228  10.594796         68 16.831683          12  8.888889
    4          994  46.189591        228 56.435644          86 63.703704
    5          784  36.431227        102 25.247525          36 26.666667
    6          119   5.529740          5  1.237624           1  0.740741
    7           14   0.650558          0  0.000000           0  0.000000
    8            1   0.046468          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.200584
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   6         -1.255278       -0.930166      -3.052659       1.284708       4.337367       -1.255278                 False
       6 171          0.219410        0.434103      -1.970541       2.742515       4.713056        0.219410                 False
       7   2          0.068306        0.068306      -0.015164       0.151776       0.166941       -0.200584                  True
       8 182          1.034819        0.645500      -1.348028       2.781590       4.129619        1.034819                 False
       9   4          0.307137        0.238622      -0.282602       0.828362       1.110964        0.307137                 False
      10 171          0.449045        0.595544      -1.624226       2.509596       4.133821        0.449045                 False
      11   2         -0.283560       -0.283560      -0.723965       0.156845       0.880810       -0.200584                  True
      14 173         -0.915962       -0.952394      -3.098623       1.022327       4.120951       -0.915962                 False
      15   2          3.076570        3.076570       2.863827       3.289312       0.425485       -0.200584                  True
      16 177          0.159500        0.189136      -1.420940       1.665684       3.086624        0.159500                 False
      17   4         -4.737561       -3.442648      -5.120152      -3.060058       2.060093       -4.737561                 False
      18 180         -1.294163       -1.242096      -3.204821       0.716200       3.921021       -1.294163                 False
      19   2         -0.628828       -0.628828      -1.020103      -0.237553       0.782551       -0.200584                  True
      24 181         -0.931679       -0.958198      -3.185345       1.216286       4.401630       -0.931679                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=779 -> miss=733 -> var=591 -> final=120
  C_loop_count: fit_time=2.05s pred_time=1.73s

  [CAL FULL C_loop_count]
    n=236 | Acc=66.10% | Within1=99.58% | Severe(|d|>=2)=0.42% | MAE=0.3432 | RMSE=0.5930 | Penalty=0.3517 | MeanDiff=-0.0466

  [TEST FULL C_loop_count]
    n=82 | Acc=58.54% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.4146 | RMSE=0.6439 | Penalty=0.4146 | MeanDiff=0.0000

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=809 -> miss=763 -> var=621 -> final=120
  F_delta_run_trend: fit_time=2.32s pred_time=1.71s

  [CAL FULL F_delta_run_trend]
    n=236 | Acc=68.22% | Within1=99.15% | Severe(|d|>=2)=0.85% | MAE=0.3263 | RMSE=0.5859 | Penalty=0.3432 | MeanDiff=-0.0381

  [TEST FULL F_delta_run_trend]
    n=82 | Acc=53.66% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.4634 | RMSE=0.6807 | Penalty=0.4634 | MeanDiff=0.0488

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         236     66.101695         0.423729           82      58.536585         100.000000          0.000000           0.414634              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         236     68.220339         0.847458           82      53.658537         100.000000          0.000000           0.463415              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop  trend            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop   both            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop either            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               6.355932         86.666667             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               5.508475         92.307692             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.500000          1.500000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               6.355932         86.666667             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               6.355932         86.666667             0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB02_CHB1_1011_1229_parquet_slot_delta_prior.csv

[8/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB02_CHB2_1011_1229.parquet
loaded shape=(2903, 784)

==================================================================================================================================
Dataset: EPLBAB02_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(2903, 785), sort_time=0.010s
label out-of-range run_value=0/2903, policy=clip
split: train=[0,2322), cal=[2322,2757), test=[2757,2903)
split sizes: train=2322, cal=435, test=146

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           30   1.291990          0  0.000000           1  0.684932
    3          505  21.748493         64 14.712644          11  7.534247
    4         1178  50.732127        228 52.413793          96 65.753425
    5          563  24.246339        132 30.344828          37 25.342466
    6           45   1.937984         11  2.528736           1  0.684932
    7            1   0.043066          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.375599
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 161          0.430279        0.426533      -1.570213       2.311249       3.881462        0.430279                 False
       6   4         -0.567920       -1.092506      -3.838536       2.178110       6.016646       -0.567920                 False
       7 171         -0.070019       -0.246414      -2.427201       1.822368       4.249569       -0.070019                 False
       8   2          1.748398        1.748398       1.226818       2.269978       1.043159       -0.375599                  True
       9 182          1.118523        1.157887      -0.736818       2.962056       3.698874        1.118523                 False
      10   4          0.502737       -0.799300      -1.348748       1.052186       2.400934        0.502737                 False
      11 171         -0.403811       -0.684509      -2.636587       1.211069       3.847656       -0.403811                 False
      14   4         -1.925610       -1.336264      -3.797390       0.535517       4.332907       -1.925610                 False
      15 173         -1.706097       -1.790354      -3.642323       0.401344       4.043667       -1.706097                 False
      16   2          2.033381        2.033381       1.792531       2.274232       0.481702       -0.375599                  True
      17 177         -0.549397       -1.160710      -3.768620       1.142372       4.910992       -0.549397                 False
      18   3         -3.873791       -3.861890      -4.200162      -3.529568       0.670594       -3.873791                 False
      19 181         -1.566002       -1.722824      -3.828903       0.612068       4.440971       -1.566002                 False
      25 182          0.227777        0.347073      -2.053489       2.735920       4.789409        0.227777                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=781 -> miss=716 -> var=573 -> final=120
  C_loop_count: fit_time=2.10s pred_time=1.76s

  [CAL FULL C_loop_count]
    n=264 | Acc=64.77% | Within1=99.62% | Severe(|d|>=2)=0.38% | MAE=0.3561 | RMSE=0.6030 | Penalty=0.3636 | MeanDiff=0.0303

  [TEST FULL C_loop_count]
    n=92 | Acc=67.39% | Within1=98.91% | Severe(|d|>=2)=1.09% | MAE=0.3370 | RMSE=0.5989 | Penalty=0.3587 | MeanDiff=0.1413

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=811 -> miss=746 -> var=603 -> final=120
  F_delta_run_trend: fit_time=2.06s pred_time=1.77s

  [CAL FULL F_delta_run_trend]
    n=262 | Acc=65.27% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3473 | RMSE=0.5893 | Penalty=0.3473 | MeanDiff=-0.0420

  [TEST FULL F_delta_run_trend]
    n=92 | Acc=68.48% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3152 | RMSE=0.5614 | Penalty=0.3152 | MeanDiff=0.0978

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         262     65.267176         0.000000           92      68.478261         100.000000          0.000000           0.315217              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         264     64.772727         0.378788           92      67.391304          98.913043          1.086957           0.358696              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               4.961832         84.615385             0.000000              14              15.217391         85.714286            100.000000             0.000000              0.142857                 False      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               4.961832         84.615385             0.000000              14              15.217391         85.714286            100.000000             0.000000              0.142857                 False      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.787879         90.000000             0.000000              24              26.086957         75.000000            100.000000             0.000000              0.250000                 False      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               7.251908         84.210526             0.000000              20              21.739130         75.000000            100.000000             0.000000              0.250000                 False      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               7.251908         84.210526             0.000000              20              21.739130         75.000000            100.000000             0.000000              0.250000                 False      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               5.303030         85.714286             0.000000              23              25.000000         73.913043            100.000000             0.000000              0.260870                 False      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.787879         90.000000             0.000000              18              19.565217         72.222222            100.000000             0.000000              0.277778                 False      0.300000               0.250000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.924242         84.615385             0.000000              21              22.826087         71.428571            100.000000             0.000000              0.285714                 False      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB02_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB02_CHB2_1011_1229_parquet_slot_delta_prior.csv

[9/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB03_CHA1_1011_1229.parquet
loaded shape=(3899, 806)

==================================================================================================================================
Dataset: EPLBAB03_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(3899, 807), sort_time=0.014s
label out-of-range run_value=1/3899, policy=clip
split: train=[0,3119), cal=[3119,3704), test=[3704,3899)
split sizes: train=3119, cal=585, test=195

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           77   2.468740         10  1.709402          18  9.230769
    3          751  24.078230        138 23.589744          65 33.333333
    4         1525  48.893876        291 49.743590          81 41.538462
    5          702  22.507214        133 22.735043          25 12.820513
    6           63   2.019878         13  2.222222           6  3.076923
    7            1   0.032062          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=0.012827
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   6          3.077064        5.016123      -0.185514      10.861910      11.047424        3.077064                 False
       6 218          0.647701        0.504936      -1.770427       2.866820       4.637248        0.647701                 False
       7   2         -1.544939       -1.544939      -1.748616      -1.341262       0.407354        0.012827                  True
       8 295          1.099258        0.883586      -1.322390       2.871424       4.193813        1.099258                 False
       9   2         -0.502404       -0.502404      -0.939432      -0.065377       0.874055        0.012827                  True
      10 222          1.253243        1.273061      -1.209743       2.977826       4.187568        1.253243                 False
      11   2          0.033702        0.033702      -0.547960       0.615364       1.163323        0.012827                  True
      14 226         -0.222898       -0.242670      -2.460757       2.199821       4.660578       -0.222898                 False
      15   1          2.064558        2.064558       2.064558       2.064558       0.000000        0.012827                  True
      16 294          0.228852        0.193600      -1.609422       2.073092       3.682514        0.228852                 False
      17   1         -2.607533       -2.607533      -2.607533      -2.607533       0.000000        0.012827                  True
      18 231         -0.685947       -0.947359      -2.991256       0.958614       3.949870       -0.685947                 False
      24 286         -1.949999       -2.028010      -4.656851       0.672482       5.329333       -1.949999                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=803 -> miss=799 -> var=644 -> final=120
  C_loop_count: fit_time=2.28s pred_time=1.78s

  [CAL FULL C_loop_count]
    n=349 | Acc=60.74% | Within1=98.28% | Severe(|d|>=2)=1.72% | MAE=0.4097 | RMSE=0.6664 | Penalty=0.4441 | MeanDiff=-0.1347

  [TEST FULL C_loop_count]
    n=112 | Acc=69.64% | Within1=99.11% | Severe(|d|>=2)=0.89% | MAE=0.3125 | RMSE=0.5748 | Penalty=0.3304 | MeanDiff=-0.0446

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=833 -> miss=829 -> var=674 -> final=120
  F_delta_run_trend: fit_time=2.12s pred_time=1.82s

  [CAL FULL F_delta_run_trend]
    n=349 | Acc=66.19% | Within1=97.71% | Severe(|d|>=2)=2.29% | MAE=0.3610 | RMSE=0.6379 | Penalty=0.4069 | MeanDiff=-0.0917

  [TEST FULL F_delta_run_trend]
    n=112 | Acc=68.75% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3125 | RMSE=0.5590 | Penalty=0.3125 | MeanDiff=0.0268

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         349     60.744986         1.719198          112      69.642857          99.107143          0.892857           0.330357              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         349     66.189112         2.292264          112      68.750000         100.000000          0.000000           0.312500              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.724928         92.307692             0.000000               5               4.464286        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop  trend            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.151862        100.000000             0.000000               3               2.678571        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.500000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.724928         92.307692             0.000000               2               1.785714        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.500000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop   both            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.151862        100.000000             0.000000               2               1.785714        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.500000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               4.297994         93.333333             0.000000              10               8.928571         90.000000            100.000000             0.000000              0.100000                 False      0.400000               0.250000        999.000000                0.000000         4.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               7.736390         92.592593             0.000000              14              12.500000         85.714286            100.000000             0.000000              0.142857                 False      0.300000               0.500000          2.000000                0.000000         2.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               7.449857         92.307692             0.000000              14              12.500000         85.714286            100.000000             0.000000              0.142857                 False      0.300000               0.250000        999.000000                0.000000         2.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.730659         90.000000             0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False      0.300000               0.250000        999.000000                0.000000         2.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB03_CHA1_1011_1229_parquet_slot_delta_prior.csv

[10/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB03_CHA2_1011_1229.parquet
loaded shape=(4244, 789)

==================================================================================================================================
Dataset: EPLBAB03_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(4244, 790), sort_time=0.021s
label out-of-range run_value=1/4244, policy=clip
split: train=[0,3395), cal=[3395,4031), test=[4031,4244)
split sizes: train=3395, cal=636, test=213

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           20   0.589102          2  0.314465           1  0.469484
    3          374  11.016200         72 11.320755          28 13.145540
    4         1637  48.217968        372 58.490566         112 52.582160
    5         1200  35.346097        180 28.301887          70 32.863850
    6          158   4.653903         10  1.572327           2  0.938967
    7            6   0.176730          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.225918
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 275         -0.763115       -0.735469      -3.319889       1.782915       5.102804       -0.763115                 False
       6   2          2.927525        2.927525       2.711048       3.144001       0.432953       -0.225918                  True
       7 217          0.836241        0.853042      -1.199785       2.979183       4.178968        0.836241                 False
       8   2          1.423321        1.423321       0.554483       2.292158       1.737675       -0.225918                  True
       9 294          0.880169        0.899825      -1.060095       2.952468       4.012563        0.880169                 False
      10   2         -0.294224       -0.294224      -1.627813       1.039366       2.667179       -0.225918                  True
      11 221         -0.200438       -0.336409      -2.041866       1.758247       3.800114       -0.200438                 False
      14   1         -0.487457       -0.487457      -0.487457      -0.487457       0.000000       -0.225918                  True
      15 225         -0.645624       -1.119606      -3.047836       0.834991       3.882828       -0.645624                 False
      17 294         -1.072124       -1.209530      -3.316246       1.189516       4.505762       -1.072124                 False
      18   1         -5.305424       -5.305424      -5.305424      -5.305424       0.000000       -0.225918                  True
      19 230         -0.986491       -1.141166      -3.392000       1.385023       4.777023       -0.986491                 False
      25 285          0.074377        0.171649      -1.938696       2.290157       4.228853        0.074377                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=786 -> miss=762 -> var=610 -> final=120
  C_loop_count: fit_time=2.07s pred_time=1.86s

  [CAL FULL C_loop_count]
    n=392 | Acc=61.22% | Within1=99.49% | Severe(|d|>=2)=0.51% | MAE=0.3929 | RMSE=0.6349 | Penalty=0.4031 | MeanDiff=-0.0255

  [TEST FULL C_loop_count]
    n=130 | Acc=59.23% | Within1=98.46% | Severe(|d|>=2)=1.54% | MAE=0.4231 | RMSE=0.6737 | Penalty=0.4538 | MeanDiff=-0.1308

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=816 -> miss=792 -> var=640 -> final=120
  F_delta_run_trend: fit_time=2.22s pred_time=1.81s

  [CAL FULL F_delta_run_trend]
    n=392 | Acc=63.78% | Within1=97.70% | Severe(|d|>=2)=2.30% | MAE=0.3878 | RMSE=0.6662 | Penalty=0.4439 | MeanDiff=0.0153

  [TEST FULL F_delta_run_trend]
    n=128 | Acc=64.84% | Within1=99.22% | Severe(|d|>=2)=0.78% | MAE=0.3594 | RMSE=0.6124 | Penalty=0.3750 | MeanDiff=-0.0156

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         392     63.775510         2.295918          128      64.843750          99.218750          0.781250           0.375000              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         392     61.224490         0.510204          130      59.230769          98.461538          1.538462           0.453846              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              45              11.479592         82.222222             0.000000              22              16.923077         86.363636            100.000000             0.000000              0.136364                 False      0.200000               0.250000        999.000000                0.000000       999.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              27               6.887755         85.185185             0.000000               7               5.384615         85.714286            100.000000             0.000000              0.142857                  True      0.200000               0.250000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              64              16.326531         79.687500             0.000000              32              24.615385         75.000000             96.875000             3.125000              0.343750                 False      0.200000               0.250000        999.000000                0.000000       999.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               7.142857         82.142857             0.000000               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000          1.500000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               6.632653         76.923077             0.000000               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               6.632653         76.923077             0.000000               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.653061         80.000000             0.000000               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              16               4.081633         81.250000             0.000000               3               2.307692         66.666667            100.000000             0.000000              0.333333                  True      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB03_CHA2_1011_1229_parquet_slot_delta_prior.csv

[11/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB03_CHB1_1011_1229.parquet
loaded shape=(3586, 762)

==================================================================================================================================
Dataset: EPLBAB03_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(3586, 763), sort_time=0.012s
label out-of-range run_value=0/3586, policy=clip
split: train=[0,2868), cal=[2868,3406), test=[3406,3586)
split sizes: train=2868, cal=538, test=180

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           34   1.185495         12  2.230483           2  1.111111
    3          441  15.376569         99 18.401487          41 22.777778
    4         1365  47.594142        270 50.185874          98 54.444444
    5          916  31.938633        143 26.579926          38 21.111111
    6          108   3.765690         14  2.602230           1  0.555556
    7            4   0.139470          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.138584
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   3          2.084576        4.478276       1.861475       5.898227       4.036752        2.084576                 False
       6 276          0.532778        0.664593      -1.694752       3.173738       4.868489        0.532778                 False
       7   1         -4.661079       -4.661079      -4.661079      -4.661079       0.000000       -0.138584                  True
       8 202          0.517249        0.816853      -1.665118       3.206189       4.871307        0.517249                 False
       9   1         -5.011925       -5.011925      -5.011925      -5.011925       0.000000       -0.138584                  True
      10 273          0.548002        0.629690      -1.301357       2.626595       3.927952        0.548002                 False
      11   1         -5.678581       -5.678581      -5.678581      -5.678581       0.000000       -0.138584                  True
      14 271         -0.409882       -0.549227      -2.937622       1.487500       4.425122       -0.409882                 False
      15   1         -6.199043       -6.199043      -6.199043      -6.199043       0.000000       -0.138584                  True
      16 204          0.234051        0.359149      -1.458241       2.292842       3.751082        0.234051                 False
      18 267         -1.046352       -0.983087      -3.029911       1.401241       4.431152       -1.046352                 False
      19   1         -6.647335       -6.647335      -6.647335      -6.647335       0.000000       -0.138584                  True
      24 214         -1.497842       -1.654010      -4.236884       1.203996       5.440879       -1.497842                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=759 -> miss=715 -> var=575 -> final=120
  C_loop_count: fit_time=2.07s pred_time=1.89s

  [CAL FULL C_loop_count]
    n=310 | Acc=64.84% | Within1=99.68% | Severe(|d|>=2)=0.32% | MAE=0.3548 | RMSE=0.6011 | Penalty=0.3613 | MeanDiff=0.0000

  [TEST FULL C_loop_count]
    n=107 | Acc=62.62% | Within1=99.07% | Severe(|d|>=2)=0.93% | MAE=0.3832 | RMSE=0.6339 | Penalty=0.4019 | MeanDiff=0.0280

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=789 -> miss=745 -> var=605 -> final=120
  F_delta_run_trend: fit_time=2.10s pred_time=1.80s

  [CAL FULL F_delta_run_trend]
    n=310 | Acc=67.42% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3258 | RMSE=0.5708 | Penalty=0.3258 | MeanDiff=-0.0032

  [TEST FULL F_delta_run_trend]
    n=107 | Acc=68.22% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3178 | RMSE=0.5637 | Penalty=0.3178 | MeanDiff=0.1308

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         310     67.419355         0.000000          107      68.224299         100.000000          0.000000           0.317757              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         310     64.838710         0.322581          107      62.616822          99.065421          0.934579           0.401869              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               6.129032         94.736842             0.000000               8               7.476636         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               6.129032         94.736842             0.000000               8               7.476636         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend  trend            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              10               3.225806        100.000000             0.000000               5               4.672897         60.000000            100.000000             0.000000              0.400000                  True      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000
F_delta_run_trend delta_run_trend   both            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              10               3.225806        100.000000             0.000000               5               4.672897         60.000000            100.000000             0.000000              0.400000                  True      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.548387         90.909091             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop  trend            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.225806        100.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               6.451613         95.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.000000               0.500000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.548387         90.909091             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB03_CHB1_1011_1229_parquet_slot_delta_prior.csv

[12/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB03_CHB2_1011_1229.parquet
loaded shape=(3866, 764)

==================================================================================================================================
Dataset: EPLBAB03_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(3866, 765), sort_time=0.012s
label out-of-range run_value=0/3866, policy=clip
split: train=[0,3092), cal=[3092,3672), test=[3672,3866)
split sizes: train=3092, cal=580, test=194

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           39   1.261320          3  0.517241           0  0.000000
    3          519  16.785252         96 16.551724          33 17.010309
    4         1595  51.584735        363 62.586207         106 54.639175
    5          873  28.234153        106 18.275862          54 27.835052
    6           61   1.972833         12  2.068966           1  0.515464
    7            5   0.161708          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.407036
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 207         -0.882465       -0.292617      -2.713491       1.879224       4.592715       -0.882465                 False
       6   1         -0.931128       -0.931128      -0.931128      -0.931128       0.000000       -0.407036                  True
       7 277          0.683685        0.566724      -1.836487       2.610960       4.447447        0.683685                 False
       8   1          3.772743        3.772743       3.772743       3.772743       0.000000       -0.407036                  True
       9 201          0.156540        0.423819      -1.510536       2.303635       3.814171        0.156540                 False
      10   1          3.561035        3.561035       3.561035       3.561035       0.000000       -0.407036                  True
      11 274         -0.370923       -0.425849      -2.603349       1.751796       4.355145       -0.370923                 False
      14   1          0.446289        0.446289       0.446289       0.446289       0.000000       -0.407036                  True
      15 272         -0.857072       -0.785989      -2.758037       1.382958       4.140995       -0.857072                 False
      16   1         -0.389721       -0.389721      -0.389721      -0.389721       0.000000       -0.407036                  True
      17 204         -1.384359       -1.418039      -3.841269       0.817063       4.658332       -1.384359                 False
      19 268         -1.018869       -1.077845      -3.314553       1.139390       4.453943       -1.018869                 False
      25 214          0.262238       -0.008767      -2.598396       2.345949       4.944345        0.262238                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=761 -> miss=717 -> var=573 -> final=120
  C_loop_count: fit_time=2.11s pred_time=1.79s

  [CAL FULL C_loop_count]
    n=354 | Acc=65.25% | Within1=98.87% | Severe(|d|>=2)=1.13% | MAE=0.3588 | RMSE=0.6175 | Penalty=0.3814 | MeanDiff=0.0424

  [TEST FULL C_loop_count]
    n=118 | Acc=64.41% | Within1=99.15% | Severe(|d|>=2)=0.85% | MAE=0.3644 | RMSE=0.6175 | Penalty=0.3814 | MeanDiff=-0.0593

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=791 -> miss=747 -> var=603 -> final=120
  F_delta_run_trend: fit_time=2.14s pred_time=1.79s

  [CAL FULL F_delta_run_trend]
    n=353 | Acc=66.01% | Within1=98.87% | Severe(|d|>=2)=1.13% | MAE=0.3513 | RMSE=0.6115 | Penalty=0.3739 | MeanDiff=-0.0227

  [TEST FULL F_delta_run_trend]
    n=118 | Acc=65.25% | Within1=99.15% | Severe(|d|>=2)=0.85% | MAE=0.3559 | RMSE=0.6106 | Penalty=0.3729 | MeanDiff=-0.0678

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         353     66.005666         1.133144          118      65.254237          99.152542          0.847458           0.372881              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         354     65.254237         1.129944          118      64.406780          99.152542          0.847458           0.381356              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              10               2.832861         90.000000             0.000000               2               1.694915        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.500000          2.000000                0.700000       999.000000            4.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.665722         85.000000             0.000000               4               3.389831         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.665722         85.000000             0.000000               4               3.389831         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.665722         85.000000             0.000000               4               3.389831         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091             0.000000               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091             0.000000               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091             0.000000               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091             0.000000               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB03_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB03_CHB2_1011_1229_parquet_slot_delta_prior.csv

[13/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB04_CHA1_1011_1229.parquet
loaded shape=(4078, 904)

==================================================================================================================================
Dataset: EPLBAB04_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(4078, 905), sort_time=0.018s
label out-of-range run_value=0/4078, policy=clip
split: train=[0,3262), cal=[3262,3874), test=[3874,4078)
split sizes: train=3262, cal=612, test=204

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           19   0.582465          1  0.163399           6  2.941176
    3          419  12.844880         88 14.379085          29 14.215686
    4         1694  51.931330        396 64.705882         116 56.862745
    5         1020  31.269160        123 20.098039          49 24.019608
    6          109   3.341508          3  0.490196           4  1.960784
    7            1   0.030656          1  0.163399           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.038116
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   3          6.500763        8.194325       3.098562      12.443308       9.344746        6.500763                 False
       6 259          1.249321        1.239709      -1.186956       3.814450       5.001407        1.249321                 False
       7   3          1.555834        0.319122      -2.264485       3.521086       5.785571        1.555834                 False
       8 281          0.251602        0.260904      -2.284811       2.842051       5.126862        0.251602                 False
      10 261          0.747513        0.911799      -1.223667       2.820671       4.044338        0.747513                 False
      11   3         -2.355032       -1.030115      -3.323857       0.601170       3.925027       -2.355032                 False
      14 260          0.520359        0.181511      -2.096044       2.182076       4.278119        0.520359                 False
      15   3          3.942270        1.442243      -0.393687       4.528187       4.921874        3.942270                 False
      16 283          0.073097        0.050213      -1.652580       1.730793       3.383373        0.073097                 False
      18 266         -0.882638       -0.828440      -3.152456       1.204303       4.356759       -0.882638                 False
      19   1          1.449205        1.449205       1.449205       1.449205       0.000000       -0.038116                  True
      24 266         -2.115662       -2.350878      -5.057612       0.052505       5.110117       -2.115662                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=901 -> miss=797 -> var=646 -> final=120
  C_loop_count: fit_time=2.18s pred_time=1.82s

  [CAL FULL C_loop_count]
    n=356 | Acc=63.48% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3652 | RMSE=0.6043 | Penalty=0.3652 | MeanDiff=0.0618

  [TEST FULL C_loop_count]
    n=120 | Acc=65.83% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3417 | RMSE=0.5845 | Penalty=0.3417 | MeanDiff=0.0417

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=931 -> miss=827 -> var=676 -> final=120
  F_delta_run_trend: fit_time=2.28s pred_time=1.78s

  [CAL FULL F_delta_run_trend]
    n=356 | Acc=71.91% | Within1=99.72% | Severe(|d|>=2)=0.28% | MAE=0.2837 | RMSE=0.5379 | Penalty=0.2893 | MeanDiff=0.0365

  [TEST FULL F_delta_run_trend]
    n=120 | Acc=69.17% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3083 | RMSE=0.5553 | Penalty=0.3083 | MeanDiff=0.0583

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         356     71.910112         0.280899          120      69.166667         100.000000          0.000000           0.308333              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         356     63.483146         0.000000          120      65.833333         100.000000          0.000000           0.341667              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              24               6.741573         83.333333             0.000000               9               7.500000        100.000000            100.000000             0.000000              0.000000                  True      0.100000               0.250000          1.500000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               3.932584         85.714286             0.000000               7               5.833333        100.000000            100.000000             0.000000              0.000000                  True      0.200000               0.500000          1.500000                0.500000         3.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               4.775281         82.352941             0.000000               7               5.833333        100.000000            100.000000             0.000000              0.000000                  True      0.200000               0.500000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.617978         85.000000             0.000000               1               0.833333        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.617978         85.000000             0.000000               1               0.833333        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              34               9.550562         82.352941             0.000000               9               7.500000         88.888889            100.000000             0.000000              0.111111                  True      0.400000               0.250000        999.000000                0.000000       999.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              21               5.898876         85.714286             0.000000               7               5.833333         71.428571            100.000000             0.000000              0.285714                  True      0.450000               0.250000        999.000000                0.500000         2.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              21               5.898876         85.714286             0.000000               7               5.833333         71.428571            100.000000             0.000000              0.285714                  True      0.450000               0.250000        999.000000                0.500000         2.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB04_CHA1_1011_1229_parquet_slot_delta_prior.csv

[14/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB04_CHA2_1011_1229.parquet
loaded shape=(4437, 886)

==================================================================================================================================
Dataset: EPLBAB04_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(4437, 887), sort_time=0.016s
label out-of-range run_value=0/4437, policy=clip
split: train=[0,3549), cal=[3549,4215), test=[4215,4437)
split sizes: train=3549, cal=666, test=222

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           26   0.732601          0  0.000000           0  0.000000
    3          529  14.905607         80 12.012012          22  9.909910
    4         1940  54.663285        399 59.909910         110 49.549550
    5          955  26.908988        178 26.726727          86 38.738739
    6           97   2.733164          9  1.351351           4  1.801802
    7            2   0.056354          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.359127
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 274         -1.231045       -0.947824      -3.206670       1.379364       4.586035       -1.231045                 False
       7 260          1.035240        0.754724      -1.381824       2.807724       4.189548        1.035240                 False
       8   3          2.324203        1.435890       0.229113       3.086824       2.857712        2.324203                 False
       9 281          0.387165        0.583683      -1.611820       2.718166       4.329987        0.387165                 False
      11 260         -0.111273       -0.003793      -2.100264       2.180434       4.280698       -0.111273                 False
      15 260         -0.400135       -0.481181      -2.809339       1.674078       4.483417       -0.400135                 False
      16   2          5.876085        5.876085       2.888535       8.863635       5.975100       -0.359127                  True
      17 283         -1.235870       -1.078601      -3.277010       1.125639       4.402649       -1.235870                 False
      19 267         -1.305805       -1.198806      -3.493063       0.796263       4.289326       -1.305805                 False
      25 267         -0.055607        0.116426      -2.264095       2.694664       4.958759       -0.055607                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=883 -> miss=799 -> var=653 -> final=120
  C_loop_count: fit_time=2.16s pred_time=1.87s

  [CAL FULL C_loop_count]
    n=409 | Acc=57.70% | Within1=99.51% | Severe(|d|>=2)=0.49% | MAE=0.4279 | RMSE=0.6616 | Penalty=0.4377 | MeanDiff=0.0073

  [TEST FULL C_loop_count]
    n=135 | Acc=55.56% | Within1=97.78% | Severe(|d|>=2)=2.22% | MAE=0.4667 | RMSE=0.7149 | Penalty=0.5111 | MeanDiff=-0.0519

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=913 -> miss=829 -> var=683 -> final=120
  F_delta_run_trend: fit_time=2.23s pred_time=1.89s

  [CAL FULL F_delta_run_trend]
    n=408 | Acc=61.52% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3848 | RMSE=0.6203 | Penalty=0.3848 | MeanDiff=-0.0515

  [TEST FULL F_delta_run_trend]
    n=135 | Acc=60.00% | Within1=99.26% | Severe(|d|>=2)=0.74% | MAE=0.4074 | RMSE=0.6498 | Penalty=0.4222 | MeanDiff=-0.0815

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         408     61.519608         0.000000          135      60.000000          99.259259          0.740741           0.422222              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         409     57.701711         0.488998          135      55.555556          97.777778          2.222222           0.511111              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.352941         76.666667             0.000000               9               6.666667        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.352941         76.666667             0.000000               9               6.666667        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.696078         81.818182             0.000000               2               1.481481        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.696078         81.818182             0.000000               2               1.481481        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              53              12.958435         77.358491             0.000000               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              48              11.735941         79.166667             0.000000               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              48              11.735941         79.166667             0.000000               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              53              12.958435         77.358491             0.000000               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB04_CHA2_1011_1229_parquet_slot_delta_prior.csv

[15/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB04_CHB1_1011_1229.parquet
loaded shape=(4552, 743)

==================================================================================================================================
Dataset: EPLBAB04_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(4552, 744), sort_time=0.015s
label out-of-range run_value=0/4552, policy=clip
split: train=[0,3641), cal=[3641,4324), test=[4324,4552)
split sizes: train=3641, cal=683, test=228

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           20   0.549300          2  0.292826           0  0.000000
    3          412  11.315573        106 15.519766          41 17.982456
    4         1738  47.734139        462 67.642753         119 52.192982
    5         1280  35.155177        111 16.251830          66 28.947368
    6          185   5.081022          2  0.292826           2  0.877193
    7            6   0.164790          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.189502
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1  10          2.167200        2.376428       1.433091       3.002142       1.569052        2.167200                 False
       6 308          0.936533        0.895186      -1.351954       3.107262       4.459216        0.936533                 False
       7   2         -0.158979       -0.158979      -0.579816       0.261857       0.841673       -0.189502                  True
       8 288          0.374352        0.361504      -1.942997       2.493821       4.436818        0.374352                 False
       9   5          2.029255        0.763520      -2.129200       3.851189       5.980389        2.029255                 False
      10 310          0.339693        0.518524      -1.644284       2.285862       3.930147        0.339693                 False
      11   2         -1.097866       -1.097866      -2.088205      -0.107527       1.980679       -0.189502                  True
      14 311         -0.611511       -0.516772      -2.875782       1.612355       4.488137       -0.611511                 False
      15   1         -0.791935       -0.791935      -0.791935      -0.791935       0.000000       -0.189502                  True
      16 289          0.110748        0.065541      -1.524544       1.533897       3.058441        0.110748                 False
      17   3         -2.056740        0.131961      -3.217245       2.386816       5.604061       -2.056740                 False
      18 305         -0.829880       -0.913350      -2.924330       1.322159       4.246489       -0.829880                 False
      19   2         -1.465769       -1.465769      -2.592673      -0.338864       2.253809       -0.189502                  True
      24 306         -1.551360       -1.772116      -4.300353       0.577097       4.877450       -1.551360                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=740 -> miss=696 -> var=550 -> final=120
  C_loop_count: fit_time=2.29s pred_time=1.97s

  [CAL FULL C_loop_count]
    n=397 | Acc=68.01% | Within1=99.50% | Severe(|d|>=2)=0.50% | MAE=0.3249 | RMSE=0.5788 | Penalty=0.3350 | MeanDiff=-0.0982

  [TEST FULL C_loop_count]
    n=136 | Acc=69.12% | Within1=99.26% | Severe(|d|>=2)=0.74% | MAE=0.3162 | RMSE=0.5752 | Penalty=0.3309 | MeanDiff=-0.1103

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=770 -> miss=726 -> var=580 -> final=120
  F_delta_run_trend: fit_time=2.24s pred_time=1.86s

  [CAL FULL F_delta_run_trend]
    n=397 | Acc=75.57% | Within1=99.75% | Severe(|d|>=2)=0.25% | MAE=0.2469 | RMSE=0.5019 | Penalty=0.2519 | MeanDiff=-0.0151

  [TEST FULL F_delta_run_trend]
    n=136 | Acc=72.79% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.2721 | RMSE=0.5216 | Penalty=0.2721 | MeanDiff=-0.0368

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         397     75.566751         0.251889          136      72.794118         100.000000          0.000000           0.272059              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         397     68.010076         0.503778          136      69.117647          99.264706          0.735294           0.330882              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               3.274559         92.307692             0.000000               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True      0.100000               0.250000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               4.030227         93.750000             0.000000               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               4.030227         93.750000             0.000000               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               3.274559         92.307692             0.000000               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True      0.100000               0.250000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.778338         93.333333             0.000000               8               5.882353         87.500000            100.000000             0.000000              0.125000                  True      0.200000               1.000000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.778338         93.333333             0.000000               8               5.882353         87.500000            100.000000             0.000000              0.125000                  True      0.200000               1.000000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              16               4.030227         93.750000             0.000000               7               5.147059         85.714286            100.000000             0.000000              0.142857                  True      0.000000               1.000000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              16               4.030227         93.750000             0.000000               7               5.147059         85.714286            100.000000             0.000000              0.142857                  True      0.000000               1.000000        999.000000                0.000000         1.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB04_CHB1_1011_1229_parquet_slot_delta_prior.csv

[16/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB04_CHB2_1011_1229.parquet
loaded shape=(4907, 745)

==================================================================================================================================
Dataset: EPLBAB04_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(4907, 746), sort_time=0.017s
label out-of-range run_value=0/4907, policy=clip
split: train=[0,3925), cal=[3925,4661), test=[4661,4907)
split sizes: train=3925, cal=736, test=246

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           44   1.121019          1  0.135870           1  0.406504
    3          722  18.394904         55  7.472826          19  7.723577
    4         2102  53.554140        420 57.065217         124 50.406504
    5          967  24.636943        255 34.646739          92 37.398374
    6           89   2.267516          5  0.679348           9  3.658537
    7            1   0.025478          0  0.000000           1  0.406504
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.280771
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 275          0.113480        0.306893      -1.859121       2.538967       4.398088        0.113480                 False
       6   5          0.279549       -0.601270      -1.834831       2.834961       4.669792        0.279549                 False
       7 307          0.328215        0.429720      -2.405743       2.982032       5.387774        0.328215                 False
       8   2          3.985415        3.985415       3.914786       4.056043       0.141256       -0.280771                  True
       9 288          0.625837        0.781608      -0.965732       2.517873       3.483605        0.625837                 False
      10   5         -0.129055        0.387715      -0.212208       1.747118       1.959326       -0.129055                 False
      11 310         -0.316230       -0.229969      -2.474710       1.996832       4.471541       -0.316230                 False
      14   4          0.312482       -0.596327      -2.645273       2.361428       5.006701        0.312482                 False
      15 311         -0.656515       -0.677781      -2.633593       1.270123       3.903716       -0.656515                 False
      16   1         -3.323463       -3.323463      -3.323463      -3.323463       0.000000       -0.280771                  True
      17 289         -1.111229       -0.945239      -3.160187       1.044468       4.204655       -1.111229                 False
      18   3         -0.507504        2.174918      -1.784077       4.792703       6.576779       -0.507504                 False
      19 305         -1.161346       -1.054375      -3.601797       1.085400       4.687197       -1.161346                 False
      24   1          1.672771        1.672771       1.672771       1.672771       0.000000       -0.280771                  True
      25 306         -0.190008       -0.170878      -2.954605       2.224028       5.178633       -0.190008                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=742 -> miss=717 -> var=567 -> final=120
  C_loop_count: fit_time=2.09s pred_time=1.87s

  [CAL FULL C_loop_count]
    n=451 | Acc=62.75% | Within1=99.78% | Severe(|d|>=2)=0.22% | MAE=0.3747 | RMSE=0.6158 | Penalty=0.3792 | MeanDiff=-0.0687

  [TEST FULL C_loop_count]
    n=154 | Acc=68.83% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3117 | RMSE=0.5583 | Penalty=0.3117 | MeanDiff=-0.0260

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=772 -> miss=747 -> var=597 -> final=120
  F_delta_run_trend: fit_time=2.12s pred_time=1.85s

  [CAL FULL F_delta_run_trend]
    n=451 | Acc=67.85% | Within1=99.78% | Severe(|d|>=2)=0.22% | MAE=0.3237 | RMSE=0.5729 | Penalty=0.3282 | MeanDiff=-0.0443

  [TEST FULL F_delta_run_trend]
    n=154 | Acc=64.94% | Within1=99.35% | Severe(|d|>=2)=0.65% | MAE=0.3571 | RMSE=0.6084 | Penalty=0.3701 | MeanDiff=0.0325

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         451     62.749446         0.221729          154      68.831169         100.000000          0.000000           0.311688              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         451     67.849224         0.221729          154      64.935065          99.350649          0.649351           0.370130              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               2.217295         90.000000             0.000000               2               1.298701        100.000000            100.000000             0.000000              0.000000                  True      0.450000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               2.217295         90.000000             0.000000               2               1.298701        100.000000            100.000000             0.000000              0.000000                  True      0.450000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593             0.000000               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593             0.000000               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593             0.000000               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593             0.000000               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              28               6.208426         82.142857             0.000000               6               3.896104         66.666667            100.000000             0.000000              0.333333                  True      0.300000               0.250000        999.000000                0.000000         4.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              28               6.208426         82.142857             0.000000               6               3.896104         66.666667            100.000000             0.000000              0.333333                  True      0.300000               0.250000        999.000000                0.000000         4.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB04_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB04_CHB2_1011_1229_parquet_slot_delta_prior.csv

[17/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB05_CHA1_1011_1229.parquet
loaded shape=(4067, 867)

==================================================================================================================================
Dataset: EPLBAB05_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(4067, 868), sort_time=0.019s
label out-of-range run_value=0/4067, policy=clip
split: train=[0,3253), cal=[3253,3863), test=[3863,4067)
split sizes: train=3253, cal=610, test=204

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           35   1.075930         18  2.950820           8  3.921569
    3          465  14.294497        168 27.540984          37 18.137255
    4         1676  51.521672        352 57.704918         127 62.254902
    5          976  30.003074         70 11.475410          32 15.686275
    6           99   3.043345          2  0.327869           0  0.000000
    7            2   0.061482          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.262118
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   6          1.442379        1.674093       0.936909       1.983824       1.046915        1.442379                 False
       6 257          0.032772        0.110012      -2.040356       2.155869       4.196224        0.032772                 False
       7   5          0.535625        0.526074      -0.039074       1.435726       1.474800        0.535625                 False
       8 275          0.164497        0.291929      -1.805344       2.541640       4.346984        0.164497                 False
       9   2          4.744200        4.744200       4.181569       5.306830       1.125261       -0.262118                  True
      10 261          0.333212        0.388846      -1.740032       2.633911       4.373943        0.333212                 False
      11   4         -2.006376       -2.312322      -2.583419      -1.735279       0.848140       -2.006376                 False
      14 259         -0.753872       -0.901449      -2.708798       1.252603       3.961401       -0.753872                 False
      15   3         -3.087641       -2.878911      -3.794128      -2.068059       1.726069       -3.087641                 False
      16 276          0.381441        0.172127      -1.821033       2.106175       3.927208        0.381441                 False
      17   2         -3.203765       -3.203765      -4.694809      -1.712720       2.982089       -0.262118                  True
      18 259         -1.291796       -1.400653      -3.344423       0.688361       4.032784       -1.291796                 False
      19   2         -0.096821       -0.096821      -0.133318      -0.060324       0.072994       -0.262118                  True
      24 283         -0.916981       -1.003428      -3.656212       1.786486       5.442698       -0.916981                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=864 -> miss=816 -> var=708 -> final=120
  C_loop_count: fit_time=2.21s pred_time=1.76s

  [CAL FULL C_loop_count]
    n=352 | Acc=64.77% | Within1=98.58% | Severe(|d|>=2)=1.42% | MAE=0.3665 | RMSE=0.6284 | Penalty=0.3949 | MeanDiff=0.0028

  [TEST FULL C_loop_count]
    n=118 | Acc=63.56% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3644 | RMSE=0.6037 | Penalty=0.3644 | MeanDiff=-0.0593

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=894 -> miss=846 -> var=738 -> final=120
  F_delta_run_trend: fit_time=2.18s pred_time=1.75s

  [CAL FULL F_delta_run_trend]
    n=352 | Acc=65.91% | Within1=99.15% | Severe(|d|>=2)=0.85% | MAE=0.3494 | RMSE=0.6054 | Penalty=0.3665 | MeanDiff=0.0199

  [TEST FULL F_delta_run_trend]
    n=118 | Acc=58.47% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.4153 | RMSE=0.6444 | Penalty=0.4153 | MeanDiff=0.0254

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         352     64.772727         1.420455          118      63.559322         100.000000          0.000000           0.364407              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         352     65.909091         0.852273          118      58.474576         100.000000          0.000000           0.415254              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend either            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               7.670455         96.296296             0.000000               9               7.627119         66.666667            100.000000             0.000000              0.333333                  True      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000
F_delta_run_trend delta_run_trend   slot            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               7.954545         96.428571             0.000000              11               9.322034         63.636364            100.000000             0.000000              0.363636                 False      0.300000               0.500000        999.000000                0.500000       999.000000            4.000000
F_delta_run_trend delta_run_trend  trend            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               7.670455         96.296296             0.000000               8               6.779661         62.500000            100.000000             0.000000              0.375000                  True      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000
F_delta_run_trend delta_run_trend   both            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               7.954545         96.428571             0.000000              10               8.474576         60.000000            100.000000             0.000000              0.400000                 False      0.300000               0.500000        999.000000                0.500000       999.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692             0.000000               5               4.237288         60.000000            100.000000             0.000000              0.400000                  True      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692             0.000000               5               4.237288         60.000000            100.000000             0.000000              0.400000                  True      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692             0.000000               4               3.389831         50.000000            100.000000             0.000000              0.500000                  True      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692             0.000000               4               3.389831         50.000000            100.000000             0.000000              0.500000                  True      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB05_CHA1_1011_1229_parquet_slot_delta_prior.csv

[18/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB05_CHA2_1011_1229.parquet
loaded shape=(4423, 852)

==================================================================================================================================
Dataset: EPLBAB05_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(4423, 853), sort_time=0.018s
label out-of-range run_value=0/4423, policy=clip
split: train=[0,3538), cal=[3538,4201), test=[4201,4423)
split sizes: train=3538, cal=663, test=222

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           45   1.271905          2  0.301659           2  0.900901
    3          670  18.937253         71 10.708899          29 13.063063
    4         1838  51.950254        359 54.147813          83 37.387387
    5          898  25.381572        218 32.880845         100 45.045045
    6           86   2.430752         13  1.960784           7  3.153153
    7            1   0.028265          0  0.000000           1  0.450450
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.156425
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 266          0.300701        0.375104      -2.101366       2.436875       4.538241        0.300701                 False
       6   3          2.736473        2.841993       2.508491       3.122736       0.614245        2.736473                 False
       7 258          0.772716        0.565099      -2.269446       3.466784       5.736231        0.772716                 False
       8   4         -4.450734       -4.000044      -5.033324      -3.417454       1.615870       -4.450734                 False
       9 277          0.359680        0.641673      -1.501415       2.558548       4.059963        0.359680                 False
      10   2          2.498467        2.498467      -0.152384       5.149319       5.301703       -0.156425                  True
      11 261         -0.428417       -0.369638      -2.497072       1.619343       4.116415       -0.428417                 False
      14   2          0.729616        0.729616       0.304495       1.154737       0.850243       -0.156425                  True
      15 259         -0.464561       -0.771706      -2.662879       1.303243       3.966122       -0.464561                 False
      16   2          0.307557        0.307557      -0.773110       1.388224       2.161334       -0.156425                  True
      17 278         -1.647394       -1.464975      -3.726870       0.819404       4.546274       -1.647394                 False
      18   1          0.096226        0.096226       0.096226       0.096226       0.000000       -0.156425                  True
      19 260         -1.128198       -1.181609      -3.472224       1.127010       4.599234       -1.128198                 False
      25 284          0.629845        0.736268      -2.054111       3.366724       5.420835        0.629845                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=849 -> miss=799 -> var=691 -> final=120
  C_loop_count: fit_time=2.19s pred_time=1.82s

  [CAL FULL C_loop_count]
    n=404 | Acc=59.65% | Within1=99.01% | Severe(|d|>=2)=0.99% | MAE=0.4158 | RMSE=0.6675 | Penalty=0.4455 | MeanDiff=-0.0594

  [TEST FULL C_loop_count]
    n=136 | Acc=60.29% | Within1=96.32% | Severe(|d|>=2)=3.68% | MAE=0.4412 | RMSE=0.7376 | Penalty=0.5441 | MeanDiff=-0.0147

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=879 -> miss=829 -> var=721 -> final=120
  F_delta_run_trend: fit_time=2.12s pred_time=1.70s

  [CAL FULL F_delta_run_trend]
    n=402 | Acc=68.16% | Within1=99.25% | Severe(|d|>=2)=0.75% | MAE=0.3284 | RMSE=0.5943 | Penalty=0.3532 | MeanDiff=-0.0348

  [TEST FULL F_delta_run_trend]
    n=136 | Acc=58.09% | Within1=97.79% | Severe(|d|>=2)=2.21% | MAE=0.4485 | RMSE=0.7225 | Penalty=0.5221 | MeanDiff=-0.0809

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         404     59.653465         0.990099          136      60.294118          96.323529          3.676471           0.544118              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         402     68.159204         0.746269          136      58.088235          97.794118          2.205882           0.522059              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692             0.000000               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692             0.000000               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692             0.000000               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692             0.000000               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000
F_delta_run_trend delta_run_trend   slot            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               4.975124         95.000000             0.000000              10               7.352941         50.000000            100.000000             0.000000              0.500000                 False      0.400000               0.500000          2.000000                0.500000       999.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               5.721393         91.304348             0.000000              10               7.352941         50.000000            100.000000             0.000000              0.500000                 False      0.400000               0.500000        999.000000                0.500000         4.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               3.980100         93.750000             0.000000               7               5.147059         42.857143            100.000000             0.000000              0.571429                  True      0.400000               0.500000        999.000000                0.000000       999.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               3.980100         93.750000             0.000000               7               5.147059         42.857143            100.000000             0.000000              0.571429                  True      0.400000               0.500000        999.000000                0.000000       999.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB05_CHA2_1011_1229_parquet_slot_delta_prior.csv

[19/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB05_CHB1_1011_1229.parquet
loaded shape=(4706, 741)

==================================================================================================================================
Dataset: EPLBAB05_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(4706, 742), sort_time=0.014s
label out-of-range run_value=0/4706, policy=clip
split: train=[0,3764), cal=[3764,4470), test=[4470,4706)
split sizes: train=3764, cal=706, test=236

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2            6   0.159405          1  0.141643           0  0.000000
    3          193   5.127524         23  3.257790          35 14.830508
    4         1366  36.291180        320 45.325779         101 42.796610
    5         1833  48.698193        325 46.033994          89 37.711864
    6          356   9.458023         37  5.240793          11  4.661017
    7           10   0.265675          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.229448
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   6          3.107597        3.136391       1.715625       5.413529       3.697905        3.107597                 False
       6 316          0.232283        0.334268      -1.778017       2.532590       4.310606        0.232283                 False
       7   4         -0.298755       -0.631516      -3.262291       2.332020       5.594311       -0.298755                 False
       8 301          0.221851        0.132503      -1.757549       2.204393       3.961943        0.221851                 False
       9   5          1.428902        0.515034      -1.609089       1.701767       3.310856        1.428902                 False
      10 317          0.530109        0.513043      -1.794651       2.648148       4.442799        0.530109                 False
      11   3         -0.147415       -0.885893      -1.791513       0.388967       2.180480       -0.147415                 False
      14 320         -1.064018       -1.133033      -3.149304       0.774771       3.924075       -1.064018                 False
      15   2          0.235013        0.235013      -0.607054       1.077080       1.684134       -0.229448                  True
      16 304          0.493788        0.293859      -1.668862       1.911124       3.579987        0.493788                 False
      17   2         -0.981741       -0.981741      -2.606009       0.642528       3.248537       -0.229448                  True
      18 321         -1.504436       -1.186884      -3.275963       0.942898       4.218861       -1.504436                 False
      19   1          1.178984        1.178984       1.178984       1.178984       0.000000       -0.229448                  True
      24 300         -1.014338       -1.171033      -3.493863       1.258488       4.752351       -1.014338                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=738 -> miss=697 -> var=567 -> final=120
  C_loop_count: fit_time=2.15s pred_time=1.88s

  [CAL FULL C_loop_count]
    n=418 | Acc=62.20% | Within1=98.56% | Severe(|d|>=2)=1.44% | MAE=0.3923 | RMSE=0.6489 | Penalty=0.4211 | MeanDiff=-0.0478

  [TEST FULL C_loop_count]
    n=142 | Acc=59.86% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.4014 | RMSE=0.6336 | Penalty=0.4014 | MeanDiff=0.1197

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=768 -> miss=727 -> var=597 -> final=120
  F_delta_run_trend: fit_time=2.22s pred_time=1.86s

  [CAL FULL F_delta_run_trend]
    n=418 | Acc=64.11% | Within1=98.33% | Severe(|d|>=2)=1.67% | MAE=0.3780 | RMSE=0.6489 | Penalty=0.4211 | MeanDiff=-0.0287

  [TEST FULL F_delta_run_trend]
    n=137 | Acc=63.50% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3650 | RMSE=0.6041 | Penalty=0.3650 | MeanDiff=0.1314

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         418     64.114833         1.674641          137      63.503650         100.000000          0.000000           0.364964              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         418     62.200957         1.435407          142      59.859155         100.000000          0.000000           0.401408              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32               7.655502         87.500000             0.000000              11               8.029197         63.636364            100.000000             0.000000              0.363636                 False      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32               7.655502         87.500000             0.000000              11               8.029197         63.636364            100.000000             0.000000              0.363636                 False      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               4.066986         88.235294             0.000000               5               3.649635         60.000000            100.000000             0.000000              0.400000                  True      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               4.066986         88.235294             0.000000               5               3.649635         60.000000            100.000000             0.000000              0.400000                  True      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000
     C_loop_count            loop   slot            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop  trend            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.200000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop   both            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.200000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop either            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB05_CHB1_1011_1229_parquet_slot_delta_prior.csv

[20/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB05_CHB2_1011_1229.parquet
loaded shape=(5078, 763)

==================================================================================================================================
Dataset: EPLBAB05_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(5078, 764), sort_time=0.015s
label out-of-range run_value=0/5078, policy=clip
split: train=[0,4062), cal=[4062,4824), test=[4824,5078)
split sizes: train=4062, cal=762, test=254

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2          143   3.520433         12  1.574803          11  4.330709
    3         1348  33.185623        164 21.522310          93 36.614173
    4         2055  50.590842        452 59.317585         113 44.488189
    5          495  12.186115        129 16.929134          37 14.566929
    6           21   0.516987          5  0.656168           0  0.000000
    7            0   0.000000          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.403784
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 289         -0.497334       -0.360405      -2.828863       1.922863       4.751726       -0.497334                 False
       6   6          0.033338       -0.291507      -2.075257       2.121189       4.196446        0.033338                 False
       7 315          0.308575        0.235416      -2.135242       2.629890       4.765133        0.308575                 False
       8   4          0.856258        0.219381      -1.727263       2.802903       4.530166        0.856258                 False
       9 301          0.465668        0.501665      -1.223902       2.248260       3.472162        0.465668                 False
      10   5          1.661327        1.693674       0.689346       2.482574       1.793228        1.661327                 False
      11 316         -0.798165       -0.877128      -3.091280       1.302896       4.394176       -0.798165                 False
      14   4         -1.097242        0.821234      -4.528313       4.252305       8.780618       -1.097242                 False
      15 320         -0.978806       -1.036655      -3.138671       1.181539       4.320210       -0.978806                 False
      16   2          2.688998        2.688998       1.797410       3.580586       1.783175       -0.403784                  True
      17 304         -0.776551       -0.995633      -3.278011       1.370443       4.648454       -0.776551                 False
      18   2         -3.316333       -3.316333      -3.868963      -2.763703       1.105260       -0.403784                  True
      19 321         -1.109428       -1.096013      -3.312164       0.998215       4.310379       -1.109428                 False
      25 301         -0.005112        0.428521      -1.996222       3.225151       5.221373       -0.005112                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=760 -> miss=698 -> var=569 -> final=120
  C_loop_count: fit_time=2.06s pred_time=1.89s

  [CAL FULL C_loop_count]
    n=471 | Acc=62.00% | Within1=97.66% | Severe(|d|>=2)=2.34% | MAE=0.4055 | RMSE=0.6788 | Penalty=0.4607 | MeanDiff=-0.0021

  [TEST FULL C_loop_count]
    n=158 | Acc=65.19% | Within1=99.37% | Severe(|d|>=2)=0.63% | MAE=0.3544 | RMSE=0.6059 | Penalty=0.3671 | MeanDiff=-0.0759

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=790 -> miss=728 -> var=599 -> final=120
  F_delta_run_trend: fit_time=2.05s pred_time=1.88s

  [CAL FULL F_delta_run_trend]
    n=468 | Acc=64.96% | Within1=98.29% | Severe(|d|>=2)=1.71% | MAE=0.3675 | RMSE=0.6338 | Penalty=0.4017 | MeanDiff=-0.0171

  [TEST FULL F_delta_run_trend]
    n=158 | Acc=62.66% | Within1=99.37% | Severe(|d|>=2)=0.63% | MAE=0.3797 | RMSE=0.6264 | Penalty=0.3924 | MeanDiff=-0.0633

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         471     61.995754         2.335456          158      65.189873          99.367089          0.632911           0.367089              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         468     64.957265         1.709402          158      62.658228          99.367089          0.632911           0.392405              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               2.972399         85.714286             0.000000               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.184713         86.666667             0.000000               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               2.972399         85.714286             0.000000               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.184713         86.666667             0.000000               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               5.982906         89.285714             0.000000               9               5.696203         77.777778            100.000000             0.000000              0.222222                  True      0.400000               0.500000        999.000000                0.700000       999.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               5.982906         89.285714             0.000000               9               5.696203         77.777778            100.000000             0.000000              0.222222                  True      0.400000               0.500000        999.000000                0.700000       999.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              29               6.196581         89.655172             0.000000              11               6.962025         72.727273            100.000000             0.000000              0.272727                 False      0.400000               0.250000        999.000000                0.700000       999.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              29               6.196581         89.655172             0.000000              11               6.962025         72.727273            100.000000             0.000000              0.272727                 False      0.400000               0.250000        999.000000                0.700000       999.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB05_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB05_CHB2_1011_1229_parquet_slot_delta_prior.csv

[21/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB06_CHA1_1011_1229.parquet
loaded shape=(3890, 787)

==================================================================================================================================
Dataset: EPLBAB06_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(3890, 788), sort_time=0.014s
label out-of-range run_value=0/3890, policy=clip
split: train=[0,3112), cal=[3112,3695), test=[3695,3890)
split sizes: train=3112, cal=583, test=195

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           74   2.377892         17  2.915952           4  2.051282
    3          720  23.136247        139 23.842196          62 31.794872
    4         1652  53.084833        342 58.662093          93 47.692308
    5          608  19.537275         83 14.236707          36 18.461538
    6           58   1.863753          2  0.343053           0  0.000000
    7            0   0.000000          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.266069
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   4          2.128649        2.127326       1.598586       2.657389       1.058804        2.128649                 False
       6 238          0.333867        0.626000      -1.605158       2.947108       4.552266        0.333867                 False
       7   2         -3.863987       -3.863987      -5.293634      -2.434340       2.859295       -0.266069                  True
       8 278          0.359856        0.440289      -1.689538       2.772382       4.461920        0.359856                 False
       9   1          3.937824        3.937824       3.937824       3.937824       0.000000       -0.266069                  True
      10 238          0.700881        0.851440      -1.385581       2.981275       4.366856        0.700881                 False
      11   2         -1.440273       -1.440273      -4.116177       1.235630       5.351807       -0.266069                  True
      14 237         -0.798676       -0.942879      -3.151962       0.971077       4.123039       -0.798676                 False
      15   1         -2.407475       -2.407475      -2.407475      -2.407475       0.000000       -0.266069                  True
      16 282          0.226180        0.208287      -1.603910       2.056073       3.659983        0.226180                 False
      17   1         -1.112740       -1.112740      -1.112740      -1.112740       0.000000       -0.266069                  True
      18 233         -1.131130       -1.189360      -3.141010       0.731911       3.872921       -1.131130                 False
      19   1         -7.574776       -7.574776      -7.574776      -7.574776       0.000000       -0.266069                  True
      24 278         -1.534885       -1.751104      -4.083788       0.525182       4.608970       -1.534885                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=784 -> miss=740 -> var=592 -> final=120
  C_loop_count: fit_time=2.30s pred_time=1.79s

  [CAL FULL C_loop_count]
    n=342 | Acc=67.25% | Within1=98.83% | Severe(|d|>=2)=1.17% | MAE=0.3392 | RMSE=0.6021 | Penalty=0.3626 | MeanDiff=-0.0058

  [TEST FULL C_loop_count]
    n=112 | Acc=67.86% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3214 | RMSE=0.5669 | Penalty=0.3214 | MeanDiff=-0.0536

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=814 -> miss=770 -> var=622 -> final=120
  F_delta_run_trend: fit_time=2.19s pred_time=1.79s

  [CAL FULL F_delta_run_trend]
    n=342 | Acc=71.35% | Within1=99.71% | Severe(|d|>=2)=0.29% | MAE=0.2895 | RMSE=0.5434 | Penalty=0.2953 | MeanDiff=-0.0439

  [TEST FULL F_delta_run_trend]
    n=112 | Acc=67.86% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3214 | RMSE=0.5669 | Penalty=0.3214 | MeanDiff=-0.1429

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         342     67.251462         1.169591          112      67.857143         100.000000          0.000000           0.321429              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         342     71.345029         0.292398          112      67.857143         100.000000          0.000000           0.321429              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.847953         90.000000             5.000000              11               9.821429         90.909091            100.000000             0.000000              0.090909                 False      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.847953         90.000000             5.000000              11               9.821429         90.909091            100.000000             0.000000              0.090909                 False      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              24               7.017544         91.666667             4.166667              12              10.714286         83.333333            100.000000             0.000000              0.166667                 False      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              24               7.017544         91.666667             4.166667              12              10.714286         83.333333            100.000000             0.000000              0.166667                 False      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   slot            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.801170        100.000000             0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop  trend            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.216374        100.000000             0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.216374        100.000000             0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop either            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.801170        100.000000             0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB06_CHA1_1011_1229_parquet_slot_delta_prior.csv

[22/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB06_CHA2_1011_1229.parquet
loaded shape=(4236, 749)

==================================================================================================================================
Dataset: EPLBAB06_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(4236, 750), sort_time=0.015s
label out-of-range run_value=0/4236, policy=clip
split: train=[0,3388), cal=[3388,4024), test=[4024,4236)
split sizes: train=3388, cal=636, test=212

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           20   0.590319          0  0.000000           0  0.000000
    3          358  10.566706         59  9.276730          26 12.264151
    4         1642  48.465171        297 46.698113         115 54.245283
    5         1184  34.946871        258 40.566038          67 31.603774
    6          175   5.165289         22  3.459119           4  1.886792
    7            9   0.265643          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.414066
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 267         -0.259216       -0.310740      -2.916021       1.891197       4.807219       -0.259216                 False
       6   1         -2.329117       -2.329117      -2.329117      -2.329117       0.000000       -0.414066                  True
       7 238          0.443747        0.371369      -2.090796       2.678655       4.769451        0.443747                 False
       8   2          3.729280        3.729280       3.648484       3.810077       0.161592       -0.414066                  True
       9 278          0.524006        0.760337      -1.353666       2.771220       4.124887        0.524006                 False
      10   1         -2.596233       -2.596233      -2.596233      -2.596233       0.000000       -0.414066                  True
      11 238         -0.532242       -0.458044      -2.748658       1.667452       4.416111       -0.532242                 False
      14   1         -3.258068       -3.258068      -3.258068      -3.258068       0.000000       -0.414066                  True
      15 237         -1.069332       -0.937694      -3.187492       1.328665       4.516157       -1.069332                 False
      16   1          2.732170        2.732170       2.732170       2.732170       0.000000       -0.414066                  True
      17 281         -1.373989       -1.585579      -3.752987       0.732662       4.485649       -1.373989                 False
      18   1         -4.167667       -4.167667      -4.167667      -4.167667       0.000000       -0.414066                  True
      19 232         -1.661964       -1.710328      -4.077057       0.439432       4.516490       -1.661964                 False
      24   1         -9.623613       -9.623613      -9.623613      -9.623613       0.000000       -0.414066                  True
      25 277          0.360264        0.412809      -1.549877       2.981739       4.531616        0.360264                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=746 -> miss=704 -> var=555 -> final=120
  C_loop_count: fit_time=2.12s pred_time=1.81s

  [CAL FULL C_loop_count]
    n=390 | Acc=58.21% | Within1=98.97% | Severe(|d|>=2)=1.03% | MAE=0.4282 | RMSE=0.6699 | Penalty=0.4487 | MeanDiff=0.0077

  [TEST FULL C_loop_count]
    n=128 | Acc=71.09% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.2891 | RMSE=0.5376 | Penalty=0.2891 | MeanDiff=-0.0234

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=776 -> miss=734 -> var=585 -> final=120
  F_delta_run_trend: fit_time=2.24s pred_time=1.83s

  [CAL FULL F_delta_run_trend]
    n=390 | Acc=62.82% | Within1=99.74% | Severe(|d|>=2)=0.26% | MAE=0.3744 | RMSE=0.6160 | Penalty=0.3795 | MeanDiff=-0.0410

  [TEST FULL F_delta_run_trend]
    n=128 | Acc=68.75% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3125 | RMSE=0.5590 | Penalty=0.3125 | MeanDiff=0.0000

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         390     58.205128         1.025641          128      71.093750         100.000000          0.000000           0.289062              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         390     62.820513         0.256410          128      68.750000         100.000000          0.000000           0.312500              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              21               5.384615         95.238095             0.000000               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop  trend            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.128205         95.000000             0.000000               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              17               4.358974        100.000000             0.000000               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop either            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              22               5.641026         95.454545             0.000000               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.500000         1.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               5.897436         91.304348             0.000000               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True      0.300000               0.250000          1.500000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.128205         85.000000             0.000000               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.128205         85.000000             0.000000               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               6.666667         88.461538             0.000000               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB06_CHA2_1011_1229_parquet_slot_delta_prior.csv

[23/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB06_CHB1_1011_1229.parquet
loaded shape=(3994, 741)

==================================================================================================================================
Dataset: EPLBAB06_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(3994, 742), sort_time=0.015s
label out-of-range run_value=0/3994, policy=clip
split: train=[0,3195), cal=[3195,3794), test=[3794,3994)
split sizes: train=3195, cal=599, test=200

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           39   1.220657          2  0.333890           0  0.000000
    3          572  17.902973         78 13.021703          32 16.000000
    4         1571  49.170579        358 59.766277         120 60.000000
    5          898  28.106416        151 25.208681          46 23.000000
    6          105   3.286385         10  1.669449           2  1.000000
    7            9   0.281690          0  0.000000           0  0.000000
    8            1   0.031299          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.014395
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   7          1.534363        2.977764      -1.106943       6.302807       7.409750        1.534363                 False
       6 283          0.813494        0.606656      -1.969785       2.913202       4.882987        0.813494                 False
       7   1         -1.033020       -1.033020      -1.033020      -1.033020       0.000000       -0.014395                  True
       8 245          0.727381        0.584176      -1.570360       2.701431       4.271791        0.727381                 False
       9   2          4.405291        4.405291       4.399014       4.411567       0.012552       -0.014395                  True
      10 285          0.684351        0.860651      -1.346008       2.888969       4.234978        0.684351                 False
      11   1         -8.314821       -8.314821      -8.314821      -8.314821       0.000000       -0.014395                  True
      14 285         -0.597103       -0.382594      -2.363426       1.445286       3.808712       -0.597103                 False
      15   1         -2.554947       -2.554947      -2.554947      -2.554947       0.000000       -0.014395                  True
      16 243          0.234577        0.325383      -1.746554       2.504686       4.251241        0.234577                 False
      17   1         -7.903833       -7.903833      -7.903833      -7.903833       0.000000       -0.014395                  True
      18 290         -0.692719       -0.958992      -3.269979       1.250518       4.520497       -0.692719                 False
      19   1         -9.198797       -9.198797      -9.198797      -9.198797       0.000000       -0.014395                  True
      24 247         -1.083282       -0.949498      -3.273870       1.470446       4.744316       -1.083282                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=738 -> miss=734 -> var=596 -> final=120
  C_loop_count: fit_time=2.10s pred_time=1.81s

  [CAL FULL C_loop_count]
    n=352 | Acc=66.19% | Within1=99.43% | Severe(|d|>=2)=0.57% | MAE=0.3438 | RMSE=0.5959 | Penalty=0.3551 | MeanDiff=-0.0426

  [TEST FULL C_loop_count]
    n=117 | Acc=71.79% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.2821 | RMSE=0.5311 | Penalty=0.2821 | MeanDiff=-0.0427

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=768 -> miss=764 -> var=626 -> final=120
  F_delta_run_trend: fit_time=2.07s pred_time=1.89s

  [CAL FULL F_delta_run_trend]
    n=352 | Acc=69.60% | Within1=99.72% | Severe(|d|>=2)=0.28% | MAE=0.3068 | RMSE=0.5590 | Penalty=0.3125 | MeanDiff=-0.0114

  [TEST FULL F_delta_run_trend]
    n=117 | Acc=72.65% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.2735 | RMSE=0.5230 | Penalty=0.2735 | MeanDiff=-0.1026

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         352     69.602273         0.284091          117      72.649573         100.000000          0.000000           0.273504              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         352     66.193182         0.568182          117      71.794872         100.000000          0.000000           0.282051              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091             0.000000               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091             0.000000               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091             0.000000               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091             0.000000               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               5.397727         89.473684             0.000000               2               1.709402        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               5.397727         89.473684             0.000000               2               1.709402        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              27               7.670455         88.888889             0.000000               6               5.128205         83.333333            100.000000             0.000000              0.166667                  True      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              27               7.670455         88.888889             0.000000               6               5.128205         83.333333            100.000000             0.000000              0.166667                  True      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB06_CHB1_1011_1229_parquet_slot_delta_prior.csv

[24/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB06_CHB2_1011_1229.parquet
loaded shape=(4309, 743)

==================================================================================================================================
Dataset: EPLBAB06_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(4309, 744), sort_time=0.012s
label out-of-range run_value=0/4309, policy=clip
split: train=[0,3447), cal=[3447,4093), test=[4093,4309)
split sizes: train=3447, cal=646, test=216

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           61   1.769655          5  0.773994           0  0.000000
    3          537  15.578764        110 17.027864          21  9.722222
    4         1698  49.260226        367 56.811146         128 59.259259
    5         1059  30.722367        156 24.148607          66 30.555556
    6           88   2.552945          8  1.238390           1  0.462963
    7            4   0.116043          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.312033
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 237         -0.119816        0.344717      -1.998751       2.435364       4.434114       -0.119816                 False
       6   2         -2.731476       -2.731476      -4.501431      -0.961521       3.539909       -0.312033                  True
       7 283          0.247532        0.506832      -1.818794       2.773542       4.592337        0.247532                 False
       8   1          0.884209        0.884209       0.884209       0.884209       0.000000       -0.312033                  True
       9 245          0.807800        0.838377      -1.287876       2.835949       4.123825        0.807800                 False
      10   2         -3.124220       -3.124220      -3.269341      -2.979098       0.290243       -0.312033                  True
      11 285         -0.254131       -0.170179      -2.454655       1.870277       4.324932       -0.254131                 False
      14   1         -1.367252       -1.367252      -1.367252      -1.367252       0.000000       -0.312033                  True
      15 287         -0.981218       -1.151201      -3.378345       0.998316       4.376660       -0.981218                 False
      16   1          5.876610        5.876610       5.876610       5.876610       0.000000       -0.312033                  True
      17 243         -0.797108       -1.240724      -3.177153       0.633797       3.810949       -0.797108                 False
      18   1          1.616001        1.616001       1.616001       1.616001       0.000000       -0.312033                  True
      19 290         -1.172125       -1.186376      -3.569704       1.213702       4.783406       -1.172125                 False
      25 247          0.035969        0.050149      -2.044947       2.025625       4.070572        0.035969                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=740 -> miss=736 -> var=597 -> final=120
  C_loop_count: fit_time=2.15s pred_time=1.84s

  [CAL FULL C_loop_count]
    n=400 | Acc=61.50% | Within1=99.00% | Severe(|d|>=2)=1.00% | MAE=0.3950 | RMSE=0.6442 | Penalty=0.4150 | MeanDiff=-0.0500

  [TEST FULL C_loop_count]
    n=133 | Acc=64.66% | Within1=99.25% | Severe(|d|>=2)=0.75% | MAE=0.3609 | RMSE=0.6131 | Penalty=0.3759 | MeanDiff=0.1203

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=770 -> miss=766 -> var=627 -> final=120
  F_delta_run_trend: fit_time=2.21s pred_time=1.83s

  [CAL FULL F_delta_run_trend]
    n=400 | Acc=66.00% | Within1=99.75% | Severe(|d|>=2)=0.25% | MAE=0.3425 | RMSE=0.5895 | Penalty=0.3475 | MeanDiff=-0.0675

  [TEST FULL F_delta_run_trend]
    n=133 | Acc=66.17% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3383 | RMSE=0.5817 | Penalty=0.3383 | MeanDiff=0.0226

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         400     66.000000         0.250000          133      66.165414         100.000000          0.000000           0.338346              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         400     61.500000         1.000000          133      64.661654          99.248120          0.751880           0.375940              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.500000         78.571429             0.000000               3               2.255639        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.250000         84.615385             0.000000               2               1.503759        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.700000       999.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.250000         84.615385             0.000000               2               1.503759        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.750000         90.909091             0.000000               1               0.751880        100.000000            100.000000             0.000000              0.000000                  True      0.000000               0.500000        999.000000                0.500000         3.000000            4.000000
F_delta_run_trend delta_run_trend   slot            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000             0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000
F_delta_run_trend delta_run_trend  trend            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000             0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000
F_delta_run_trend delta_run_trend   both            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000             0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000
F_delta_run_trend delta_run_trend either            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000             0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB06_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB06_CHB2_1011_1229_parquet_slot_delta_prior.csv

[25/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB07_CHA1_1011_1229.parquet
loaded shape=(4691, 869)

==================================================================================================================================
Dataset: EPLBAB07_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(4691, 870), sort_time=0.016s
label out-of-range run_value=0/4691, policy=clip
split: train=[0,3752), cal=[3752,4456), test=[4456,4691)
split sizes: train=3752, cal=704, test=235

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2            9   0.239872          0  0.000000           0  0.000000
    3          273   7.276119         23  3.267045          20  8.510638
    4         1561  41.604478        244 34.659091          88 37.446809
    5         1596  42.537313        376 53.409091         103 43.829787
    6          292   7.782516         58  8.238636          23  9.787234
    7           21   0.559701          3  0.426136           1  0.425532
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.034515
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   7         -1.349060       -1.706664      -3.413937       0.906725       4.320662       -1.349060                 False
       6 299          0.296284        0.448589      -1.544212       2.569096       4.113308        0.296284                 False
       7   2         -1.279043       -1.279043      -2.515376      -0.042710       2.472666       -0.034515                  True
       8 314          0.534898        0.609951      -1.566334       2.354970       3.921304        0.534898                 False
       9   4          3.613474        3.932218       2.872572       4.673120       1.800548        3.613474                 False
      10 300          0.858700        0.620625      -1.720122       2.868315       4.588437        0.858700                 False
      11   2         -0.015921       -0.015921      -2.288468       2.256627       4.545095       -0.034515                  True
      14 295         -0.546577       -0.478318      -2.428986       1.289558       3.718544       -0.546577                 False
      15   1         -0.563927       -0.563927      -0.563927      -0.563927       0.000000       -0.034515                  True
      16 319          0.342712        0.278480      -1.724951       2.313019       4.037970        0.342712                 False
      17   3         -0.941620       -0.607582      -1.069046      -0.313137       0.755909       -0.941620                 False
      18 300         -0.821468       -0.820541      -2.841883       1.370923       4.212806       -0.821468                 False
      19   1         -1.386890       -1.386890      -1.386890      -1.386890       0.000000       -0.034515                  True
      24 332         -0.792965       -0.721039      -3.444979       1.964726       5.409705       -0.792965                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=866 -> miss=797 -> var=649 -> final=120
  C_loop_count: fit_time=2.08s pred_time=1.83s

  [CAL FULL C_loop_count]
    n=411 | Acc=66.67% | Within1=99.76% | Severe(|d|>=2)=0.24% | MAE=0.3358 | RMSE=0.5836 | Penalty=0.3406 | MeanDiff=0.0535

  [TEST FULL C_loop_count]
    n=140 | Acc=61.43% | Within1=97.86% | Severe(|d|>=2)=2.14% | MAE=0.4071 | RMSE=0.6708 | Penalty=0.4500 | MeanDiff=-0.0071

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=896 -> miss=827 -> var=679 -> final=120
  F_delta_run_trend: fit_time=2.24s pred_time=1.87s

  [CAL FULL F_delta_run_trend]
    n=410 | Acc=70.49% | Within1=99.76% | Severe(|d|>=2)=0.24% | MAE=0.2976 | RMSE=0.5499 | Penalty=0.3024 | MeanDiff=0.0098

  [TEST FULL F_delta_run_trend]
    n=139 | Acc=66.19% | Within1=97.84% | Severe(|d|>=2)=2.16% | MAE=0.3597 | RMSE=0.6347 | Penalty=0.4029 | MeanDiff=-0.1151

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         410     70.487805         0.243902          139      66.187050          97.841727          2.158273           0.402878              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         411     66.666667         0.243309          140      61.428571          97.857143          2.142857           0.450000              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              38               9.268293         78.947368             0.000000               9               6.474820         88.888889            100.000000             0.000000              0.111111                  True      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              38               9.268293         78.947368             0.000000               9               6.474820         88.888889            100.000000             0.000000              0.111111                  True      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              21               5.121951         80.952381             0.000000              13               9.352518         76.923077            100.000000             0.000000              0.230769                 False      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              21               5.121951         80.952381             0.000000              13               9.352518         76.923077            100.000000             0.000000              0.230769                 False      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667             0.000000               6               4.285714         50.000000            100.000000             0.000000              0.500000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667             0.000000               6               4.285714         50.000000            100.000000             0.000000              0.500000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667             0.000000               4               2.857143         50.000000            100.000000             0.000000              0.500000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667             0.000000               4               2.857143         50.000000            100.000000             0.000000              0.500000                  True      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB07_CHA1_1011_1229_parquet_slot_delta_prior.csv

[26/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB07_CHA2_1011_1229.parquet
loaded shape=(5082, 851)

==================================================================================================================================
Dataset: EPLBAB07_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(5082, 852), sort_time=0.019s
label out-of-range run_value=0/5082, policy=clip
split: train=[0,4065), cal=[4065,4827), test=[4827,5082)
split sizes: train=4065, cal=762, test=255

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2          149   3.665437         42  5.511811          29 11.372549
    3         1258  30.947109        333 43.700787         128 50.196078
    4         2073  50.996310        339 44.488189          84 32.941176
    5          542  13.333333         47  6.167979          13  5.098039
    6           40   0.984010          1  0.131234           1  0.392157
    7            3   0.073801          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.381643
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 303          0.306686        0.084336      -2.013453       1.967751       3.981203        0.306686                 False
       6   4          2.222315        1.535524       1.141408       2.616431       1.475023        2.222315                 False
       7 299         -0.351080       -0.071471      -2.270371       2.278445       4.548817       -0.351080                 False
       8   2         -4.606516       -4.606516      -5.360449      -3.852583       1.507866       -0.381643                  True
       9 313          0.222374        0.523699      -1.536163       2.566685       4.102848        0.222374                 False
      10   4          2.781343        2.710716       0.217086       5.274973       5.057887        2.781343                 False
      11 299         -0.231552       -0.420779      -2.651464       2.061651       4.713115       -0.231552                 False
      14   3         -1.846134       -2.394593      -3.152501      -1.362455       1.790046       -1.846134                 False
      15 295         -0.870495       -1.039850      -3.031683       0.954067       3.985750       -0.870495                 False
      16   1         -8.565262       -8.565262      -8.565262      -8.565262       0.000000       -0.381643                  True
      17 319         -1.321110       -1.111091      -3.385086       1.200917       4.586003       -1.321110                 False
      18   3         -2.086956       -0.280142      -3.424977       1.961287       5.386264       -2.086956                 False
      19 300         -1.336102       -1.157891      -3.384875       1.278768       4.663643       -1.336102                 False
      25 331          0.930199        1.066630      -1.479160       4.064395       5.543555        0.930199                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=848 -> miss=799 -> var=649 -> final=120
  C_loop_count: fit_time=2.16s pred_time=1.90s

  [CAL FULL C_loop_count]
    n=469 | Acc=58.42% | Within1=98.72% | Severe(|d|>=2)=1.28% | MAE=0.4286 | RMSE=0.6739 | Penalty=0.4542 | MeanDiff=0.0235

  [TEST FULL C_loop_count]
    n=158 | Acc=69.62% | Within1=98.73% | Severe(|d|>=2)=1.27% | MAE=0.3165 | RMSE=0.5846 | Penalty=0.3418 | MeanDiff=0.0633

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=878 -> miss=829 -> var=679 -> final=120
  F_delta_run_trend: fit_time=2.19s pred_time=1.91s

  [CAL FULL F_delta_run_trend]
    n=465 | Acc=60.22% | Within1=99.78% | Severe(|d|>=2)=0.22% | MAE=0.4000 | RMSE=0.6358 | Penalty=0.4043 | MeanDiff=-0.0172

  [TEST FULL F_delta_run_trend]
    n=156 | Acc=67.31% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3269 | RMSE=0.5718 | Penalty=0.3269 | MeanDiff=0.0321

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         469     58.422175         1.279318          158      69.620253          98.734177          1.265823           0.341772              120
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         465     60.215054         0.215054          156      67.307692         100.000000          0.000000           0.326923              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               3.225806         80.000000             0.000000               9               5.769231         77.777778            100.000000             0.000000              0.222222                  True      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              25               5.376344         80.000000             0.000000              20              12.820513         75.000000            100.000000             0.000000              0.250000                 False      0.100000               0.250000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.345416         90.909091             0.000000               3               1.898734         66.666667            100.000000             0.000000              0.333333                  True      0.400000               0.750000        999.000000                0.700000       999.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.345416         90.909091             0.000000               3               1.898734         66.666667            100.000000             0.000000              0.333333                  True      0.400000               0.750000        999.000000                0.700000       999.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               4.946237         86.956522             0.000000              15               9.615385         60.000000            100.000000             0.000000              0.400000                 False      0.050000               0.250000        999.000000                0.500000         4.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               4.086022         89.473684             0.000000              16              10.256410         56.250000            100.000000             0.000000              0.437500                 False      0.050000               0.250000          2.000000                0.500000         4.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.345416         90.909091             0.000000               9               5.696203         55.555556            100.000000             0.000000              0.444444                  True      0.000000               0.750000        999.000000                0.500000         3.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.345416         90.909091             0.000000               8               5.063291         37.500000            100.000000             0.000000              0.625000                  True      0.400000               0.250000        999.000000                0.500000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB07_CHA2_1011_1229_parquet_slot_delta_prior.csv

[27/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB07_CHB1_1011_1229.parquet
loaded shape=(3569, 801)

==================================================================================================================================
Dataset: EPLBAB07_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(3569, 802), sort_time=0.012s
label out-of-range run_value=0/3569, policy=clip
split: train=[0,2855), cal=[2855,3390), test=[3390,3569)
split sizes: train=2855, cal=535, test=179

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           23   0.805604          0  0.000000           6  3.351955
    3          243   8.511384         96 17.943925          38 21.229050
    4         1136  39.789842        330 61.682243         102 56.983240
    5         1184  41.471103        102 19.065421          31 17.318436
    6          257   9.001751          7  1.308411           2  1.117318
    7           12   0.420315          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.167812
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   5         -1.318382       -0.529124      -3.883350       1.919319       5.802670       -1.318382                 False
       6 239          0.468540        0.621192      -1.461469       2.575428       4.036897        0.468540                 False
       7   5          2.480309        2.389215       1.673759       2.603233       0.929474        2.480309                 False
       8 229          0.326326        0.332489      -2.250526       2.433331       4.683857        0.326326                 False
       9   2         -0.248550       -0.248550      -0.832827       0.335726       1.168552       -0.167812                  True
      10 244          0.828773        0.569393      -1.478558       2.708191       4.186749        0.828773                 False
      11   3          2.273682        2.974321       2.160446       3.437876       1.277430        2.273682                 False
      14 248         -0.280802       -0.322188      -2.344740       1.677814       4.022554       -0.280802                 False
      15   3          4.804920        4.257886       3.008425       5.780865       2.772440        4.804920                 False
      16 228          0.302620        0.177550      -1.939610       2.191664       4.131274        0.302620                 False
      17   1         -8.305058       -8.305058      -8.305058      -8.305058       0.000000       -0.167812                  True
      18 246         -1.230161       -1.196034      -3.249733       0.884032       4.133765       -1.230161                 False
      19   3         -0.266533       -1.298672      -2.866077       0.784803       3.650881       -0.266533                 False
      24 217         -1.942619       -1.817078      -4.108891       0.949520       5.058411       -1.942619                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=798 -> miss=715 -> var=614 -> final=120
  C_loop_count: fit_time=2.19s pred_time=1.78s

  [CAL FULL C_loop_count]
    n=314 | Acc=64.01% | Within1=98.73% | Severe(|d|>=2)=1.27% | MAE=0.3726 | RMSE=0.6309 | Penalty=0.3981 | MeanDiff=-0.0287

  [TEST FULL C_loop_count]
    n=104 | Acc=65.38% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3462 | RMSE=0.5883 | Penalty=0.3462 | MeanDiff=0.0577

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=828 -> miss=745 -> var=644 -> final=120
  F_delta_run_trend: fit_time=2.26s pred_time=1.77s

  [CAL FULL F_delta_run_trend]
    n=313 | Acc=65.81% | Within1=99.04% | Severe(|d|>=2)=0.96% | MAE=0.3514 | RMSE=0.6088 | Penalty=0.3706 | MeanDiff=0.0256

  [TEST FULL F_delta_run_trend]
    n=104 | Acc=69.23% | Within1=98.08% | Severe(|d|>=2)=1.92% | MAE=0.3269 | RMSE=0.6045 | Penalty=0.3654 | MeanDiff=0.0192

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         313     65.814696         0.958466          104      69.230769          98.076923          1.923077           0.365385              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         314     64.012739         1.273885          104      65.384615         100.000000          0.000000           0.346154              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              69              22.044728         86.956522             0.000000              23              22.115385         86.956522            100.000000             0.000000              0.130435                 False      0.300000               0.250000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              68              21.725240         86.764706             0.000000              23              22.115385         86.956522            100.000000             0.000000              0.130435                 False      0.300000               0.250000        999.000000                0.000000         3.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.821656         91.666667             0.000000               9               8.653846         77.777778            100.000000             0.000000              0.222222                  True      0.000000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.821656         91.666667             0.000000               9               8.653846         77.777778            100.000000             0.000000              0.222222                  True      0.000000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              33              10.543131         87.878788             0.000000               8               7.692308         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              33              10.543131         87.878788             0.000000               8               7.692308         75.000000            100.000000             0.000000              0.250000                  True      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.821656         91.666667             0.000000              10               9.615385         70.000000            100.000000             0.000000              0.300000                 False      0.000000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.821656         91.666667             0.000000              11              10.576923         63.636364            100.000000             0.000000              0.363636                 False      0.000000               0.250000        999.000000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB07_CHB1_1011_1229_parquet_slot_delta_prior.csv

[28/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB07_CHB2_1011_1229.parquet
loaded shape=(3860, 803)

==================================================================================================================================
Dataset: EPLBAB07_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(3860, 804), sort_time=0.013s
label out-of-range run_value=0/3860, policy=clip
split: train=[0,3088), cal=[3088,3667), test=[3667,3860)
split sizes: train=3088, cal=579, test=193

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           78   2.525907          0  0.000000           5  2.590674
    3          749  24.255181         62 10.708117          32 16.580311
    4         1561  50.550518        332 57.340242         107 55.440415
    5          636  20.595855        176 30.397237          47 24.352332
    6           61   1.975389          9  1.554404           2  1.036269
    7            3   0.097150          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.525634
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 213         -0.710297       -0.634819      -3.003822       1.534885       4.538708       -0.710297                 False
       6   3         -0.863926       -1.084307      -1.281162      -0.777261       0.503901       -0.863926                 False
       7 239          0.378273        0.401804      -1.791468       2.608171       4.399639        0.378273                 False
       8   4         -0.391087       -1.172940      -1.709177       0.145150       1.854327       -0.391087                 False
       9 231          0.657768        0.677901      -1.499965       2.481208       3.981173        0.657768                 False
      10   2         -1.420561       -1.420561      -1.530774      -1.310348       0.220427       -0.525634                  True
      11 244         -0.551796       -0.469144      -2.478567       1.324629       3.803196       -0.551796                 False
      14   2         -1.702813       -1.702813      -1.867777      -1.537849       0.329928       -0.525634                  True
      15 248         -1.274408       -0.926659      -3.277254       1.430734       4.707988       -1.274408                 False
      16   3          1.331572        2.538465       0.957744       3.515740       2.557997        1.331572                 False
      17 229         -1.051033       -1.068050      -3.162743       0.810413       3.973156       -1.051033                 False
      18   1         -5.372070       -5.372070      -5.372070      -5.372070       0.000000       -0.525634                  True
      19 246         -1.313210       -1.282820      -3.513677       1.357067       4.870743       -1.313210                 False
      24   1         -3.086563       -3.086563      -3.086563      -3.086563       0.000000       -0.525634                  True
      25 218         -0.688011       -0.329380      -3.467997       2.408212       5.876209       -0.688011                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=800 -> miss=717 -> var=619 -> final=120
  C_loop_count: fit_time=2.13s pred_time=1.81s

  [CAL FULL C_loop_count]
    n=356 | Acc=62.92% | Within1=99.16% | Severe(|d|>=2)=0.84% | MAE=0.3792 | RMSE=0.6293 | Penalty=0.3961 | MeanDiff=0.0309

  [TEST FULL C_loop_count]
    n=119 | Acc=59.66% | Within1=98.32% | Severe(|d|>=2)=1.68% | MAE=0.4202 | RMSE=0.6736 | Penalty=0.4538 | MeanDiff=0.0000

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=830 -> miss=747 -> var=649 -> final=120
  F_delta_run_trend: fit_time=2.03s pred_time=1.77s

  [CAL FULL F_delta_run_trend]
    n=351 | Acc=71.51% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.2849 | RMSE=0.5338 | Penalty=0.2849 | MeanDiff=-0.0171

  [TEST FULL F_delta_run_trend]
    n=118 | Acc=66.10% | Within1=98.31% | Severe(|d|>=2)=1.69% | MAE=0.3559 | RMSE=0.6244 | Penalty=0.3898 | MeanDiff=-0.0847

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         351     71.509972         0.000000          118      66.101695          98.305085          1.694915           0.389830              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         356     62.921348         0.842697          119      59.663866          98.319328          1.680672           0.453782              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   both            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.617978         95.000000             0.000000               6               5.042017        100.000000            100.000000             0.000000              0.000000                  True      0.400000               1.000000        999.000000                0.500000       999.000000            4.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               3.988604         92.857143             0.000000               1               0.847458        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               3.988604         92.857143             0.000000               1               0.847458        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               3.988604         92.857143             0.000000               1               0.847458        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   slot            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               4.213483        100.000000             0.000000               8               6.722689         87.500000            100.000000             0.000000              0.125000                  True      0.450000               1.000000        999.000000                0.000000       999.000000            4.000000
     C_loop_count            loop  trend            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              21               5.898876         95.238095             0.000000               7               5.882353         85.714286            100.000000             0.000000              0.142857                  True      0.400000               1.000000        999.000000                0.500000       999.000000            4.000000
     C_loop_count            loop either            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.617978         95.000000             0.000000               7               5.882353         85.714286            100.000000             0.000000              0.142857                  True      0.400000               1.000000        999.000000                0.500000         4.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               8.547009         93.333333             0.000000               6               5.084746         83.333333            100.000000             0.000000              0.166667                  True      0.400000               0.500000          1.500000                0.500000         4.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB07_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB07_CHB2_1011_1229_parquet_slot_delta_prior.csv

[29/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB08_CHA1_1011_1229.parquet
loaded shape=(4458, 883)

==================================================================================================================================
Dataset: EPLBAB08_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(4458, 884), sort_time=0.021s
label out-of-range run_value=0/4458, policy=clip
split: train=[0,3566), cal=[3566,4235), test=[4235,4458)
split sizes: train=3566, cal=669, test=223

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           85   2.383623          4  0.597907           2  0.896861
    3          892  25.014021        180 26.905830          49 21.973094
    4         1911  53.589456        391 58.445441         104 46.636771
    5          650  18.227706         93 13.901345          63 28.251121
    6           27   0.757151          1  0.149477           5  2.242152
    7            1   0.028043          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=0.008427
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   7          4.381821        3.543741       1.248723       4.743093       3.494370        4.381821                 False
       6 296          0.766963        0.700355      -1.422174       2.662028       4.084201        0.766963                 False
       7   1          5.074392        5.074392       5.074392       5.074392       0.000000        0.008427                  True
       8 293          0.118073       -0.076581      -2.158428       2.264530       4.422958        0.118073                 False
       9   2         -3.822165       -3.822165      -3.826798      -3.817531       0.009268        0.008427                  True
      10 265          1.031914        0.855025      -0.808613       2.865671       3.674284        1.031914                 False
      14 266         -0.417830       -0.590367      -2.405045       1.337404       3.742449       -0.417830                 False
      16 328          0.268481        0.310739      -1.665261       2.284171       3.949432        0.268481                 False
      17   1          7.314577        7.314577       7.314577       7.314577       0.000000        0.008427                  True
      18 262         -0.617035       -0.942356      -3.188178       1.189086       4.377264       -0.617035                 False
      24 330         -0.958236       -0.728283      -3.187957       1.722816       4.910773       -0.958236                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=880 -> miss=835 -> var=709 -> final=120
  C_loop_count: fit_time=2.10s pred_time=1.88s

  [CAL FULL C_loop_count]
    n=376 | Acc=67.55% | Within1=99.47% | Severe(|d|>=2)=0.53% | MAE=0.3298 | RMSE=0.5835 | Penalty=0.3404 | MeanDiff=-0.0532

  [TEST FULL C_loop_count]
    n=128 | Acc=64.84% | Within1=99.22% | Severe(|d|>=2)=0.78% | MAE=0.3594 | RMSE=0.6124 | Penalty=0.3750 | MeanDiff=0.0000

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=910 -> miss=865 -> var=739 -> final=120
  F_delta_run_trend: fit_time=2.11s pred_time=1.85s

  [CAL FULL F_delta_run_trend]
    n=376 | Acc=70.21% | Within1=99.73% | Severe(|d|>=2)=0.27% | MAE=0.3005 | RMSE=0.5530 | Penalty=0.3059 | MeanDiff=-0.0293

  [TEST FULL F_delta_run_trend]
    n=128 | Acc=67.19% | Within1=99.22% | Severe(|d|>=2)=0.78% | MAE=0.3359 | RMSE=0.5929 | Penalty=0.3516 | MeanDiff=-0.0234

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         376     70.212766         0.265957          128      67.187500          99.218750          0.781250           0.351562              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         376     67.553191         0.531915          128      64.843750          99.218750          0.781250           0.375000              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.989362         86.666667             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.250000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.989362         86.666667             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.989362         86.666667             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.989362         86.666667             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.250000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.925532         90.909091             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.300000               0.250000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.925532         90.909091             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.300000               0.250000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.925532         90.909091             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.300000               0.250000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.925532         90.909091             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.300000               0.250000        999.000000                0.000000         1.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB08_CHA1_1011_1229_parquet_slot_delta_prior.csv

[30/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB08_CHA2_1011_1229.parquet
loaded shape=(4783, 865)

==================================================================================================================================
Dataset: EPLBAB08_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(4783, 866), sort_time=0.020s
label out-of-range run_value=0/4783, policy=clip
split: train=[0,3826), cal=[3826,4543), test=[4543,4783)
split sizes: train=3826, cal=717, test=240

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           31   0.810246          1  0.139470           1  0.416667
    3          524  13.695766         57  7.949791          30 12.500000
    4         1968  51.437533        388 54.114365         115 47.916667
    5         1167  30.501830        253 35.285914          84 35.000000
    6          136   3.554626         18  2.510460          10  4.166667
    7            0   0.000000          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.313164
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 246         -1.322710       -0.970701      -2.986185       1.429873       4.416057       -1.322710                 False
       6   1         -4.889816       -4.889816      -4.889816      -4.889816       0.000000       -0.313164                  True
       7 298          0.442103        0.524853      -1.952509       2.622787       4.575295        0.442103                 False
       8   1         -1.387936       -1.387936      -1.387936      -1.387936       0.000000       -0.313164                  True
       9 293          0.381062        0.281188      -1.724022       2.315430       4.039452        0.381062                 False
      10   2         -4.376778       -4.376778      -5.423618      -3.329937       2.093681       -0.313164                  True
      11 266         -0.307426       -0.236166      -2.905137       2.211741       5.116878       -0.307426                 False
      14   2         -5.933483       -5.933483      -6.253105      -5.613861       0.639244       -0.313164                  True
      15 268         -0.933874       -0.769042      -3.094248       1.445518       4.539766       -0.933874                 False
      17 328         -1.106726       -1.034770      -3.394773       1.292232       4.687005       -1.106726                 False
      18   1         -8.460787       -8.460787      -8.460787      -8.460787       0.000000       -0.313164                  True
      19 263         -1.057629       -1.083503      -3.395312       1.162365       4.557677       -1.057629                 False
      25 332          1.007044        1.134029      -1.570810       3.755078       5.325889        1.007044                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=862 -> miss=817 -> var=690 -> final=120
  C_loop_count: fit_time=2.17s pred_time=1.90s

  [CAL FULL C_loop_count]
    n=424 | Acc=58.49% | Within1=97.64% | Severe(|d|>=2)=2.36% | MAE=0.4387 | RMSE=0.6970 | Penalty=0.4858 | MeanDiff=0.0566

  [TEST FULL C_loop_count]
    n=146 | Acc=57.53% | Within1=99.32% | Severe(|d|>=2)=0.68% | MAE=0.4315 | RMSE=0.6672 | Penalty=0.4452 | MeanDiff=-0.0068

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=892 -> miss=847 -> var=720 -> final=120
  F_delta_run_trend: fit_time=2.07s pred_time=1.88s

  [CAL FULL F_delta_run_trend]
    n=423 | Acc=64.78% | Within1=99.76% | Severe(|d|>=2)=0.24% | MAE=0.3546 | RMSE=0.5994 | Penalty=0.3593 | MeanDiff=-0.0378

  [TEST FULL F_delta_run_trend]
    n=145 | Acc=65.52% | Within1=99.31% | Severe(|d|>=2)=0.69% | MAE=0.3517 | RMSE=0.6046 | Penalty=0.3655 | MeanDiff=-0.1172

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         423     64.775414         0.236407          145      65.517241          99.310345          0.689655           0.365517              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         424     58.490566         2.358491          146      57.534247          99.315068          0.684932           0.445205              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              29               6.855792         86.206897             0.000000               3               2.068966        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.700000         2.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.600473         90.909091             0.000000               1               0.689655        100.000000            100.000000             0.000000              0.000000                  True      0.200000               0.250000        999.000000                0.700000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.600473         90.909091             0.000000               1               0.689655        100.000000            100.000000             0.000000              0.000000                  True      0.200000               0.250000        999.000000                0.700000         1.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              10               2.364066         90.000000             0.000000               1               0.689655          0.000000            100.000000             0.000000              1.000000                  True      0.400000               0.250000          2.000000                0.000000         2.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.830189         83.333333             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.000000               0.250000        999.000000                0.700000         1.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.594340         81.818182             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.200000               0.750000        999.000000                0.700000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.594340         81.818182             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.200000               0.750000        999.000000                0.700000         1.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.830189         83.333333             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.000000               0.250000        999.000000                0.700000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB08_CHA2_1011_1229_parquet_slot_delta_prior.csv

[31/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB08_CHB1_1011_1229.parquet
loaded shape=(4047, 762)

==================================================================================================================================
Dataset: EPLBAB08_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(4047, 763), sort_time=0.014s
label out-of-range run_value=0/4047, policy=clip
split: train=[0,3237), cal=[3237,3844), test=[3844,4047)
split sizes: train=3237, cal=607, test=203

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2          195   6.024096         12  1.976936          32 15.763547
    3         1186  36.638863        165 27.182867          87 42.857143
    4         1526  47.142416        345 56.836903          78 38.423645
    5          328  10.132839         84 13.838550           6  2.955665
    6            2   0.061786          1  0.164745           0  0.000000
    7            0   0.000000          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.248783
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   5          2.151834        2.128811       1.275673       2.334896       1.059223        2.151834                 False
       6 267          0.187248       -0.067387      -2.299509       2.493528       4.793037        0.187248                 False
       7   1          2.075769        2.075769       2.075769       2.075769       0.000000       -0.248783                  True
       8 270          0.445794        0.419229      -1.432616       2.292727       3.725342        0.445794                 False
       9   1          3.079184        3.079184       3.079184       3.079184       0.000000       -0.248783                  True
      10 299          0.317844        0.305693      -1.991502       2.740496       4.731997        0.317844                 False
      11   2          2.866529        2.866529       2.422659       3.310398       0.887738       -0.248783                  True
      14 299         -1.215416       -1.028316      -3.039207       1.216738       4.255944       -1.215416                 False
      15   2         -2.170442       -2.170442      -3.737176      -0.603707       3.133470       -0.248783                  True
      16 237          0.272636        0.120999      -1.735703       1.836227       3.571930        0.272636                 False
      18 304         -0.798529       -0.888214      -3.197742       1.174000       4.371741       -0.798529                 False
      19   1          8.686472        8.686472       8.686472       8.686472       0.000000       -0.248783                  True
      24 235         -1.210243       -1.111342      -3.681285       1.432154       5.113439       -1.210243                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=759 -> miss=734 -> var=583 -> final=120
  C_loop_count: fit_time=2.15s pred_time=1.81s

  [CAL FULL C_loop_count]
    n=370 | Acc=62.43% | Within1=99.73% | Severe(|d|>=2)=0.27% | MAE=0.3784 | RMSE=0.6195 | Penalty=0.3838 | MeanDiff=0.0270

  [TEST FULL C_loop_count]
    n=121 | Acc=64.46% | Within1=99.17% | Severe(|d|>=2)=0.83% | MAE=0.3636 | RMSE=0.6166 | Penalty=0.3802 | MeanDiff=-0.0165

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=789 -> miss=764 -> var=613 -> final=120
  F_delta_run_trend: fit_time=2.07s pred_time=1.82s

  [CAL FULL F_delta_run_trend]
    n=370 | Acc=67.84% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3216 | RMSE=0.5671 | Penalty=0.3216 | MeanDiff=-0.0243

  [TEST FULL F_delta_run_trend]
    n=121 | Acc=68.60% | Within1=99.17% | Severe(|d|>=2)=0.83% | MAE=0.3223 | RMSE=0.5821 | Penalty=0.3388 | MeanDiff=-0.0413

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         370     67.837838         0.000000          121      68.595041          99.173554          0.826446           0.338843              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         370     62.432432         0.270270          121      64.462810          99.173554          0.826446           0.380165              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              25               6.756757         96.000000             0.000000               8               6.611570         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.750000        999.000000                0.000000       999.000000            4.000000
F_delta_run_trend delta_run_trend  trend            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              25               6.756757         96.000000             0.000000               8               6.611570         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.750000        999.000000                0.000000       999.000000            4.000000
F_delta_run_trend delta_run_trend   both            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              25               6.756757         96.000000             0.000000               8               6.611570         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.750000        999.000000                0.000000       999.000000            4.000000
F_delta_run_trend delta_run_trend either            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              25               6.756757         96.000000             0.000000               8               6.611570         75.000000            100.000000             0.000000              0.250000                  True      0.300000               0.750000        999.000000                0.000000       999.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.783784         92.857143             0.000000               3               2.479339         66.666667            100.000000             0.000000              0.333333                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.783784         92.857143             0.000000               3               2.479339         66.666667            100.000000             0.000000              0.333333                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.783784         92.857143             0.000000               4               3.305785         50.000000            100.000000             0.000000              0.500000                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.783784         92.857143             0.000000               4               3.305785         50.000000            100.000000             0.000000              0.500000                  True      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB08_CHB1_1011_1229_parquet_slot_delta_prior.csv

[32/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB08_CHB2_1011_1229.parquet
loaded shape=(4429, 763)

==================================================================================================================================
Dataset: EPLBAB08_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(4429, 764), sort_time=0.016s
label out-of-range run_value=0/4429, policy=clip
split: train=[0,3543), cal=[3543,4207), test=[4207,4429)
split sizes: train=3543, cal=664, test=222

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2            5   0.141123          1  0.150602           0  0.000000
    3          272   7.677110         42  6.325301           6  2.702703
    4         1574  44.425628        364 54.819277          63 28.378378
    5         1477  41.687835        232 34.939759         113 50.900901
    6          207   5.842506         25  3.765060          39 17.567568
    7            8   0.225797          0  0.000000           1  0.450450
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.378971
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 298         -0.605995       -0.610012      -2.669343       1.606138       4.275482       -0.605995                 False
       6   1         -2.840076       -2.840076      -2.840076      -2.840076       0.000000       -0.378971                  True
       7 264          0.287452        0.269307      -1.715380       2.570766       4.286146        0.287452                 False
       8   2          2.336002        2.336002       0.909922       3.762083       2.852161       -0.378971                  True
       9 270          0.735035        1.055650      -1.153924       3.171545       4.325469        0.735035                 False
      10   1         -7.801216       -7.801216      -7.801216      -7.801216       0.000000       -0.378971                  True
      11 298         -0.393570       -0.477488      -3.105507       1.904342       5.009849       -0.393570                 False
      15 298         -0.968622       -0.926624      -2.897443       1.044647       3.942090       -0.968622                 False
      16   1         -6.821831       -6.821831      -6.821831      -6.821831       0.000000       -0.378971                  True
      17 238         -1.279013       -1.060123      -3.408156       1.312912       4.721068       -1.279013                 False
      19 303         -0.846848       -0.797745      -3.058659       1.381294       4.439953       -0.846848                 False
      25 234         -0.007235        0.308709      -2.429438       3.218857       5.648294       -0.007235                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=760 -> miss=735 -> var=609 -> final=120
  C_loop_count: fit_time=1.97s pred_time=1.84s

  [CAL FULL C_loop_count]
    n=425 | Acc=59.06% | Within1=98.59% | Severe(|d|>=2)=1.41% | MAE=0.4235 | RMSE=0.6721 | Penalty=0.4518 | MeanDiff=0.0141

  [TEST FULL C_loop_count]
    n=137 | Acc=64.23% | Within1=97.81% | Severe(|d|>=2)=2.19% | MAE=0.3796 | RMSE=0.6507 | Penalty=0.4234 | MeanDiff=0.1168

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=790 -> miss=765 -> var=639 -> final=120
  F_delta_run_trend: fit_time=2.12s pred_time=1.84s

  [CAL FULL F_delta_run_trend]
    n=425 | Acc=63.53% | Within1=99.06% | Severe(|d|>=2)=0.94% | MAE=0.3741 | RMSE=0.6269 | Penalty=0.3929 | MeanDiff=0.0447

  [TEST FULL F_delta_run_trend]
    n=137 | Acc=67.15% | Within1=98.54% | Severe(|d|>=2)=1.46% | MAE=0.3431 | RMSE=0.6101 | Penalty=0.3723 | MeanDiff=0.0365

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         425     63.529412         0.941176          137      67.153285          98.540146          1.459854           0.372263              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         425     59.058824         1.411765          137      64.233577          97.810219          2.189781           0.423358              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              44              10.352941         77.272727             0.000000               8               5.839416         75.000000            100.000000             0.000000              0.250000                  True      0.450000               0.500000        999.000000                0.000000       999.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              44              10.352941         77.272727             0.000000               8               5.839416         75.000000            100.000000             0.000000              0.250000                  True      0.450000               0.500000        999.000000                0.000000       999.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               4.470588         84.210526             0.000000               3               2.189781         33.333333            100.000000             0.000000              0.666667                  True      0.300000               0.500000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               3.058824         84.615385             0.000000               3               2.189781         33.333333            100.000000             0.000000              0.666667                  True      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               4.470588         84.210526             0.000000               3               2.189781         33.333333            100.000000             0.000000              0.666667                  True      0.300000               0.500000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               2.352941         80.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.000000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               2.352941         80.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.000000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              12               2.823529         91.666667             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.300000               0.250000          1.500000                0.000000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB08_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB08_CHB2_1011_1229_parquet_slot_delta_prior.csv

[33/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB09_CHA1_1011_1229.parquet
loaded shape=(4472, 884)

==================================================================================================================================
Dataset: EPLBAB09_CHA1_1011_1229.parquet
==================================================================================================================================
shape=(4472, 885), sort_time=0.018s
label out-of-range run_value=0/4472, policy=clip
split: train=[0,3577), cal=[3577,4248), test=[4248,4472)
split sizes: train=3577, cal=671, test=224

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2          101   2.823595          0  0.000000           0  0.000000
    3          785  21.945765         22  3.278689          14  6.250000
    4         1725  48.224769        301 44.858420          87 38.839286
    5          890  24.881185        299 44.560358         102 45.535714
    6           74   2.068773         49  7.302534          20  8.928571
    7            2   0.055913          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           1  0.446429

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.116737
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   6          2.005316        3.869889       1.142246       5.207761       4.065515        2.005316                 False
       6 285          0.719738        0.528842      -1.665157       2.675140       4.340298        0.719738                 False
       7   3         -0.056889       -1.414761      -2.467905       0.317320       2.785225       -0.056889                 False
       8 307          0.253868        0.297387      -1.703627       2.228359       3.931986        0.253868                 False
       9   2          0.474207        0.474207       0.473075       0.475339       0.002264       -0.116737                  True
      10 288          0.251050        0.237251      -2.196689       2.714724       4.911412        0.251050                 False
      11   3          1.191128        0.615194      -0.700214       2.218569       2.918783        1.191128                 False
      14 284         -0.671508       -0.608707      -2.487178       1.356027       3.843205       -0.671508                 False
      15   3          1.267303       -0.457842      -2.049914       1.996803       4.046718        1.267303                 False
      16 307          0.418165        0.216588      -1.639013       2.209097       3.848110        0.418165                 False
      17   2         -3.732305       -3.732305      -4.971893      -2.492716       2.479177       -0.116737                  True
      18 285         -1.169567       -1.049875      -3.143562       0.713928       3.857491       -1.169567                 False
      19   2         -0.825724       -0.825724      -1.699748       0.048301       1.748049       -0.116737                  True
      24 302         -0.618631       -0.672926      -3.055216       1.505179       4.560395       -0.618631                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=881 -> miss=816 -> var=685 -> final=120
  C_loop_count: fit_time=2.17s pred_time=1.86s

  [CAL FULL C_loop_count]
    n=389 | Acc=62.98% | Within1=99.49% | Severe(|d|>=2)=0.51% | MAE=0.3753 | RMSE=0.6210 | Penalty=0.3856 | MeanDiff=0.0720

  [TEST FULL C_loop_count]
    n=131 | Acc=64.89% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3511 | RMSE=0.5926 | Penalty=0.3511 | MeanDiff=0.1221

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=911 -> miss=846 -> var=715 -> final=120
  F_delta_run_trend: fit_time=2.10s pred_time=1.83s

  [CAL FULL F_delta_run_trend]
    n=389 | Acc=67.87% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3213 | RMSE=0.5669 | Penalty=0.3213 | MeanDiff=-0.0231

  [TEST FULL F_delta_run_trend]
    n=131 | Acc=65.65% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3435 | RMSE=0.5861 | Penalty=0.3435 | MeanDiff=0.0076

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         389     67.866324         0.000000          131      65.648855         100.000000          0.000000           0.343511              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         389     62.982005         0.514139          131      64.885496         100.000000          0.000000           0.351145              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.712082         86.666667             0.000000               5               3.816794        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.712082         86.666667             0.000000               5               3.816794        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.712082         86.666667             0.000000               5               3.816794        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.712082         86.666667             0.000000               5               3.816794        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.084833         91.666667             0.000000               2               1.526718        100.000000            100.000000             0.000000              0.000000                  True      0.450000               1.000000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.341902         92.307692             0.000000               2               1.526718        100.000000            100.000000             0.000000              0.000000                  True      0.450000               1.000000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.084833         91.666667             0.000000               2               1.526718        100.000000            100.000000             0.000000              0.000000                  True      0.450000               1.000000        999.000000                0.000000         1.000000            4.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.341902         92.307692             0.000000               2               1.526718        100.000000            100.000000             0.000000              0.000000                  True      0.450000               1.000000        999.000000                0.000000         1.000000            4.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHA1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHA1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB09_CHA1_1011_1229_parquet_slot_delta_prior.csv

[34/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB09_CHA2_1011_1229.parquet
loaded shape=(4857, 866)

==================================================================================================================================
Dataset: EPLBAB09_CHA2_1011_1229.parquet
==================================================================================================================================
shape=(4857, 867), sort_time=0.016s
label out-of-range run_value=0/4857, policy=clip
split: train=[0,3885), cal=[3885,4614), test=[4614,4857)
split sizes: train=3885, cal=729, test=243

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           79   2.033462          7  0.960219           0  0.000000
    3          916  23.577864        225 30.864198          76 31.275720
    4         1911  49.189189        401 55.006859         115 47.325103
    5          874  22.496782         92 12.620027          51 20.987654
    6          104   2.676963          4  0.548697           0  0.000000
    7            1   0.025740          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           1  0.411523
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.243233
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 296         -0.396286       -0.031756      -2.336394       2.217637       4.554031       -0.396286                 False
       6   2         -1.628613       -1.628613      -2.773834      -0.483391       2.290443       -0.243233                  True
       7 286          0.075306       -0.035408      -2.063520       1.864366       3.927886        0.075306                 False
       8   3         -1.830418       -1.257931      -3.959268       1.157162       5.116429       -1.830418                 False
       9 307          0.174782        0.477151      -1.399817       2.440823       3.840640        0.174782                 False
      10   2         -2.395792       -2.395792      -2.826884      -1.964700       0.862185       -0.243233                  True
      11 288         -0.521783       -0.373446      -2.667718       2.055614       4.723332       -0.521783                 False
      14   2         -0.731155       -0.731155      -1.803019       0.340708       2.143726       -0.243233                  True
      15 284         -0.926625       -0.858975      -3.436495       1.434187       4.870682       -0.926625                 False
      16   3          0.312263        0.566965      -1.231918       2.238498       3.470416        0.312263                 False
      17 306         -0.868229       -0.795896      -2.988307       1.497797       4.486104       -0.868229                 False
      18   2         -5.750809       -5.750809      -5.820827      -5.680790       0.140038       -0.243233                  True
      19 285         -0.961372       -0.907402      -3.010616       1.094788       4.105404       -0.961372                 False
      24   1         -3.105854       -3.105854      -3.105854      -3.105854       0.000000       -0.243233                  True
      25 302          0.914946        0.969125      -1.602434       3.635291       5.237725        0.914946                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=863 -> miss=798 -> var=679 -> final=120
  C_loop_count: fit_time=1.97s pred_time=1.90s

  [CAL FULL C_loop_count]
    n=447 | Acc=62.19% | Within1=98.88% | Severe(|d|>=2)=1.12% | MAE=0.3893 | RMSE=0.6416 | Penalty=0.4116 | MeanDiff=-0.0224

  [TEST FULL C_loop_count]
    n=149 | Acc=58.39% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.4161 | RMSE=0.6451 | Penalty=0.4161 | MeanDiff=-0.0537

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=893 -> miss=828 -> var=708 -> final=120
  F_delta_run_trend: fit_time=2.11s pred_time=1.89s

  [CAL FULL F_delta_run_trend]
    n=447 | Acc=64.88% | Within1=99.33% | Severe(|d|>=2)=0.67% | MAE=0.3579 | RMSE=0.6094 | Penalty=0.3714 | MeanDiff=-0.0134

  [TEST FULL F_delta_run_trend]
    n=149 | Acc=65.10% | Within1=100.00% | Severe(|d|>=2)=0.00% | MAE=0.3490 | RMSE=0.5908 | Penalty=0.3490 | MeanDiff=-0.0671

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         447     64.876957         0.671141          149      65.100671         100.000000          0.000000           0.348993              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         447     62.192394         1.118568          149      58.389262         100.000000          0.000000           0.416107              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              31               6.935123         77.419355             0.000000              17              11.409396         88.235294            100.000000             0.000000              0.117647                 False      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              31               6.935123         77.419355             0.000000              17              11.409396         88.235294            100.000000             0.000000              0.117647                 False      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend   slot fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              37               8.277405         78.378378             0.000000              20              13.422819         80.000000            100.000000             0.000000              0.200000                 False      0.200000               0.500000        999.000000                0.000000         1.000000          999.000000
F_delta_run_trend delta_run_trend either fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              37               8.277405         78.378378             0.000000              20              13.422819         80.000000            100.000000             0.000000              0.200000                 False      0.200000               0.500000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.684564         83.333333             0.000000               7               4.697987         71.428571            100.000000             0.000000              0.285714                  True      0.450000               0.250000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.684564         83.333333             0.000000               7               4.697987         71.428571            100.000000             0.000000              0.285714                  True      0.450000               0.250000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               2.908277         84.615385             0.000000               8               5.369128         62.500000            100.000000             0.000000              0.375000                  True      0.450000               0.250000        999.000000                0.000000         2.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               2.908277         84.615385             0.000000               9               6.040268         55.555556            100.000000             0.000000              0.444444                  True      0.450000               0.250000        999.000000                0.000000         2.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHA2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHA2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB09_CHA2_1011_1229_parquet_slot_delta_prior.csv

[35/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB09_CHB1_1011_1229.parquet
loaded shape=(4340, 822)

==================================================================================================================================
Dataset: EPLBAB09_CHB1_1011_1229.parquet
==================================================================================================================================
shape=(4340, 823), sort_time=0.009s
label out-of-range run_value=0/4340, policy=clip
split: train=[0,3472), cal=[3472,4123), test=[4123,4340)
split sizes: train=3472, cal=651, test=217

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2          119   3.427419          0  0.000000           1  0.460829
    3          831  23.934332         58  8.909370          26 11.981567
    4         1676  48.271889        367 56.374808         110 50.691244
    5          773  22.263825        208 31.950845          60 27.649770
    6           72   2.073733         17  2.611367          20  9.216590
    7            1   0.028802          1  0.153610           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.354885
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1   9         -0.970032       -0.080955      -2.071178       2.036839       4.108017       -0.970032                 False
       6 296          0.168885        0.108590      -2.111026       2.495982       4.607008        0.168885                 False
       7   2          0.185623        0.185623      -0.288788       0.660034       0.948822       -0.354885                  True
       8 275          0.321678        0.346981      -1.794737       2.547508       4.342245        0.321678                 False
       9   3         -1.718676       -0.462683      -2.150538       0.597177       2.747715       -1.718676                 False
      10 296          0.345531        0.158833      -1.914162       2.283391       4.197553        0.345531                 False
      11   2         -4.807215       -4.807215      -5.231059      -4.383370       0.847689       -0.354885                  True
      14 300         -0.894653       -0.859446      -2.823449       1.167274       3.990724       -0.894653                 False
      15   2         -0.872791       -0.872791      -0.971979      -0.773603       0.198376       -0.354885                  True
      16 276          0.348377        0.151679      -1.827003       2.017881       3.844884        0.348377                 False
      17   3         -4.196487       -3.890177      -4.401094      -3.532415       0.868679       -4.196487                 False
      18 299         -0.937550       -1.118566      -3.345370       1.093537       4.438908       -0.937550                 False
      19   2         -1.933641       -1.933641      -3.679105      -0.188178       3.490927       -0.354885                  True
      24 281         -1.254116       -1.265781      -3.877104       0.781708       4.658812       -1.254116                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=819 -> miss=754 -> var=641 -> final=120
  C_loop_count: fit_time=2.01s pred_time=1.86s

  [CAL FULL C_loop_count]
    n=385 | Acc=58.44% | Within1=98.18% | Severe(|d|>=2)=1.82% | MAE=0.4338 | RMSE=0.6857 | Penalty=0.4701 | MeanDiff=0.0078

  [TEST FULL C_loop_count]
    n=125 | Acc=61.60% | Within1=97.60% | Severe(|d|>=2)=2.40% | MAE=0.4080 | RMSE=0.6753 | Penalty=0.4560 | MeanDiff=0.0720

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=849 -> miss=784 -> var=671 -> final=120
  F_delta_run_trend: fit_time=2.25s pred_time=1.84s

  [CAL FULL F_delta_run_trend]
    n=385 | Acc=60.78% | Within1=99.48% | Severe(|d|>=2)=0.52% | MAE=0.3974 | RMSE=0.6386 | Penalty=0.4078 | MeanDiff=-0.0597

  [TEST FULL F_delta_run_trend]
    n=125 | Acc=68.00% | Within1=99.20% | Severe(|d|>=2)=0.80% | MAE=0.3280 | RMSE=0.5865 | Penalty=0.3440 | MeanDiff=-0.0400

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         385     60.779221         0.519481          125      68.000000          99.200000          0.800000           0.344000              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         385     58.441558         1.818182          125      61.600000          97.600000          2.400000           0.456000              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend   slot            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               3.896104        100.000000             0.000000               6               4.800000        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000
F_delta_run_trend delta_run_trend  trend            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               3.896104        100.000000             0.000000               6               4.800000        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000
F_delta_run_trend delta_run_trend   both            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               3.896104        100.000000             0.000000               6               4.800000        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000
F_delta_run_trend delta_run_trend either            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               3.896104        100.000000             0.000000               6               4.800000        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.896104         93.333333             0.000000               2               1.600000        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.500000         1.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.376623         92.307692             0.000000               2               1.600000        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.376623         92.307692             0.000000               2               1.600000        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.896104         93.333333             0.000000               2               1.600000        100.000000            100.000000             0.000000              0.000000                  True      0.300000               0.250000        999.000000                0.500000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHB1_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHB1_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB09_CHB1_1011_1229_parquet_slot_delta_prior.csv

[36/36] /ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB09_CHB2_1011_1229.parquet
loaded shape=(4679, 823)

==================================================================================================================================
Dataset: EPLBAB09_CHB2_1011_1229.parquet
==================================================================================================================================
shape=(4679, 824), sort_time=0.009s
label out-of-range run_value=0/4679, policy=clip
split: train=[0,3743), cal=[3743,4445), test=[4445,4679)
split sizes: train=3743, cal=702, test=234

----------------------------------------------------------------------------------------------------------------------------------
Loop distribution by split
----------------------------------------------------------------------------------------------------------------------------------
 loop  train_count  train_pct  cal_count   cal_pct  test_count  test_pct
    2           25   0.667913          1  0.142450           2  0.854701
    3          531  14.186481        146 20.797721          42 17.948718
    4         1848  49.372161        401 57.122507         117 50.000000
    5         1186  31.685814        142 20.227920          67 28.632479
    6          152   4.060914         12  1.709402           6  2.564103
    7            1   0.026717          0  0.000000           0  0.000000
    8            0   0.000000          0  0.000000           0  0.000000
    9            0   0.000000          0  0.000000           0  0.000000

----------------------------------------------------------------------------------------------------------------------------------
Slot-delta prior learned from TRAIN only
----------------------------------------------------------------------------------------------------------------------------------
slot_delta_min_count=3
global_delta_run_median=-0.387918
 slot_id   n  delta_run_median  delta_run_mean  delta_run_p25  delta_run_p75  delta_run_iqr  used_delta_run  used_global_fallback
       1 259         -1.063232       -0.799376      -2.886243       1.430061       4.316304       -1.063232                 False
       6   3          0.868462        2.397052       0.709423       3.320386       2.610963        0.868462                 False
       7 296          0.056596        0.150075      -1.689645       2.219707       3.909353        0.056596                 False
       8   2          0.054823        0.054823      -1.189515       1.299161       2.488676       -0.387918                  True
       9 275          0.188614        0.422368      -1.553537       2.390852       3.944389        0.188614                 False
      10   3          1.773468        1.822765       0.280741       3.340140       3.059400        1.773468                 False
      11 296         -0.146559       -0.181757      -2.429399       1.857935       4.287334       -0.146559                 False
      14   3         -1.958134       -2.272084      -2.875752      -1.511440       1.364312       -1.958134                 False
      15 300         -0.344109       -0.483044      -2.710374       1.403666       4.114039       -0.344109                 False
      16   2          0.111563        0.111563      -0.920991       1.144116       2.065107       -0.387918                  True
      17 276         -0.644810       -0.864387      -3.013684       1.360639       4.374323       -0.644810                 False
      18   3          1.392235       -0.074144      -0.992634       1.577536       2.570169        1.392235                 False
      19 299         -1.412281       -1.300078      -3.116763       0.851088       3.967851       -1.412281                 False
      25 282          0.019176       -0.014560      -2.543036       2.658796       5.201833        0.019176                 False

==================================================================================================================================
Training selected routes and selecting high-conf rules on CAL
==================================================================================================================================

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict C_loop_count
----------------------------------------------------------------------------------------------------------------------------------
  principle: Directly regress the final ordinal loop_count and round/clip the continuous loop score.
  implementation_diff: Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.
  C_loop_count: feature pruning raw=820 -> miss=776 -> var=656 -> final=120
  C_loop_count: fit_time=2.07s pred_time=1.88s

  [CAL FULL C_loop_count]
    n=433 | Acc=55.43% | Within1=98.38% | Severe(|d|>=2)=1.62% | MAE=0.4619 | RMSE=0.7030 | Penalty=0.4942 | MeanDiff=0.0970

  [TEST FULL C_loop_count]
    n=143 | Acc=62.94% | Within1=99.30% | Severe(|d|>=2)=0.70% | MAE=0.3776 | RMSE=0.6258 | Penalty=0.3916 | MeanDiff=-0.0979

----------------------------------------------------------------------------------------------------------------------------------
Fit/Predict F_delta_run_trend
----------------------------------------------------------------------------------------------------------------------------------
  principle: Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.
  implementation_diff: Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.
  F_delta_run_trend: feature pruning raw=850 -> miss=806 -> var=686 -> final=120
  F_delta_run_trend: fit_time=2.21s pred_time=1.90s

  [CAL FULL F_delta_run_trend]
    n=433 | Acc=62.36% | Within1=99.54% | Severe(|d|>=2)=0.46% | MAE=0.3811 | RMSE=0.6247 | Penalty=0.3903 | MeanDiff=-0.0069

  [TEST FULL F_delta_run_trend]
    n=143 | Acc=68.53% | Within1=98.60% | Severe(|d|>=2)=1.40% | MAE=0.3287 | RMSE=0.5972 | Penalty=0.3566 | MeanDiff=-0.0629

==================================================================================================================================
SUMMARY 1: Full CAL/TEST for selected routes
==================================================================================================================================
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         433     62.355658         0.461894          143      68.531469          98.601399          1.398601           0.356643              120
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         433     55.427252         1.616628          143      62.937063          99.300699          0.699301           0.391608              120

==================================================================================================================================
SUMMARY 2: CAL-selected high-conf rules applied to TEST
==================================================================================================================================
            model     target_type   mode          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_severe  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max
F_delta_run_trend delta_run_trend  trend fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              18               4.157044         88.888889             0.000000               2               1.398601        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.700000         2.000000          999.000000
F_delta_run_trend delta_run_trend   both fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              18               4.157044         88.888889             0.000000               2               1.398601        100.000000            100.000000             0.000000              0.000000                  True      0.400000               0.250000        999.000000                0.700000         2.000000          999.000000
     C_loop_count            loop  trend fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              96              22.170901         71.875000             0.000000              24              16.783217         79.166667            100.000000             0.000000              0.208333                 False      0.000000               0.250000        999.000000                0.700000         3.000000          999.000000
     C_loop_count            loop either fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              78              18.013857         73.076923             0.000000              28              19.580420         78.571429            100.000000             0.000000              0.214286                 False      0.200000               0.250000        999.000000                0.700000         2.000000          999.000000
     C_loop_count            loop   slot fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              33               7.621247         75.757576             0.000000              22              15.384615         77.272727            100.000000             0.000000              0.227273                 False      0.400000               1.000000        999.000000                0.000000         1.000000          999.000000
     C_loop_count            loop   both fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              24               5.542725         75.000000             0.000000              13               9.090909         69.230769            100.000000             0.000000              0.307692                 False      0.200000               1.000000        999.000000                0.000000         1.000000            4.000000
F_delta_run_trend delta_run_trend   slot            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.540416        100.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.250000        999.000000                0.500000         1.000000          999.000000
F_delta_run_trend delta_run_trend either            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.540416        100.000000             0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True      0.400000               0.250000        999.000000                0.500000         1.000000          999.000000

Saved CSV:
  full_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHB2_1011_1229_parquet_summary_full_cal_test.csv
  selected_rules_summary=./results/two_routes_calibrated_high_conf/EPLBAB09_CHB2_1011_1229_parquet_summary_cal_selected_test_applied.csv
  slot_prior=./results/two_routes_calibrated_high_conf/EPLBAB09_CHB2_1011_1229_parquet_slot_delta_prior.csv

==================================================================================================================================
ALL DATASETS DONE
==================================================================================================================================
success=36/36 total_time=2448.8s

Combined full summary:
            model     target_type                                                                                    principle                                                                                                                                                                         implementation_diff  cal_full_n  cal_full_acc  cal_full_severe  test_full_n  test_full_acc  test_full_within1  test_full_severe  test_full_penalty  n_features_used                         dataset
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         266     61.654135         0.751880           89      56.179775          97.752809          2.247191           0.505618              120 EPLBAB01_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         265     65.283019         0.754717           88      67.045455          98.863636          1.136364           0.363636              120 EPLBAB01_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         304     62.828947         0.657895          103      64.077670         100.000000          0.000000           0.359223              120 EPLBAB01_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         301     67.774086         0.332226          102      64.705882         100.000000          0.000000           0.352941              120 EPLBAB01_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         269     62.081784         0.743494           94      62.765957          98.936170          1.063830           0.404255              120 EPLBAB01_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         269     68.773234         1.115242           94      62.765957          98.936170          1.063830           0.404255              120 EPLBAB01_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         309     58.252427         0.970874          104      58.653846         100.000000          0.000000           0.413462              120 EPLBAB01_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         309     62.135922         0.970874          104      67.307692          98.076923          1.923077           0.384615              120 EPLBAB01_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         251     67.330677         1.195219           82      60.975610          98.780488          1.219512           0.426829              120 EPLBAB02_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         251     70.119522         0.398406           82      67.073171         100.000000          0.000000           0.329268              120 EPLBAB02_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         289     66.782007         1.730104           95      64.210526          95.789474          4.210526           0.484211              120 EPLBAB02_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         289     68.512111         1.038062           95      64.210526          96.842105          3.157895           0.452632              120 EPLBAB02_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         236     66.101695         0.423729           82      58.536585         100.000000          0.000000           0.414634              120 EPLBAB02_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         236     68.220339         0.847458           82      53.658537         100.000000          0.000000           0.463415              120 EPLBAB02_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         264     64.772727         0.378788           92      67.391304          98.913043          1.086957           0.358696              120 EPLBAB02_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         262     65.267176         0.000000           92      68.478261         100.000000          0.000000           0.315217              120 EPLBAB02_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         349     60.744986         1.719198          112      69.642857          99.107143          0.892857           0.330357              120 EPLBAB03_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         349     66.189112         2.292264          112      68.750000         100.000000          0.000000           0.312500              120 EPLBAB03_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         392     61.224490         0.510204          130      59.230769          98.461538          1.538462           0.453846              120 EPLBAB03_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         392     63.775510         2.295918          128      64.843750          99.218750          0.781250           0.375000              120 EPLBAB03_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         310     64.838710         0.322581          107      62.616822          99.065421          0.934579           0.401869              120 EPLBAB03_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         310     67.419355         0.000000          107      68.224299         100.000000          0.000000           0.317757              120 EPLBAB03_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         354     65.254237         1.129944          118      64.406780          99.152542          0.847458           0.381356              120 EPLBAB03_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         353     66.005666         1.133144          118      65.254237          99.152542          0.847458           0.372881              120 EPLBAB03_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         356     63.483146         0.000000          120      65.833333         100.000000          0.000000           0.341667              120 EPLBAB04_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         356     71.910112         0.280899          120      69.166667         100.000000          0.000000           0.308333              120 EPLBAB04_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         409     57.701711         0.488998          135      55.555556          97.777778          2.222222           0.511111              120 EPLBAB04_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         408     61.519608         0.000000          135      60.000000          99.259259          0.740741           0.422222              120 EPLBAB04_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         397     68.010076         0.503778          136      69.117647          99.264706          0.735294           0.330882              120 EPLBAB04_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         397     75.566751         0.251889          136      72.794118         100.000000          0.000000           0.272059              120 EPLBAB04_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         451     62.749446         0.221729          154      68.831169         100.000000          0.000000           0.311688              120 EPLBAB04_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         451     67.849224         0.221729          154      64.935065          99.350649          0.649351           0.370130              120 EPLBAB04_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         352     64.772727         1.420455          118      63.559322         100.000000          0.000000           0.364407              120 EPLBAB05_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         352     65.909091         0.852273          118      58.474576         100.000000          0.000000           0.415254              120 EPLBAB05_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         404     59.653465         0.990099          136      60.294118          96.323529          3.676471           0.544118              120 EPLBAB05_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         402     68.159204         0.746269          136      58.088235          97.794118          2.205882           0.522059              120 EPLBAB05_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         418     62.200957         1.435407          142      59.859155         100.000000          0.000000           0.401408              120 EPLBAB05_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         418     64.114833         1.674641          137      63.503650         100.000000          0.000000           0.364964              120 EPLBAB05_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         471     61.995754         2.335456          158      65.189873          99.367089          0.632911           0.367089              120 EPLBAB05_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         468     64.957265         1.709402          158      62.658228          99.367089          0.632911           0.392405              120 EPLBAB05_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         342     67.251462         1.169591          112      67.857143         100.000000          0.000000           0.321429              120 EPLBAB06_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         342     71.345029         0.292398          112      67.857143         100.000000          0.000000           0.321429              120 EPLBAB06_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         390     58.205128         1.025641          128      71.093750         100.000000          0.000000           0.289062              120 EPLBAB06_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         390     62.820513         0.256410          128      68.750000         100.000000          0.000000           0.312500              120 EPLBAB06_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         352     66.193182         0.568182          117      71.794872         100.000000          0.000000           0.282051              120 EPLBAB06_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         352     69.602273         0.284091          117      72.649573         100.000000          0.000000           0.273504              120 EPLBAB06_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         400     61.500000         1.000000          133      64.661654          99.248120          0.751880           0.375940              120 EPLBAB06_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         400     66.000000         0.250000          133      66.165414         100.000000          0.000000           0.338346              120 EPLBAB06_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         411     66.666667         0.243309          140      61.428571          97.857143          2.142857           0.450000              120 EPLBAB07_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         410     70.487805         0.243902          139      66.187050          97.841727          2.158273           0.402878              120 EPLBAB07_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         469     58.422175         1.279318          158      69.620253          98.734177          1.265823           0.341772              120 EPLBAB07_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         465     60.215054         0.215054          156      67.307692         100.000000          0.000000           0.326923              120 EPLBAB07_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         314     64.012739         1.273885          104      65.384615         100.000000          0.000000           0.346154              120 EPLBAB07_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         313     65.814696         0.958466          104      69.230769          98.076923          1.923077           0.365385              120 EPLBAB07_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         356     62.921348         0.842697          119      59.663866          98.319328          1.680672           0.453782              120 EPLBAB07_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         351     71.509972         0.000000          118      66.101695          98.305085          1.694915           0.389830              120 EPLBAB07_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         376     67.553191         0.531915          128      64.843750          99.218750          0.781250           0.375000              120 EPLBAB08_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         376     70.212766         0.265957          128      67.187500          99.218750          0.781250           0.351562              120 EPLBAB08_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         424     58.490566         2.358491          146      57.534247          99.315068          0.684932           0.445205              120 EPLBAB08_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         423     64.775414         0.236407          145      65.517241          99.310345          0.689655           0.365517              120 EPLBAB08_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         370     62.432432         0.270270          121      64.462810          99.173554          0.826446           0.380165              120 EPLBAB08_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         370     67.837838         0.000000          121      68.595041          99.173554          0.826446           0.338843              120 EPLBAB08_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         425     59.058824         1.411765          137      64.233577          97.810219          2.189781           0.423358              120 EPLBAB08_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         425     63.529412         0.941176          137      67.153285          98.540146          1.459854           0.372263              120 EPLBAB08_CHB2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         389     62.982005         0.514139          131      64.885496         100.000000          0.000000           0.351145              120 EPLBAB09_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         389     67.866324         0.000000          131      65.648855         100.000000          0.000000           0.343511              120 EPLBAB09_CHA1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         447     62.192394         1.118568          149      58.389262         100.000000          0.000000           0.416107              120 EPLBAB09_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         447     64.876957         0.671141          149      65.100671         100.000000          0.000000           0.348993              120 EPLBAB09_CHA2_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         385     58.441558         1.818182          125      61.600000          97.600000          2.400000           0.456000              120 EPLBAB09_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         385     60.779221         0.519481          125      68.000000          99.200000          0.800000           0.344000              120 EPLBAB09_CHB1_1011_1229.parquet
     C_loop_count            loop      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.         433     55.427252         1.616628          143      62.937063          99.300699          0.699301           0.391608              120 EPLBAB09_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.         433     62.355658         0.461894          143      68.531469          98.601399          1.398601           0.356643              120 EPLBAB09_CHB2_1011_1229.parquet

Combined selected-rule test summary:
            model     target_type   mode  boundary_min  abs_ref_bias_loop_max  rule_run_gap_max  ref_trend_abs_corr_min  ref_run_std_max  slot_delta_iqr_max  hc_min_ref  hc_max_comp_shift  n  coverage_pct   accuracy    within1  severe_ge2  penalty  mean_diff          select_reason                                                                                    principle                                                                                                                                                                         implementation_diff  cal_selected_n  cal_selected_coverage  cal_selected_acc  cal_selected_within1  cal_selected_severe  cal_selected_penalty  test_applied_n  test_applied_coverage  test_applied_acc  test_applied_within1  test_applied_severe  test_applied_penalty  test_warning_small_n                         dataset
     C_loop_count            loop   slot      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000           2                  1 12      4.511278  83.333333 100.000000    0.000000 0.166667   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.511278         83.333333            100.000000             0.000000              0.166667               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB01_CHA1_1011_1229.parquet
     C_loop_count            loop  trend      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000           2                  1 11      4.135338  81.818182 100.000000    0.000000 0.181818   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               4.135338         81.818182            100.000000             0.000000              0.181818               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB01_CHA1_1011_1229.parquet
     C_loop_count            loop   both      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000           2                  1 11      4.135338  81.818182 100.000000    0.000000 0.181818   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               4.135338         81.818182            100.000000             0.000000              0.181818               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB01_CHA1_1011_1229.parquet
     C_loop_count            loop either      0.400000               0.500000        999.000000                0.000000       999.000000            4.000000           2                  1 12      4.511278  83.333333 100.000000    0.000000 0.166667   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.511278         83.333333            100.000000             0.000000              0.166667               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB01_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 16      6.037736  81.250000 100.000000    0.000000 0.187500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000            100.000000             0.000000              0.187500               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 16      6.037736  81.250000 100.000000    0.000000 0.187500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000            100.000000             0.000000              0.187500               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 16      6.037736  81.250000 100.000000    0.000000 0.187500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000            100.000000             0.000000              0.187500               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 16      6.037736  81.250000 100.000000    0.000000 0.187500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               6.037736         81.250000            100.000000             0.000000              0.187500               5               5.681818         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA1_1011_1229.parquet
     C_loop_count            loop   slot      0.200000               1.500000        999.000000                0.000000         4.000000            4.000000           2                  1 20      6.578947  90.000000 100.000000    0.000000 0.100000   0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               6.578947         90.000000            100.000000             0.000000              0.100000               2               1.941748        100.000000            100.000000             0.000000              0.000000                  True EPLBAB01_CHA2_1011_1229.parquet
     C_loop_count            loop  trend      0.000000               0.750000        999.000000                0.000000         4.000000            4.000000           2                  1 10      3.289474  90.000000 100.000000    0.000000 0.100000  -0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.289474         90.000000            100.000000             0.000000              0.100000               5               4.854369         80.000000            100.000000             0.000000              0.200000                  True EPLBAB01_CHA2_1011_1229.parquet
     C_loop_count            loop   both      0.200000               1.500000        999.000000                0.000000         4.000000            4.000000           2                  1 20      6.578947  90.000000 100.000000    0.000000 0.100000   0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               6.578947         90.000000            100.000000             0.000000              0.100000               2               1.941748        100.000000            100.000000             0.000000              0.000000                  True EPLBAB01_CHA2_1011_1229.parquet
     C_loop_count            loop either      0.200000               0.750000        999.000000                0.000000       999.000000            4.000000           2                  1 10      3.289474  90.000000 100.000000    0.000000 0.100000   0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.289474         90.000000            100.000000             0.000000              0.100000               3               2.912621         33.333333            100.000000             0.000000              0.666667                  True EPLBAB01_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000           2                  1 14      4.651163  92.857143 100.000000    0.000000 0.071429   0.071429 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143            100.000000             0.000000              0.071429               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000           2                  1 14      4.651163  92.857143 100.000000    0.000000 0.071429   0.071429 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143            100.000000             0.000000              0.071429               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000           2                  1 14      4.651163  92.857143 100.000000    0.000000 0.071429   0.071429 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143            100.000000             0.000000              0.071429               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.200000               0.250000        999.000000                0.000000         4.000000            4.000000           2                  1 14      4.651163  92.857143 100.000000    0.000000 0.071429   0.071429 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               4.651163         92.857143            100.000000             0.000000              0.071429               5               4.901961         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHA2_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               1.000000        999.000000                0.000000         1.000000          999.000000           2                  1 12      4.460967  91.666667 100.000000    0.000000 0.083333  -0.083333 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.460967         91.666667            100.000000             0.000000              0.083333               8               8.510638         37.500000            100.000000             0.000000              0.625000                  True EPLBAB01_CHB1_1011_1229.parquet
     C_loop_count            loop  trend      0.400000               1.000000        999.000000                0.000000         2.000000          999.000000           2                  1 14      5.204461  85.714286 100.000000    0.000000 0.142857  -0.142857 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               5.204461         85.714286            100.000000             0.000000              0.142857              12              12.765957         58.333333            100.000000             0.000000              0.416667                 False EPLBAB01_CHB1_1011_1229.parquet
     C_loop_count            loop   both      0.400000               1.000000        999.000000                0.000000         2.000000          999.000000           2                  1 13      4.832714  92.307692 100.000000    0.000000 0.076923  -0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.832714         92.307692            100.000000             0.000000              0.076923              12              12.765957         58.333333            100.000000             0.000000              0.416667                 False EPLBAB01_CHB1_1011_1229.parquet
     C_loop_count            loop either      0.400000               1.000000        999.000000                0.000000         1.000000          999.000000           2                  1 13      4.832714  84.615385 100.000000    0.000000 0.153846  -0.153846 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.832714         84.615385            100.000000             0.000000              0.153846               8               8.510638         37.500000            100.000000             0.000000              0.625000                  True EPLBAB01_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 12      4.460967  83.333333 100.000000    0.000000 0.166667   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              12               4.460967         83.333333            100.000000             0.000000              0.166667              10              10.638298         50.000000            100.000000             0.000000              0.500000                 False EPLBAB01_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.000000         3.000000          999.000000           2                  1 32     11.895911  84.375000 100.000000    0.000000 0.156250  -0.093750 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32              11.895911         84.375000            100.000000             0.000000              0.156250              11              11.702128         63.636364            100.000000             0.000000              0.363636                 False EPLBAB01_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.000000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 32     11.895911  87.500000 100.000000    0.000000 0.125000   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32              11.895911         87.500000            100.000000             0.000000              0.125000              19              20.212766         52.631579            100.000000             0.000000              0.473684                 False EPLBAB01_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 12      4.460967  83.333333 100.000000    0.000000 0.166667   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              12               4.460967         83.333333            100.000000             0.000000              0.166667              10              10.638298         50.000000            100.000000             0.000000              0.500000                 False EPLBAB01_CHB1_1011_1229.parquet
     C_loop_count            loop   slot      0.200000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 25      8.090615  80.000000 100.000000    0.000000 0.200000   0.040000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              25               8.090615         80.000000            100.000000             0.000000              0.200000               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHB2_1011_1229.parquet
     C_loop_count            loop  trend      0.200000               0.500000        999.000000                0.500000       999.000000            4.000000           2                  1 14      4.530744  85.714286 100.000000    0.000000 0.142857  -0.142857 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               4.530744         85.714286            100.000000             0.000000              0.142857               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHB2_1011_1229.parquet
     C_loop_count            loop   both      0.200000               0.500000        999.000000                0.500000       999.000000            4.000000           2                  1 12      3.883495  83.333333 100.000000    0.000000 0.166667  -0.166667 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               3.883495         83.333333            100.000000             0.000000              0.166667               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHB2_1011_1229.parquet
     C_loop_count            loop either      0.200000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 26      8.414239  80.769231 100.000000    0.000000 0.192308   0.038462 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              26               8.414239         80.769231            100.000000             0.000000              0.192308               5               4.807692         60.000000            100.000000             0.000000              0.400000                  True EPLBAB01_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.200000               0.250000          1.500000                0.000000         3.000000            4.000000           2                  1 22      7.119741  81.818182 100.000000    0.000000 0.181818   0.090909 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              22               7.119741         81.818182            100.000000             0.000000              0.181818               6               5.769231         66.666667            100.000000             0.000000              0.333333                  True EPLBAB01_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.200000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 23      7.443366  78.260870 100.000000    0.000000 0.217391   0.043478 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               7.443366         78.260870            100.000000             0.000000              0.217391               6               5.769231         66.666667            100.000000             0.000000              0.333333                  True EPLBAB01_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.100000               0.250000        999.000000                0.000000       999.000000            4.000000           2                  1 46     14.886731  78.260870 100.000000    0.000000 0.217391   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              46              14.886731         78.260870            100.000000             0.000000              0.217391              14              13.461538         64.285714            100.000000             0.000000              0.357143                 False EPLBAB01_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.200000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 23      7.443366  78.260870 100.000000    0.000000 0.217391   0.043478 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               7.443366         78.260870            100.000000             0.000000              0.217391               6               5.769231         66.666667            100.000000             0.000000              0.333333                  True EPLBAB01_CHB2_1011_1229.parquet
     C_loop_count            loop   slot      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000           2                  1 19      7.569721  94.736842 100.000000    0.000000 0.052632   0.052632 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842            100.000000             0.000000              0.052632               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
     C_loop_count            loop  trend      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000           2                  1 19      7.569721  94.736842 100.000000    0.000000 0.052632   0.052632 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842            100.000000             0.000000              0.052632               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
     C_loop_count            loop   both      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000           2                  1 19      7.569721  94.736842 100.000000    0.000000 0.052632   0.052632 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842            100.000000             0.000000              0.052632               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
     C_loop_count            loop either      0.200000               1.000000        999.000000                0.500000         3.000000            4.000000           2                  1 19      7.569721  94.736842 100.000000    0.000000 0.052632   0.052632 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               7.569721         94.736842            100.000000             0.000000              0.052632               3               3.658537        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 20      7.968127  90.000000 100.000000    0.000000 0.100000   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               7.968127         90.000000            100.000000             0.000000              0.100000               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 17      6.772908  94.117647 100.000000    0.000000 0.058824  -0.058824 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               6.772908         94.117647            100.000000             0.000000              0.058824               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 17      6.772908  94.117647 100.000000    0.000000 0.058824  -0.058824 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               6.772908         94.117647            100.000000             0.000000              0.058824               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 20      7.968127  90.000000 100.000000    0.000000 0.100000   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               7.968127         90.000000            100.000000             0.000000              0.100000               1               1.219512        100.000000            100.000000             0.000000              0.000000                  True EPLBAB02_CHA1_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000           2                  1 14      4.844291  85.714286 100.000000    0.000000 0.142857   0.142857 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               4.844291         85.714286            100.000000             0.000000              0.142857               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True EPLBAB02_CHA2_1011_1229.parquet
     C_loop_count            loop  trend      0.300000               0.500000        999.000000                0.000000         1.000000          999.000000           2                  1 13      4.498270  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.498270         92.307692            100.000000             0.000000              0.076923              11              11.578947         72.727273             90.909091             9.090909              0.545455                 False EPLBAB02_CHA2_1011_1229.parquet
     C_loop_count            loop   both      0.300000               0.500000        999.000000                0.000000         1.000000          999.000000           2                  1 12      4.152249  91.666667 100.000000    0.000000 0.083333   0.083333 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               4.152249         91.666667            100.000000             0.000000              0.083333               9               9.473684         66.666667             88.888889            11.111111              0.666667                  True EPLBAB02_CHA2_1011_1229.parquet
     C_loop_count            loop either      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000           2                  1 15      5.190311  86.666667 100.000000    0.000000 0.133333   0.133333 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               5.190311         86.666667            100.000000             0.000000              0.133333               7               7.368421         85.714286            100.000000             0.000000              0.142857                  True EPLBAB02_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.250000          2.000000                0.000000         1.000000          999.000000           2                  1 15      5.190311  93.333333 100.000000    0.000000 0.066667  -0.066667 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               5.190311         93.333333            100.000000             0.000000              0.066667              12              12.631579         66.666667            100.000000             0.000000              0.333333                 False EPLBAB02_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.250000        999.000000                0.000000         4.000000          999.000000           2                  1 16      5.536332  93.750000 100.000000    0.000000 0.062500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               5.536332         93.750000            100.000000             0.000000              0.062500               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True EPLBAB02_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.250000        999.000000                0.000000       999.000000          999.000000           2                  1 17      5.882353  94.117647 100.000000    0.000000 0.058824  -0.058824 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               5.882353         94.117647            100.000000             0.000000              0.058824               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True EPLBAB02_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.250000        999.000000                0.500000         4.000000          999.000000           2                  1 17      5.882353  88.235294 100.000000    0.000000 0.117647  -0.117647 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               5.882353         88.235294            100.000000             0.000000              0.117647               5               5.263158         80.000000            100.000000             0.000000              0.200000                  True EPLBAB02_CHA2_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 10      4.237288 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000            100.000000             0.000000              0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
     C_loop_count            loop  trend      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 10      4.237288 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000            100.000000             0.000000              0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
     C_loop_count            loop   both      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 10      4.237288 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000            100.000000             0.000000              0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
     C_loop_count            loop either      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 10      4.237288 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               4.237288        100.000000            100.000000             0.000000              0.000000               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 15      6.355932  86.666667 100.000000    0.000000 0.133333   0.133333 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               6.355932         86.666667            100.000000             0.000000              0.133333               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.500000          1.500000                0.000000         3.000000            4.000000           2                  1 13      5.508475  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               5.508475         92.307692            100.000000             0.000000              0.076923               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 15      6.355932  86.666667 100.000000    0.000000 0.133333   0.133333 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               6.355932         86.666667            100.000000             0.000000              0.133333               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 15      6.355932  86.666667 100.000000    0.000000 0.133333   0.133333 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               6.355932         86.666667            100.000000             0.000000              0.133333               4               4.878049         75.000000            100.000000             0.000000              0.250000                  True EPLBAB02_CHB1_1011_1229.parquet
     C_loop_count            loop   slot      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000           2                  1 10      3.787879  90.000000 100.000000    0.000000 0.100000   0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.787879         90.000000            100.000000             0.000000              0.100000              24              26.086957         75.000000            100.000000             0.000000              0.250000                 False EPLBAB02_CHB2_1011_1229.parquet
     C_loop_count            loop  trend      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000           2                  1 14      5.303030  85.714286 100.000000    0.000000 0.142857   0.142857 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               5.303030         85.714286            100.000000             0.000000              0.142857              23              25.000000         73.913043            100.000000             0.000000              0.260870                 False EPLBAB02_CHB2_1011_1229.parquet
     C_loop_count            loop   both      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000           2                  1 13      4.924242  84.615385 100.000000    0.000000 0.153846   0.153846 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               4.924242         84.615385            100.000000             0.000000              0.153846              21              22.826087         71.428571            100.000000             0.000000              0.285714                 False EPLBAB02_CHB2_1011_1229.parquet
     C_loop_count            loop either      0.300000               0.250000        999.000000                0.500000         1.000000          999.000000           2                  1 10      3.787879  90.000000 100.000000    0.000000 0.100000   0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.787879         90.000000            100.000000             0.000000              0.100000              18              19.565217         72.222222            100.000000             0.000000              0.277778                 False EPLBAB02_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 19      7.251908  84.210526 100.000000    0.000000 0.157895   0.157895 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               7.251908         84.210526            100.000000             0.000000              0.157895              20              21.739130         75.000000            100.000000             0.000000              0.250000                 False EPLBAB02_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000           2                  1 13      4.961832  84.615385 100.000000    0.000000 0.153846   0.153846 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               4.961832         84.615385            100.000000             0.000000              0.153846              14              15.217391         85.714286            100.000000             0.000000              0.142857                 False EPLBAB02_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.250000        999.000000                0.500000         2.000000          999.000000           2                  1 13      4.961832  84.615385 100.000000    0.000000 0.153846   0.153846 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               4.961832         84.615385            100.000000             0.000000              0.153846              14              15.217391         85.714286            100.000000             0.000000              0.142857                 False EPLBAB02_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 19      7.251908  84.210526 100.000000    0.000000 0.157895   0.157895 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               7.251908         84.210526            100.000000             0.000000              0.157895              20              21.739130         75.000000            100.000000             0.000000              0.250000                 False EPLBAB02_CHB2_1011_1229.parquet
     C_loop_count            loop   slot      0.300000               0.500000        999.000000                0.000000         2.000000            4.000000           2                  1 13      3.724928  92.307692 100.000000    0.000000 0.076923  -0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.724928         92.307692            100.000000             0.000000              0.076923               2               1.785714        100.000000            100.000000             0.000000              0.000000                  True EPLBAB03_CHA1_1011_1229.parquet
     C_loop_count            loop  trend      0.300000               0.500000        999.000000                0.000000         2.000000            4.000000           2                  1 11      3.151862 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.151862        100.000000            100.000000             0.000000              0.000000               3               2.678571        100.000000            100.000000             0.000000              0.000000                  True EPLBAB03_CHA1_1011_1229.parquet
     C_loop_count            loop   both      0.300000               0.500000        999.000000                0.000000         2.000000            4.000000           2                  1 11      3.151862 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.151862        100.000000            100.000000             0.000000              0.000000               2               1.785714        100.000000            100.000000             0.000000              0.000000                  True EPLBAB03_CHA1_1011_1229.parquet
     C_loop_count            loop either      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 13      3.724928  92.307692 100.000000    0.000000 0.076923  -0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.724928         92.307692            100.000000             0.000000              0.076923               5               4.464286        100.000000            100.000000             0.000000              0.000000                  True EPLBAB03_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.500000          2.000000                0.000000         2.000000          999.000000           2                  1 27      7.736390  92.592593 100.000000    0.000000 0.074074  -0.074074 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               7.736390         92.592593            100.000000             0.000000              0.074074              14              12.500000         85.714286            100.000000             0.000000              0.142857                 False EPLBAB03_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 20      5.730659  90.000000 100.000000    0.000000 0.100000  -0.100000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.730659         90.000000            100.000000             0.000000              0.100000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False EPLBAB03_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.250000        999.000000                0.000000         4.000000          999.000000           2                  1 15      4.297994  93.333333 100.000000    0.000000 0.066667  -0.066667 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              15               4.297994         93.333333            100.000000             0.000000              0.066667              10               8.928571         90.000000            100.000000             0.000000              0.100000                 False EPLBAB03_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 26      7.449857  92.307692 100.000000    0.000000 0.076923  -0.076923 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               7.449857         92.307692            100.000000             0.000000              0.076923              14              12.500000         85.714286            100.000000             0.000000              0.142857                 False EPLBAB03_CHA1_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000           2                  1 16      4.081633  81.250000 100.000000    0.000000 0.187500   0.062500 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              16               4.081633         81.250000            100.000000             0.000000              0.187500               3               2.307692         66.666667            100.000000             0.000000              0.333333                  True EPLBAB03_CHA2_1011_1229.parquet
     C_loop_count            loop  trend      0.200000               0.250000        999.000000                0.000000       999.000000          999.000000           2                  1 45     11.479592  82.222222 100.000000    0.000000 0.177778   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              45              11.479592         82.222222            100.000000             0.000000              0.177778              22              16.923077         86.363636            100.000000             0.000000              0.136364                 False EPLBAB03_CHA2_1011_1229.parquet
     C_loop_count            loop   both      0.200000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 27      6.887755  85.185185 100.000000    0.000000 0.148148   0.074074 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              27               6.887755         85.185185            100.000000             0.000000              0.148148               7               5.384615         85.714286            100.000000             0.000000              0.142857                  True EPLBAB03_CHA2_1011_1229.parquet
     C_loop_count            loop either      0.200000               0.250000        999.000000                0.000000       999.000000          999.000000           2                  1 64     16.326531  79.687500 100.000000    0.000000 0.203125   0.046875 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              64              16.326531         79.687500            100.000000             0.000000              0.203125              32              24.615385         75.000000             96.875000             3.125000              0.343750                 False EPLBAB03_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.400000               0.250000          1.500000                0.000000         3.000000          999.000000           2                  1 28      7.142857  82.142857 100.000000    0.000000 0.178571   0.107143 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               7.142857         82.142857            100.000000             0.000000              0.178571               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000           2                  1 26      6.632653  76.923077 100.000000    0.000000 0.230769   0.076923 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               6.632653         76.923077            100.000000             0.000000              0.230769               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000           2                  1 26      6.632653  76.923077 100.000000    0.000000 0.230769   0.076923 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               6.632653         76.923077            100.000000             0.000000              0.230769               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000           2                  1 30      7.653061  80.000000 100.000000    0.000000 0.200000   0.066667 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.653061         80.000000            100.000000             0.000000              0.200000               4               3.125000         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHA2_1011_1229.parquet
     C_loop_count            loop   slot      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 11      3.548387  90.909091 100.000000    0.000000 0.090909  -0.090909 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.548387         90.909091            100.000000             0.000000              0.090909               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB03_CHB1_1011_1229.parquet
     C_loop_count            loop  trend      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 10      3.225806 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               3.225806        100.000000            100.000000             0.000000              0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB03_CHB1_1011_1229.parquet
     C_loop_count            loop   both      0.000000               0.500000        999.000000                0.500000         1.000000          999.000000           2                  1 20      6.451613  95.000000 100.000000    0.000000 0.050000   0.050000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               6.451613         95.000000            100.000000             0.000000              0.050000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB03_CHB1_1011_1229.parquet
     C_loop_count            loop either      0.200000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 11      3.548387  90.909091 100.000000    0.000000 0.090909  -0.090909 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.548387         90.909091            100.000000             0.000000              0.090909               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB03_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 19      6.129032  94.736842 100.000000    0.000000 0.052632  -0.052632 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               6.129032         94.736842            100.000000             0.000000              0.052632               8               7.476636         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000           2                  1 10      3.225806 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              10               3.225806        100.000000            100.000000             0.000000              0.000000               5               4.672897         60.000000            100.000000             0.000000              0.400000                  True EPLBAB03_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000           2                  1 10      3.225806 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              10               3.225806        100.000000            100.000000             0.000000              0.000000               5               4.672897         60.000000            100.000000             0.000000              0.400000                  True EPLBAB03_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 19      6.129032  94.736842 100.000000    0.000000 0.052632  -0.052632 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              19               6.129032         94.736842            100.000000             0.000000              0.052632               8               7.476636         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHB1_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 11      3.107345  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091            100.000000             0.000000              0.090909               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True EPLBAB03_CHB2_1011_1229.parquet
     C_loop_count            loop  trend      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 11      3.107345  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091            100.000000             0.000000              0.090909               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True EPLBAB03_CHB2_1011_1229.parquet
     C_loop_count            loop   both      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 11      3.107345  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091            100.000000             0.000000              0.090909               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True EPLBAB03_CHB2_1011_1229.parquet
     C_loop_count            loop either      0.400000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 11      3.107345  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.107345         90.909091            100.000000             0.000000              0.090909               1               0.847458          0.000000            100.000000             0.000000              1.000000                  True EPLBAB03_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 20      5.665722  85.000000 100.000000    0.000000 0.150000  -0.050000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.665722         85.000000            100.000000             0.000000              0.150000               4               3.389831         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.500000          2.000000                0.700000       999.000000            4.000000           2                  1 10      2.832861  90.000000 100.000000    0.000000 0.100000  -0.100000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              10               2.832861         90.000000            100.000000             0.000000              0.100000               2               1.694915        100.000000            100.000000             0.000000              0.000000                  True EPLBAB03_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 20      5.665722  85.000000 100.000000    0.000000 0.150000  -0.050000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.665722         85.000000            100.000000             0.000000              0.150000               4               3.389831         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 20      5.665722  85.000000 100.000000    0.000000 0.150000  -0.050000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.665722         85.000000            100.000000             0.000000              0.150000               4               3.389831         75.000000            100.000000             0.000000              0.250000                  True EPLBAB03_CHB2_1011_1229.parquet
     C_loop_count            loop   slot      0.450000               0.250000        999.000000                0.500000         2.000000          999.000000           2                  1 21      5.898876  85.714286 100.000000    0.000000 0.142857   0.047619 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              21               5.898876         85.714286            100.000000             0.000000              0.142857               7               5.833333         71.428571            100.000000             0.000000              0.285714                  True EPLBAB04_CHA1_1011_1229.parquet
     C_loop_count            loop  trend      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 20      5.617978  85.000000 100.000000    0.000000 0.150000   0.050000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.617978         85.000000            100.000000             0.000000              0.150000               1               0.833333        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA1_1011_1229.parquet
     C_loop_count            loop   both      0.300000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 20      5.617978  85.000000 100.000000    0.000000 0.150000   0.050000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.617978         85.000000            100.000000             0.000000              0.150000               1               0.833333        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA1_1011_1229.parquet
     C_loop_count            loop either      0.450000               0.250000        999.000000                0.500000         2.000000          999.000000           2                  1 21      5.898876  85.714286 100.000000    0.000000 0.142857   0.047619 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              21               5.898876         85.714286            100.000000             0.000000              0.142857               7               5.833333         71.428571            100.000000             0.000000              0.285714                  True EPLBAB04_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.100000               0.250000          1.500000                0.000000         1.000000          999.000000           2                  1 24      6.741573  83.333333 100.000000    0.000000 0.166667   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              24               6.741573         83.333333            100.000000             0.000000              0.166667               9               7.500000        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.200000               0.500000          1.500000                0.500000         3.000000            4.000000           2                  1 14      3.932584  85.714286 100.000000    0.000000 0.142857   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              14               3.932584         85.714286            100.000000             0.000000              0.142857               7               5.833333        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.250000        999.000000                0.000000       999.000000          999.000000           2                  1 34      9.550562  82.352941 100.000000    0.000000 0.176471   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              34               9.550562         82.352941            100.000000             0.000000              0.176471               9               7.500000         88.888889            100.000000             0.000000              0.111111                  True EPLBAB04_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.200000               0.500000        999.000000                0.500000         3.000000            4.000000           2                  1 17      4.775281  82.352941 100.000000    0.000000 0.176471   0.058824 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               4.775281         82.352941            100.000000             0.000000              0.176471               7               5.833333        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA1_1011_1229.parquet
     C_loop_count            loop   slot      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000           2                  1 53     12.958435  77.358491 100.000000    0.000000 0.226415  -0.037736 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              53              12.958435         77.358491            100.000000             0.000000              0.226415               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True EPLBAB04_CHA2_1011_1229.parquet
     C_loop_count            loop  trend      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000           2                  1 48     11.735941  79.166667 100.000000    0.000000 0.208333  -0.041667 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              48              11.735941         79.166667            100.000000             0.000000              0.208333               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True EPLBAB04_CHA2_1011_1229.parquet
     C_loop_count            loop   both      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000           2                  1 48     11.735941  79.166667 100.000000    0.000000 0.208333  -0.041667 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              48              11.735941         79.166667            100.000000             0.000000              0.208333               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True EPLBAB04_CHA2_1011_1229.parquet
     C_loop_count            loop either      0.000000               0.250000        999.000000                0.500000         1.000000          999.000000           2                  1 53     12.958435  77.358491 100.000000    0.000000 0.226415  -0.037736 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              53              12.958435         77.358491            100.000000             0.000000              0.226415               4               2.962963         50.000000            100.000000             0.000000              0.500000                  True EPLBAB04_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000           2                  1 30      7.352941  76.666667 100.000000    0.000000 0.233333   0.100000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.352941         76.666667            100.000000             0.000000              0.233333               9               6.666667        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 11      2.696078  81.818182 100.000000    0.000000 0.181818   0.181818 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.696078         81.818182            100.000000             0.000000              0.181818               2               1.481481        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 11      2.696078  81.818182 100.000000    0.000000 0.181818   0.181818 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.696078         81.818182            100.000000             0.000000              0.181818               2               1.481481        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.250000        999.000000                0.700000         4.000000          999.000000           2                  1 30      7.352941  76.666667 100.000000    0.000000 0.233333   0.100000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              30               7.352941         76.666667            100.000000             0.000000              0.233333               9               6.666667        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHA2_1011_1229.parquet
     C_loop_count            loop   slot      0.200000               1.000000        999.000000                0.000000         1.000000            4.000000           2                  1 15      3.778338  93.333333 100.000000    0.000000 0.066667  -0.066667 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.778338         93.333333            100.000000             0.000000              0.066667               8               5.882353         87.500000            100.000000             0.000000              0.125000                  True EPLBAB04_CHB1_1011_1229.parquet
     C_loop_count            loop  trend      0.000000               1.000000        999.000000                0.000000         1.000000            4.000000           2                  1 16      4.030227  93.750000 100.000000    0.000000 0.062500  -0.062500 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              16               4.030227         93.750000            100.000000             0.000000              0.062500               7               5.147059         85.714286            100.000000             0.000000              0.142857                  True EPLBAB04_CHB1_1011_1229.parquet
     C_loop_count            loop   both      0.000000               1.000000        999.000000                0.000000         1.000000            4.000000           2                  1 16      4.030227  93.750000 100.000000    0.000000 0.062500  -0.062500 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              16               4.030227         93.750000            100.000000             0.000000              0.062500               7               5.147059         85.714286            100.000000             0.000000              0.142857                  True EPLBAB04_CHB1_1011_1229.parquet
     C_loop_count            loop either      0.200000               1.000000        999.000000                0.000000         1.000000            4.000000           2                  1 15      3.778338  93.333333 100.000000    0.000000 0.066667  -0.066667 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.778338         93.333333            100.000000             0.000000              0.066667               8               5.882353         87.500000            100.000000             0.000000              0.125000                  True EPLBAB04_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.100000               0.250000        999.000000                0.000000         1.000000            4.000000           2                  1 13      3.274559  92.307692 100.000000    0.000000 0.076923  -0.076923 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               3.274559         92.307692            100.000000             0.000000              0.076923               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000           2                  1 16      4.030227  93.750000 100.000000    0.000000 0.062500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               4.030227         93.750000            100.000000             0.000000              0.062500               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.000000               0.250000        999.000000                0.000000         1.000000            4.000000           2                  1 16      4.030227  93.750000 100.000000    0.000000 0.062500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               4.030227         93.750000            100.000000             0.000000              0.062500               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.100000               0.250000        999.000000                0.000000         1.000000            4.000000           2                  1 13      3.274559  92.307692 100.000000    0.000000 0.076923  -0.076923 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              13               3.274559         92.307692            100.000000             0.000000              0.076923               5               3.676471        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHB1_1011_1229.parquet
     C_loop_count            loop   slot      0.450000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 10      2.217295  90.000000 100.000000    0.000000 0.100000  -0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               2.217295         90.000000            100.000000             0.000000              0.100000               2               1.298701        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHB2_1011_1229.parquet
     C_loop_count            loop  trend      0.300000               0.250000        999.000000                0.000000         4.000000            4.000000           2                  1 28      6.208426  82.142857 100.000000    0.000000 0.178571   0.035714 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              28               6.208426         82.142857            100.000000             0.000000              0.178571               6               3.896104         66.666667            100.000000             0.000000              0.333333                  True EPLBAB04_CHB2_1011_1229.parquet
     C_loop_count            loop   both      0.300000               0.250000        999.000000                0.000000         4.000000            4.000000           2                  1 28      6.208426  82.142857 100.000000    0.000000 0.178571   0.035714 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              28               6.208426         82.142857            100.000000             0.000000              0.178571               6               3.896104         66.666667            100.000000             0.000000              0.333333                  True EPLBAB04_CHB2_1011_1229.parquet
     C_loop_count            loop either      0.450000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 10      2.217295  90.000000 100.000000    0.000000 0.100000  -0.100000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              10               2.217295         90.000000            100.000000             0.000000              0.100000               2               1.298701        100.000000            100.000000             0.000000              0.000000                  True EPLBAB04_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 27      5.986696  92.592593 100.000000    0.000000 0.074074   0.074074 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593            100.000000             0.000000              0.074074               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True EPLBAB04_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 27      5.986696  92.592593 100.000000    0.000000 0.074074   0.074074 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593            100.000000             0.000000              0.074074               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True EPLBAB04_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 27      5.986696  92.592593 100.000000    0.000000 0.074074   0.074074 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593            100.000000             0.000000              0.074074               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True EPLBAB04_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.500000        999.000000                0.000000         3.000000            4.000000           2                  1 27      5.986696  92.592593 100.000000    0.000000 0.074074   0.074074 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               5.986696         92.592593            100.000000             0.000000              0.074074               7               4.545455         71.428571            100.000000             0.000000              0.285714                  True EPLBAB04_CHB2_1011_1229.parquet
     C_loop_count            loop   slot      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000           2                  1 13      3.693182  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692            100.000000             0.000000              0.076923               5               4.237288         60.000000            100.000000             0.000000              0.400000                  True EPLBAB05_CHA1_1011_1229.parquet
     C_loop_count            loop  trend      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000           2                  1 13      3.693182  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692            100.000000             0.000000              0.076923               4               3.389831         50.000000            100.000000             0.000000              0.500000                  True EPLBAB05_CHA1_1011_1229.parquet
     C_loop_count            loop   both      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000           2                  1 13      3.693182  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692            100.000000             0.000000              0.076923               4               3.389831         50.000000            100.000000             0.000000              0.500000                  True EPLBAB05_CHA1_1011_1229.parquet
     C_loop_count            loop either      0.300000               0.500000        999.000000                0.000000         1.000000            4.000000           2                  1 13      3.693182  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.693182         92.307692            100.000000             0.000000              0.076923               5               4.237288         60.000000            100.000000             0.000000              0.400000                  True EPLBAB05_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.500000        999.000000                0.500000       999.000000            4.000000           2                  1 28      7.954545  96.428571 100.000000    0.000000 0.035714  -0.035714            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               7.954545         96.428571            100.000000             0.000000              0.035714              11               9.322034         63.636364            100.000000             0.000000              0.363636                 False EPLBAB05_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000           2                  1 27      7.670455  96.296296 100.000000    0.000000 0.037037  -0.037037            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               7.670455         96.296296            100.000000             0.000000              0.037037               8               6.779661         62.500000            100.000000             0.000000              0.375000                  True EPLBAB05_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.500000        999.000000                0.500000       999.000000            4.000000           2                  1 28      7.954545  96.428571 100.000000    0.000000 0.035714  -0.035714            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               7.954545         96.428571            100.000000             0.000000              0.035714              10               8.474576         60.000000            100.000000             0.000000              0.400000                 False EPLBAB05_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000           2                  1 27      7.670455  96.296296 100.000000    0.000000 0.037037  -0.037037            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              27               7.670455         96.296296            100.000000             0.000000              0.037037               9               7.627119         66.666667            100.000000             0.000000              0.333333                  True EPLBAB05_CHA1_1011_1229.parquet
     C_loop_count            loop   slot      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000           2                  1 13      3.217822  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692            100.000000             0.000000              0.076923               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHA2_1011_1229.parquet
     C_loop_count            loop  trend      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000           2                  1 13      3.217822  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692            100.000000             0.000000              0.076923               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHA2_1011_1229.parquet
     C_loop_count            loop   both      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000           2                  1 13      3.217822  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692            100.000000             0.000000              0.076923               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHA2_1011_1229.parquet
     C_loop_count            loop either      0.000000               0.500000        999.000000                0.700000         2.000000            4.000000           2                  1 13      3.217822  92.307692 100.000000    0.000000 0.076923   0.076923 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.217822         92.307692            100.000000             0.000000              0.076923               3               2.205882        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.400000               0.500000          2.000000                0.500000       999.000000          999.000000           2                  1 20      4.975124  95.000000 100.000000    0.000000 0.050000  -0.050000            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               4.975124         95.000000            100.000000             0.000000              0.050000              10               7.352941         50.000000            100.000000             0.000000              0.500000                 False EPLBAB05_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.500000        999.000000                0.000000       999.000000          999.000000           2                  1 16      3.980100  93.750000 100.000000    0.000000 0.062500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               3.980100         93.750000            100.000000             0.000000              0.062500               7               5.147059         42.857143            100.000000             0.000000              0.571429                  True EPLBAB05_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.500000        999.000000                0.000000       999.000000          999.000000           2                  1 16      3.980100  93.750000 100.000000    0.000000 0.062500  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              16               3.980100         93.750000            100.000000             0.000000              0.062500               7               5.147059         42.857143            100.000000             0.000000              0.571429                  True EPLBAB05_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.500000        999.000000                0.500000         4.000000          999.000000           2                  1 23      5.721393  91.304348 100.000000    0.000000 0.086957  -0.086957 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               5.721393         91.304348            100.000000             0.000000              0.086957              10               7.352941         50.000000            100.000000             0.000000              0.500000                 False EPLBAB05_CHA2_1011_1229.parquet
     C_loop_count            loop   slot      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 14      3.349282 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000            100.000000             0.000000              0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB05_CHB1_1011_1229.parquet
     C_loop_count            loop  trend      0.200000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 14      3.349282 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000            100.000000             0.000000              0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB05_CHB1_1011_1229.parquet
     C_loop_count            loop   both      0.200000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 14      3.349282 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000            100.000000             0.000000              0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB05_CHB1_1011_1229.parquet
     C_loop_count            loop either      0.300000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 14      3.349282 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.349282        100.000000            100.000000             0.000000              0.000000               0               0.000000               NaN                   NaN                  NaN                   NaN                  True EPLBAB05_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 17      4.066986  88.235294 100.000000    0.000000 0.117647   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               4.066986         88.235294            100.000000             0.000000              0.117647               5               3.649635         60.000000            100.000000             0.000000              0.400000                  True EPLBAB05_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000           2                  1 32      7.655502  87.500000 100.000000    0.000000 0.125000  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32               7.655502         87.500000            100.000000             0.000000              0.125000              11               8.029197         63.636364            100.000000             0.000000              0.363636                 False EPLBAB05_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000           2                  1 32      7.655502  87.500000 100.000000    0.000000 0.125000  -0.062500 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              32               7.655502         87.500000            100.000000             0.000000              0.125000              11               8.029197         63.636364            100.000000             0.000000              0.363636                 False EPLBAB05_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.250000        999.000000                0.000000         3.000000            4.000000           2                  1 17      4.066986  88.235294 100.000000    0.000000 0.117647   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              17               4.066986         88.235294            100.000000             0.000000              0.117647               5               3.649635         60.000000            100.000000             0.000000              0.400000                  True EPLBAB05_CHB1_1011_1229.parquet
     C_loop_count            loop   slot      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 14      2.972399  85.714286 100.000000    0.000000 0.142857   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               2.972399         85.714286            100.000000             0.000000              0.142857               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHB2_1011_1229.parquet
     C_loop_count            loop  trend      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 15      3.184713  86.666667 100.000000    0.000000 0.133333   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.184713         86.666667            100.000000             0.000000              0.133333               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHB2_1011_1229.parquet
     C_loop_count            loop   both      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 14      2.972399  85.714286 100.000000    0.000000 0.142857   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               2.972399         85.714286            100.000000             0.000000              0.142857               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHB2_1011_1229.parquet
     C_loop_count            loop either      0.000000               0.250000        999.000000                0.000000         2.000000            4.000000           2                  1 15      3.184713  86.666667 100.000000    0.000000 0.133333   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              15               3.184713         86.666667            100.000000             0.000000              0.133333               2               1.265823        100.000000            100.000000             0.000000              0.000000                  True EPLBAB05_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.400000               0.250000        999.000000                0.700000       999.000000          999.000000           2                  1 29      6.196581  89.655172 100.000000    0.000000 0.103448   0.034483 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              29               6.196581         89.655172            100.000000             0.000000              0.103448              11               6.962025         72.727273            100.000000             0.000000              0.272727                 False EPLBAB05_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.500000        999.000000                0.700000       999.000000          999.000000           2                  1 28      5.982906  89.285714 100.000000    0.000000 0.107143   0.035714 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               5.982906         89.285714            100.000000             0.000000              0.107143               9               5.696203         77.777778            100.000000             0.000000              0.222222                  True EPLBAB05_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.500000        999.000000                0.700000       999.000000          999.000000           2                  1 28      5.982906  89.285714 100.000000    0.000000 0.107143   0.035714 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              28               5.982906         89.285714            100.000000             0.000000              0.107143               9               5.696203         77.777778            100.000000             0.000000              0.222222                  True EPLBAB05_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.250000        999.000000                0.700000       999.000000          999.000000           2                  1 29      6.196581  89.655172 100.000000    0.000000 0.103448   0.034483 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              29               6.196581         89.655172            100.000000             0.000000              0.103448              11               6.962025         72.727273            100.000000             0.000000              0.272727                 False EPLBAB05_CHB2_1011_1229.parquet
     C_loop_count            loop   slot      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000           2                  1 13      3.801170 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.801170        100.000000            100.000000             0.000000              0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False EPLBAB06_CHA1_1011_1229.parquet
     C_loop_count            loop  trend      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000           2                  1 11      3.216374 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.216374        100.000000            100.000000             0.000000              0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False EPLBAB06_CHA1_1011_1229.parquet
     C_loop_count            loop   both      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000           2                  1 11      3.216374 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               3.216374        100.000000            100.000000             0.000000              0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False EPLBAB06_CHA1_1011_1229.parquet
     C_loop_count            loop either      0.450000               0.750000        999.000000                0.000000         1.000000          999.000000           2                  1 13      3.801170 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.801170        100.000000            100.000000             0.000000              0.000000              10               8.928571         80.000000            100.000000             0.000000              0.200000                 False EPLBAB06_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 24      7.017544  91.666667  95.833333    4.166667 0.208333  -0.125000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              24               7.017544         91.666667             95.833333             4.166667              0.208333              12              10.714286         83.333333            100.000000             0.000000              0.166667                 False EPLBAB06_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 20      5.847953  90.000000  95.000000    5.000000 0.250000  -0.150000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.847953         90.000000             95.000000             5.000000              0.250000              11               9.821429         90.909091            100.000000             0.000000              0.090909                 False EPLBAB06_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 20      5.847953  90.000000  95.000000    5.000000 0.250000  -0.150000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.847953         90.000000             95.000000             5.000000              0.250000              11               9.821429         90.909091            100.000000             0.000000              0.090909                 False EPLBAB06_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 24      7.017544  91.666667  95.833333    4.166667 0.208333  -0.125000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              24               7.017544         91.666667             95.833333             4.166667              0.208333              12              10.714286         83.333333            100.000000             0.000000              0.166667                 False EPLBAB06_CHA1_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               0.500000        999.000000                0.500000         1.000000          999.000000           2                  1 21      5.384615  95.238095 100.000000    0.000000 0.047619   0.047619            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              21               5.384615         95.238095            100.000000             0.000000              0.047619               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHA2_1011_1229.parquet
     C_loop_count            loop  trend      0.400000               0.500000        999.000000                0.000000         1.000000          999.000000           2                  1 20      5.128205  95.000000 100.000000    0.000000 0.050000  -0.050000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              20               5.128205         95.000000            100.000000             0.000000              0.050000               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHA2_1011_1229.parquet
     C_loop_count            loop   both      0.400000               0.500000        999.000000                0.500000         1.000000          999.000000           2                  1 17      4.358974 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              17               4.358974        100.000000            100.000000             0.000000              0.000000               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHA2_1011_1229.parquet
     C_loop_count            loop either      0.400000               0.500000        999.000000                0.500000         1.000000          999.000000           2                  1 22      5.641026  95.454545 100.000000    0.000000 0.045455   0.045455            meet_target      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              22               5.641026         95.454545            100.000000             0.000000              0.045455               3               2.343750        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.250000          1.500000                0.000000         1.000000          999.000000           2                  1 23      5.897436  91.304348 100.000000    0.000000 0.086957  -0.086957 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              23               5.897436         91.304348            100.000000             0.000000              0.086957               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True EPLBAB06_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 20      5.128205  85.000000 100.000000    0.000000 0.150000  -0.150000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.128205         85.000000            100.000000             0.000000              0.150000               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True EPLBAB06_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 20      5.128205  85.000000 100.000000    0.000000 0.150000  -0.150000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              20               5.128205         85.000000            100.000000             0.000000              0.150000               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True EPLBAB06_CHA2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.250000        999.000000                0.000000         1.000000          999.000000           2                  1 26      6.666667  88.461538 100.000000    0.000000 0.115385  -0.115385 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              26               6.666667         88.461538            100.000000             0.000000              0.115385               7               5.468750         85.714286            100.000000             0.000000              0.142857                  True EPLBAB06_CHA2_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000           2                  1 19      5.397727  89.473684 100.000000    0.000000 0.105263   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               5.397727         89.473684            100.000000             0.000000              0.105263               2               1.709402        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB1_1011_1229.parquet
     C_loop_count            loop  trend      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000           2                  1 27      7.670455  88.888889 100.000000    0.000000 0.111111  -0.037037 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              27               7.670455         88.888889            100.000000             0.000000              0.111111               6               5.128205         83.333333            100.000000             0.000000              0.166667                  True EPLBAB06_CHB1_1011_1229.parquet
     C_loop_count            loop   both      0.400000               0.250000        999.000000                0.000000         3.000000          999.000000           2                  1 27      7.670455  88.888889 100.000000    0.000000 0.111111  -0.037037 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              27               7.670455         88.888889            100.000000             0.000000              0.111111               6               5.128205         83.333333            100.000000             0.000000              0.166667                  True EPLBAB06_CHB1_1011_1229.parquet
     C_loop_count            loop either      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000           2                  1 19      5.397727  89.473684 100.000000    0.000000 0.105263   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              19               5.397727         89.473684            100.000000             0.000000              0.105263               2               1.709402        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000           2                  1 11      3.125000  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091            100.000000             0.000000              0.090909               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000           2                  1 11      3.125000  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091            100.000000             0.000000              0.090909               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000           2                  1 11      3.125000  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091            100.000000             0.000000              0.090909               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.300000               0.250000        999.000000                0.500000       999.000000            4.000000           2                  1 11      3.125000  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               3.125000         90.909091            100.000000             0.000000              0.090909               3               2.564103        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB1_1011_1229.parquet
     C_loop_count            loop   slot      0.000000               0.500000        999.000000                0.700000       999.000000            4.000000           2                  1 13      3.250000  84.615385 100.000000    0.000000 0.153846   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.250000         84.615385            100.000000             0.000000              0.153846               2               1.503759        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB2_1011_1229.parquet
     C_loop_count            loop  trend      0.000000               0.500000        999.000000                0.500000         3.000000            4.000000           2                  1 13      3.250000  84.615385 100.000000    0.000000 0.153846   0.000000 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              13               3.250000         84.615385            100.000000             0.000000              0.153846               2               1.503759        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB2_1011_1229.parquet
     C_loop_count            loop   both      0.000000               0.500000        999.000000                0.500000         3.000000            4.000000           2                  1 11      2.750000  90.909091 100.000000    0.000000 0.090909   0.090909 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              11               2.750000         90.909091            100.000000             0.000000              0.090909               1               0.751880        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB2_1011_1229.parquet
     C_loop_count            loop either      0.000000               0.500000        999.000000                0.500000         3.000000            4.000000           2                  1 14      3.500000  78.571429 100.000000    0.000000 0.214286  -0.071429 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              14               3.500000         78.571429            100.000000             0.000000              0.214286               3               2.255639        100.000000            100.000000             0.000000              0.000000                  True EPLBAB06_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000           2                  1 11      2.750000 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000            100.000000             0.000000              0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True EPLBAB06_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000           2                  1 11      2.750000 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000            100.000000             0.000000              0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True EPLBAB06_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000           2                  1 11      2.750000 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000            100.000000             0.000000              0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True EPLBAB06_CHB2_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.200000               0.250000        999.000000                0.500000         3.000000            4.000000           2                  1 11      2.750000 100.000000 100.000000    0.000000 0.000000   0.000000            meet_target Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              11               2.750000        100.000000            100.000000             0.000000              0.000000               2               1.503759         50.000000            100.000000             0.000000              0.500000                  True EPLBAB06_CHB2_1011_1229.parquet
     C_loop_count            loop   slot      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 12      2.919708  91.666667 100.000000    0.000000 0.083333   0.083333 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667            100.000000             0.000000              0.083333               6               4.285714         50.000000            100.000000             0.000000              0.500000                  True EPLBAB07_CHA1_1011_1229.parquet
     C_loop_count            loop  trend      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 12      2.919708  91.666667 100.000000    0.000000 0.083333   0.083333 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667            100.000000             0.000000              0.083333               4               2.857143         50.000000            100.000000             0.000000              0.500000                  True EPLBAB07_CHA1_1011_1229.parquet
     C_loop_count            loop   both      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 12      2.919708  91.666667 100.000000    0.000000 0.083333   0.083333 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667            100.000000             0.000000              0.083333               4               2.857143         50.000000            100.000000             0.000000              0.500000                  True EPLBAB07_CHA1_1011_1229.parquet
     C_loop_count            loop either      0.400000               0.250000        999.000000                0.000000         2.000000          999.000000           2                  1 12      2.919708  91.666667 100.000000    0.000000 0.083333   0.083333 fallback_best_accuracy      Directly regress the final ordinal loop_count and round/clip the continuous loop score.                                                                  Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.              12               2.919708         91.666667            100.000000             0.000000              0.083333               6               4.285714         50.000000            100.000000             0.000000              0.500000                  True EPLBAB07_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   slot      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000           2                  1 21      5.121951  80.952381 100.000000    0.000000 0.190476   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              21               5.121951         80.952381            100.000000             0.000000              0.190476              13               9.352518         76.923077            100.000000             0.000000              0.230769                 False EPLBAB07_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend  trend      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000           2                  1 38      9.268293  78.947368 100.000000    0.000000 0.210526   0.052632 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              38               9.268293         78.947368            100.000000             0.000000              0.210526               9               6.474820         88.888889            100.000000             0.000000              0.111111                  True EPLBAB07_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend   both      0.400000               0.500000        999.000000                0.000000         3.000000          999.000000           2                  1 38      9.268293  78.947368 100.000000    0.000000 0.210526   0.052632 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              38               9.268293         78.947368            100.000000             0.000000              0.210526               9               6.474820         88.888889            100.000000             0.000000              0.111111                  True EPLBAB07_CHA1_1011_1229.parquet
F_delta_run_trend delta_run_trend either      0.400000               0.250000        999.000000                0.500000         3.000000          999.000000           2                  1 21      5.121951  80.952381 100.000000    0.000000 0.190476   0.000000 fallback_best_accuracy Predict reference-relative delta_run using FDC plus lot-level reference slot trend features. Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.              21               5.121951         80.952381            100.000000             0.000000              0.190476              13               9.352518         76.923077            100.000000             0.000000              0.230769                 False EPLBAB07_CHA1_1011_1229.parquet

Process finished, exiting now...
