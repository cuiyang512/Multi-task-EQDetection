# Multi-task_Neural_Network_for_Earthquake_Classification_and_Phase_Picking
Earthquake Signal Detection Using a Multi-Scale Feature Fusion Network with Hybrid Attention Mechanism

## The test data
Regarding the labels of testing and and testing set. Please refer to the following link: [https://drive.google.com/drive/folders/1swZXzO71owRDCR9urhVchGwqrP3ecT1n?usp=drive_link](https://drive.google.com/drive/folders/1swZXzO71owRDCR9urhVchGwqrP3ecT1n?usp=sharing)

## list of the recent updated files:
1- "P_error_proposed_1.5w.npy" -> The error of P-wave samples predicted by the proposed method

2- "S_error_proposed_1.5w.npy" -> The error of S-wave samples predicted by the proposed method

3- "P_wave_phase_picking_10w_random_1006_256_100.h5" -> The pretrained model of P-wave picking using the proposed method

4- "S_wave_phase_picking_10w_random_1006_256_200.h5" -> The pretrained model of S-wave picking using the proposed method

5- "test_trainer_021.h5" -> The best pretrained model of EQCCT for S-wave arrival picking

6- "test_trainer_024.h5" -> The best pretrained model of EQCCT for P-wave arrival picking

7- "test_trainer_EQCCT_P_retrain.h5" -> The retrained P-wave picking model of EQCCT using 85,000 samples randomly selected from TXED

8- "test_trainer_EQCCT_S_retrain.h5" -> The retrained S-wave picking model of EQCCT using 85,000 samples randomly selected from TXED

9- "X_test_results_EQCCT_P" -> The output of P-wave arrival picking using "test_trainer_024.h5". The test data are 15,000 samples randomly selected from the TXED using different random seed with the training data

10- "X_test_results_EQCCT_S" -> The output of S-wave arrival picking using "test_trainer_021.h5". The test data are 15,000 samples randomly selected from the TXED using different random seed with the training data

11- "X_test_results_EQCCT_P_retrain" -> The output of P-wave arrival picking using "test_trainer_EQCCT_P_retrain.h5". The test data are 15,000 samples randomly selected from the TXED using different random seed with the training data

12- "X_test_results_EQCCT_S_retrain" -> The output of S-wave arrival picking using "test_trainer_EQCCT_S_retrain.h5". The test data are 15,000 samples randomly selected from the TXED using different random seed with the training data

13- "signalid_random_1.5w.npy" -> The index of signal waveforms in the randomly selected testing data 
