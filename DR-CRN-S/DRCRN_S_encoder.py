# Copyright (c) 2020, Ioana Bica

import logging
import numpy as np

from DR_CRN_S_model import DR_CRN_S_Model
from utils.evaluation_utils import write_results_to_file, load_trained_model, get_processed_data



def fit_DRCRN_S_encoder(dataset_train, dataset_val, model_name, model_dir, hyperparams_file,
                    b_hyperparam_opt):
    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]
    num_inputs = dataset_train['current_covariates'].shape[-1] + dataset_train['current_treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    hyperparams = dict()
    num_simulations = 50
    best_validation_mse = 1000000

    if b_hyperparam_opt:
        logging.info("Performing hyperparameter optimization")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            hyperparams['rnn_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['dr_size'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['fc_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * (hyperparams['dr_size']))
            hyperparams['learning_rate'] = np.random.choice([0.01, 0.001])
            hyperparams['batch_size'] = np.random.choice([64, 128, 256])
            hyperparams['rnn_keep_prob'] = np.random.choice([0.7, 0.8, 0.9])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = DR_CRN_S_Model(params, hyperparams)
            model.train(dataset_train, dataset_val, model_name, model_dir)
            validation_mse, _ = model.evaluate_predictions(dataset_val)

            if (validation_mse < best_validation_mse):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_mse, validation_mse))
                best_validation_mse = validation_mse
                best_hyperparams = hyperparams.copy()

            logging.info("Best hyperparams: \n {}".format(best_hyperparams))

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        logging.info("Using default hyperparameters")
        # best_hyperparams = {
        #     'rnn_hidden_units': 12,
        #     'dr_size': 12,
        #     # 'dr_size': 24,
        #     # 'fc_hidden_units': 36,
        #     'fc_hidden_units': 12,
        #     'learning_rate': 0.01,
        #     # 'batch_size': 128,
        #     'batch_size': 128,
        #     'rnn_keep_prob': 0.9}

        best_hyperparams = { #γ=5
            'rnn_hidden_units': 12,
            'dr_size': 12,
            # 'dr_size': 24,
            # 'fc_hidden_units': 36,
            'fc_hidden_units': 12,
            'learning_rate': 0.01,
            # 'batch_size': 128,
            'batch_size': 128,
            'rnn_keep_prob': 0.9}

        # best_hyperparams = {  # γ=3
        #     'rnn_hidden_units': 18,
        #     'dr_size': 24,
        #     # 'dr_size': 24,
        #     # 'fc_hidden_units': 36,
        #     'fc_hidden_units': 72,
        #     'learning_rate': 0.001,
        #     # 'batch_size': 128,
        #     'batch_size': 64,
        #     'rnn_keep_prob': 0.9}

        # best_hyperparams = {  # γ=1
        #     'rnn_hidden_units': 12,
        #     'dr_size': 12,
        #     # 'dr_size': 24,
        #     # 'fc_hidden_units': 36,
        #     'fc_hidden_units': 12,
        #     'learning_rate': 0.01,
        #     # 'batch_size': 128,
        #     'batch_size': 128,
        #     'rnn_keep_prob': 0.9}
        logging.info("Best hyperparams: \n {}".format(best_hyperparams))
        write_results_to_file(hyperparams_file, best_hyperparams) # todo 这个文件的保存有点问题，改一下保存方法

    model = DR_CRN_S_Model(params, best_hyperparams)
    model.train(dataset_train, dataset_val, model_name, model_dir)#这里暂时取消训练，直接载入已训练模型



def DRCRN_S_encoder(pickle_map, models_dir,
                     encoder_model_name, encoder_hyperparams_file,
                     b_encoder_hyperparm_tuning):

    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    test_data = pickle_map['test_data']
    scaling_data = pickle_map['scaling_data']

    training_processed = get_processed_data(training_data, scaling_data)
    validation_processed = get_processed_data(validation_data, scaling_data)
    test_processed = get_processed_data(test_data, scaling_data)

    # if b_encoder_hyperparm_tuning:

    fit_DRCRN_S_encoder(dataset_train=training_processed, dataset_val=validation_processed,
                    model_name=encoder_model_name, model_dir=models_dir,
                    hyperparams_file=encoder_hyperparams_file, b_hyperparam_opt=b_encoder_hyperparm_tuning)
    # else:
    #     pass

    encoder = load_trained_model(validation_processed, encoder_hyperparams_file, encoder_model_name, models_dir)
    mean_mse, mse = encoder.evaluate_predictions(test_processed)

    rmse = (np.sqrt(np.mean(mse))) / 1150 * 100  # Max tumour volume = 1150

    return rmse
