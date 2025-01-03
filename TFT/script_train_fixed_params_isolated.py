# coding=utf-8 
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params.py {expt_name} {output_folder} [--use_gpu yes/no] [--gpu_id ID]

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved.
  --use_gpu: Whether to use GPU for training ('yes' or 'no'). Default is 'no'.
  --gpu_id: ID of the GPU to use. Default is '0'.
"""

import argparse
import datetime as dte
import os
import sys

def parse_args_and_set_gpu():
    parser = argparse.ArgumentParser(description="Train TFT model with fixed parameters.")
    parser.add_argument(
        "expt_name",
        metavar="e",
        type=str,
        nargs="?",
        default="volatility",
        help="Experiment Name. Default='volatility'."
    )
    parser.add_argument(
        "output_folder",
        metavar="f",
        type=str,
        nargs="?",
        default=".",
        help="Path to folder for experiment output. Default='.'."
    )
    parser.add_argument(
        "use_gpu",
        metavar="g",
        type=str,
        choices=["yes", "no"],
        default="no",
        help="Whether to use GPU for training. Choices: 'yes', 'no'. Default='no'."
    )
    parser.add_argument(
        "gpu_id",
        type=str,
        default="0",
        help="ID of the GPU to use. Default='0'."
    )
    
    args = parser.parse_args()
    
    expt_name = args.expt_name
    output_folder = args.output_folder
    use_gpu = args.use_gpu.lower() == "yes"
    gpu_id = args.gpu_id
    
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Using GPU ID: {gpu_id}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU for training.")
    
    return expt_name, output_folder, use_gpu, gpu_id

expt_name, output_folder, use_gpu, gpu_id = parse_args_and_set_gpu()

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer

def main(expt_name,
         use_gpu,
         gpu_id,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):
    num_repeats = 1

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from "
            "AbstractDataFormatter! Type={}".format(type(data_formatter))
        )

    default_keras_session = tf.keras.backend.get_session()

    if use_gpu:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.visible_device_list = '0'
    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("*** Training from defined parameters for {} ***".format(expt_name))

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    print("*** Loading hyperparameter manager ***")
    opt_manager = HyperparamOptManager(
        {k: [params[k]] for k in params},
        fixed_params,
        model_folder
    )

    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_loss = np.inf
    for _ in range(num_repeats):

        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

            tf.keras.backend.set_session(sess)

            params = opt_manager.get_next_parameters()
            model = ModelClass(params, use_cudnn=use_gpu)

            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.global_variables_initializer())
            model.fit()

            val_loss = model.evaluate()

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss

            tf.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)

        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            return data[
                [
                    col for col in data.columns
                    if col not in {"forecast_time", "identifier"}
                ]
            ]

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets),
            extract_numerical_data(p50_forecast),
            0.5
        )
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets),
            extract_numerical_data(p90_forecast),
            0.9
        )

        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()
    ))

if __name__ == "__main__":
    name = expt_name
    output_folder = output_folder
    use_tensorflow_with_gpu = use_gpu
    gpu_id = gpu_id

    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    main(
        expt_name=name,
        use_gpu=use_tensorflow_with_gpu,
        gpu_id=gpu_id,
        model_folder=os.path.join(config.model_folder, "fixed"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=False
    )
