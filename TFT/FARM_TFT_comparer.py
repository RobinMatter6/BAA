import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #select GPU
import re
import pandas as pd
from typing import Dict
from data_formatters.visitors import VisitorsFormatter
from TFT.tft_model_loader import TFTModelLoader
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import json

def load_FARM_importance(path, lookback, lookforward):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    
    shortened_data = {}
    print(f"lookforward: {lookforward}")
    print(f"lookback: {lookback}")
    
    for feature, importance_list in data.items():
        if not isinstance(importance_list, list):
            print(f"Warning: Importance for feature '{feature}' is not a list. Skipping.")
            continue
        if lookforward > 1:
            sliced_importance = importance_list[lookback:-(lookforward-1)]
        else:
            sliced_importance = importance_list[lookback:]
        
        shortened_data[feature] = sliced_importance
    return shortened_data

def get_TFT_feature_importance(weights):
    decoder_self_attn = weights['decoder_self_attn'] 
    historical_flags = weights['historical_flags']  
    future_flags = weights['future_flags'] 
    decoder_self_attn_slice = decoder_self_attn[0, :, -1, :]
    N = decoder_self_attn_slice.shape[0]
    T = decoder_self_attn_slice.shape[1]
    num_feats = future_flags.shape[2] 
    lookback = historical_flags.shape[1] 
    print(f"history: {lookback}")
    feature_table = np.zeros((N, num_feats, T))

    for i in range(N):
        attn_vec = decoder_self_attn_slice[i]
        attn_hist = attn_vec[:lookback]
        attn_fut  = attn_vec[lookback:]
        hist_flags_i = historical_flags[i, :, :7]
        fut_flags_i  = future_flags[i]
        hist_weighted = hist_flags_i * attn_hist[:, None]
        fut_weighted  = fut_flags_i  * attn_fut[:, None]
        for f in range(num_feats):
            feature_table[i, f, :lookback] = hist_weighted[:, f]
            feature_table[i, f, lookback:] = fut_weighted[:, f]
    return np.sum(feature_table, axis=2)

def r2_TFT_FARM(
    FARM_file_path,
    weights,
    pod,
    feature_names,
    feature_map,
):
    TFT_feature_importance = get_TFT_feature_importance(
        weights=weights
    )
    lookback = weights["historical_flags"].shape[1]
    lookforward = weights["future_flags"].shape[1]
    num_features = weights["future_flags"].shape[2]

    FARM_importance_dict = load_FARM_importance(FARM_file_path, lookback, lookforward)
    if FARM_importance_dict is None:
        return {}

    feature_indices = list(range(num_features))
    filtered_feature_map = {idx: feature_map(idx) for idx in feature_indices}
    TFT_feature_importance = TFT_feature_importance
    FARM_importance_filtered = {
        feature: FARM_importance_dict.get(feature, None) for feature in filtered_feature_map.values()
    }

    tft_std_list_for_df = []
    farm_std_list_for_df = []
    valid_features_r2 = {}

    for idx, feature in filtered_feature_map.items():
        farm_importance = FARM_importance_filtered.get(feature)
        tft_importance = TFT_feature_importance[:, idx]
        
        prediction_horizon = weights["future_flags"].shape[1]
        tft_importance = tft_importance[:-prediction_horizon]
        farm_importance = farm_importance[prediction_horizon:]

        tft_importance = tft_importance.reshape(-1, 1) 
        farm_importance = np.array(farm_importance).reshape(-1, 1)

        tft_scaler = StandardScaler()
        tft_importance_standardized = tft_scaler.fit_transform(tft_importance)

        farm_scaler = StandardScaler()
        farm_importance_standardized = farm_scaler.fit_transform(farm_importance)

        tft_std_list_for_df.append(tft_importance_standardized.flatten())
        farm_std_list_for_df.append(farm_importance_standardized.flatten())

        r2 = r2_score(tft_importance_standardized, farm_importance_standardized)

        valid_features_r2[feature] = r2

        print(f"R² for feature '{feature}': {r2}")

    missing_features = [
        feature for feature, value in FARM_importance_filtered.items() if value is None
    ]
    if missing_features:
        print(f"\nThe following features are missing in FARM_importance_dict and were skipped: {missing_features}")
    print(f"\nR^2 values for pod {pod}:")
    for feature, r2_value in valid_features_r2.items():
        print(f"Feature: {feature}, R^2: {r2_value}")
    return valid_features_r2

def parse_pod_name(pod_name: str) -> Dict[str, str]:
    pattern = r'^pod_(with_ex|no_ex|just_[a-z_]+)_(day|week)_(30min|60min)$'
    match = re.match(pattern, pod_name)
    ex_flag, time_period, time_interval = match.groups()
    return {
        'ex_flag': ex_flag,
        'time_period': time_period,
        'time_interval': time_interval
    }
def main():
    base_path = "/opt/BAA/TFT/models/hyper_parameter_tuning/day_60min"
    pods_to_be_compared = [
        "pod_just_absolute_humidity_day_60min",
        "pod_just_air_temperature_day_60min",
        "pod_just_sunshine_duration_day_60min",
        "pod_just_air_pressure_day_60min",
        "pod_just_precipitation_duration_day_60min",
        "pod_just_wind_speed_day_60min",
    ]
    # pods_to_be_compared = [   
    #     # "pod_just_absolute_humidity_week_60min",   
    #     "pod_just_air_temperature_week_60min",
    #     "pod_just_sunshine_duration_week_60min",
    #     "pod_just_air_pressure_week_60min",
    #     "pod_just_precipitation_duration_week_60min",
    #     "pod_just_wind_speed_week_60min",
    # ]
    
    all_r2_results = {}
    for pod in pods_to_be_compared:
        tf.reset_default_graph()
        with tf.Session() as sess:
            print(f"\nEvaluating pod: {pod}")
            pod_path = os.path.join(base_path, pod)
            if not os.path.isdir(pod_path):
                print(f"Directory for pod '{pod}' does not exist at path: {pod_path}")
                continue
            config = parse_pod_name(pod)
            ex_flag = config['ex_flag']
            raw_feature = ex_flag[5:]
            parts = raw_feature.split('_')
            external_feature_name = ' '.join([p.capitalize() for p in parts])
            feature_names = [
                'minute_of_hour',
                'hour_of_day',
                'day_of_week',
                'week_of_year',
                'month_of_year',
                'day_of_month',
                external_feature_name
            ]
            feature_map = {i: n for i, n in enumerate(feature_names)}

            model_loader = TFTModelLoader(
                data_formatter_class=VisitorsFormatter,
                base_path=pod_path
            )
            model_loader.parse_config()
            model_loader.load_data()
            lowest_loss_position = model_loader.find_lowest_loss_position(
                model_loader.results_csv_path
            )
            model = model_loader.load_model(model_id=lowest_loss_position)
            combined_df = pd.concat(
                [model_loader.train, model_loader.valid, model_loader.test],
                axis=0,
                ignore_index=True
            )
            combined_df = combined_df[1:]
            weights = model.get_attention(combined_df)
            FARM_file_path = "../FARM/FARM_importance.json"
            r2_dict = r2_TFT_FARM(
                FARM_file_path=FARM_file_path,
                weights=weights,
                pod=pod,
                feature_names=feature_names,
                feature_map=lambda x: feature_map[x],
            )
            all_r2_results[pod] = r2_dict

        feature_width = 40
        r2_width = 15
        header = f"{'Pod':<60}"
        header += f"{'Feature':<{feature_width}}"
        header += f"{'R²':<{r2_width}}"
        print("\n" + "="* (60 + feature_width + r2_width))
        print(header)
        print("="* (60 + feature_width + r2_width))

        for pod in pods_to_be_compared:
            if pod not in all_r2_results:
                continue
            for feature in all_r2_results[pod]:
                r2 = all_r2_results[pod][feature]
                line = f"{pod:<60}"
                line += f"{feature:<{feature_width}}"
                line += f"{r2:<{r2_width}.4f}"
                print(line)
        print("="* (60 + feature_width + r2_width))

if __name__ == "__main__":
    main()
