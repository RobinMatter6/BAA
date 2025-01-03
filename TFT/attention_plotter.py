import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #select GPU
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from data_formatters.visitors import VisitorsFormatter
from TFT.tft_model_loader import TFTModelLoader  
import tensorflow.compat.v1 as tf
import numpy as np

def get_feature_importance_overall(weights):
    decoder_self_attn = weights['decoder_self_attn'] 
    historical_flags = weights['historical_flags']  
    future_flags = weights['future_flags'] 
    decoder_self_attn_slice = decoder_self_attn[0, :, -1, :]
    N = decoder_self_attn_slice.shape[0]
    T = decoder_self_attn_slice.shape[1]
    num_feats = future_flags.shape[2] 
    lookback = historical_flags.shape[1] 
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


def plot_attention_of_feature_for_average_day(
    weights,
    pod,
    feature_map,
):
    base_save_dir = os.path.join("attention_plots", pod)
    os.makedirs(base_save_dir, exist_ok=True) 
    feature_importance_overall = get_feature_importance_overall(
        weights=weights,
    )
    time = weights['time']
    historical_flags = weights['historical_flags']
    current = historical_flags.shape[1] + 1
    time = time[:, current]
    time = pd.to_datetime(time)
    number_of_features = weights["future_flags"].shape[2]
    prediction_horizon = weights["future_flags"].shape[1]
    print(prediction_horizon)
    time = time[prediction_horizon:]
    for feature_idx in range(number_of_features):
        feature_name = feature_map.get(feature_idx, f"Feature_{feature_idx}_Importance")
        feature_values = feature_importance_overall[:,feature_idx]
        feature_values = feature_values[:-prediction_horizon]
        df = pd.DataFrame({
            'time': time,
            'importance': feature_values
        })
        df['hour'] = df['time'].dt.hour
        df_avg_by_hour = df.groupby('hour')['importance'].mean()
        plt.figure(figsize=(12, 6))
        plt.plot(df_avg_by_hour.index, df_avg_by_hour.values, marker='o', label=f'{feature_name} Relevance')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Relevance (Attention Weight)')
        plt.title(f'Relevance of {feature_name} Over an Average 24h Day')
        plt.xticks(range(0, 24))
        y_max = df_avg_by_hour.max()
        if y_max == 0:
            plt.ylim(0, 1)
        else:
            padding = y_max * 0.05
            plt.ylim(0, y_max + padding)
        plt.tight_layout()
        plt.legend()
        safe_feature_name = re.sub(r'[\\/*?:"<>|]', "_", feature_name)
        filename = f"{safe_feature_name}_relevance_average_day.png"
        save_path = os.path.join(base_save_dir, filename)
        plt.savefig(save_path)
        print(f"Saved average-day plot for {feature_name} to {save_path}")
        plt.close()

def parse_pod_name(pod_name: str) -> Dict[str, str]:
    pattern = r'^pod_(with_ex|no_ex|just_[a-z_]+)_(day|week)_(30min|60min)$'
    match = re.match(pattern, pod_name)
    if not match:
        raise ValueError(f"Pod name '{pod_name}' does not match the expected pattern.")
    
    ex_flag, time_period, time_interval = match.groups()
    return {
        'ex_flag': ex_flag,
        'time_period': time_period,
        'time_interval': time_interval
    }

def main():
    base_path = "/opt/BAA/TFT/models/hyper_parameter_tuning/day_60min"
    pods_to_be_plotted = [
        "pod_no_ex_day_60min",
        "pod_just_absolute_humidity_day_60min",
        "pod_just_air_temperature_day_60min",
        "pod_just_sunshine_duration_day_60min",
        "pod_just_air_pressure_day_60min",
        "pod_just_precipitation_duration_day_60min",
        "pod_just_wind_speed_day_60min",
    ]

    # pods_to_be_plotted = [
    #     "pod_just_absolute_humidity_week_60min",
    #     "pod_just_air_temperature_week_60min",
    #     "pod_just_sunshine_duration_week_60min",
    #     "pod_just_air_pressure_week_60min",
    #     "pod_just_precipitation_duration_week_60min",
    #     "pod_just_wind_speed_week_60min",
    # ]

    for pod in pods_to_be_plotted:
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
            weights = model.get_attention(combined_df)
            plot_attention_of_feature_for_average_day(
                weights, pod, feature_map,
            )
if __name__ == "__main__":
    main()