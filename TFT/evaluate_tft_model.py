import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #select GPU
from typing import Dict
import pandas as pd
from TFT.tft_model_loader import TFTModelLoader
from data_formatters.visitors import VisitorsFormatter
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

class TFTModelEvaluator:
    def __init__(self, model_loader: TFTModelLoader, plot_dir: str = "evaluation_plots"):
        self.model_loader = model_loader
        self.plot_dir = plot_dir
        self._ensure_plot_dir_exists()

    def _ensure_plot_dir_exists(self):
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def evaluate_model(self, model, pod_name: str) -> float:
        predictions = model.predict(self.model_loader.valid, return_targets=True)
        print(f"prediction: {predictions}")
        p50_df = predictions['p50']
        targets_df = predictions['targets']
        p50_last = p50_df.iloc[:, -1]
        targets_last = targets_df.iloc[:, -1]
        mae = mean_absolute_error(targets_last, p50_last)
        merge_df = pd.DataFrame({
            'forecast_time': p50_df['forecast_time'],
            'p50_last': p50_last,
            'targets_last': targets_last
        })
        self.plot_predictions_vs_actual(
            merge_df,
            pod_name=pod_name,
            n_subplots=2
        )
        return mae
    def run(self, pod_name: str) -> float:
        self.model_loader.parse_config()
        self.model_loader.load_data()
        lowest_loss_position = self.model_loader.find_lowest_loss_position(
            self.model_loader.results_csv_path
        )
        print(f"\nUse model position: {lowest_loss_position}")
        try:
            model = self.model_loader.load_model(lowest_loss_position)
            mae = self.evaluate_model(model, pod_name)
            return mae
        except Exception as e:
            print(f"Error occured while loading model: {e}")
            raise
    def plot_predictions_vs_actual(
        self,
        merged_df: pd.DataFrame,
        pod_name: str,
        n_subplots: int = 1
    ):
        if len(merged_df) > 1000:
            merged_df = merged_df.iloc[:1000]
        if n_subplots > 1:
            segment_length = len(merged_df) // n_subplots
        else:
            segment_length = len(merged_df)
        _, axes = plt.subplots(n_subplots, 1, figsize=(20, 6 * n_subplots), dpi=300)
        if n_subplots == 1:
            axes = [axes]
        for i in range(n_subplots):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = merged_df.iloc[start_idx:end_idx]

            ax = axes[i]
            ax.plot(
                segment['forecast_time'],
                segment['p50_last'],
                label='Vorhersage'
            )
            ax.plot(
                segment['forecast_time'],
                segment['targets_last'],
                label='Tatsächlich'
            )
            ax.set_xlabel('forecast_time')
            ax.set_ylabel('Visitors')
            ax.set_title(f'Vorhersage vs. Tatsächlich (Segment {i+1})')
            ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, f"{pod_name}_predictions_vs_actual.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")

def main():
    base_path = "/opt/BAA/TFT/models/hyper_parameter_tuning/day_60min"
    
    pods_to_be_evaluated = [
        "pod_no_ex_day_60min",
        "pod_just_absolute_humidity_day_60min",
        "pod_just_air_temperature_day_60min",
        "pod_just_sunshine_duration_day_60min",
        "pod_just_air_pressure_day_60min",
        "pod_just_precipitation_duration_day_60min",
        "pod_just_wind_speed_day_60min",
    ]

    # pods_to_be_evaluated = [
    #     "pod_no_ex_week_60min",
    #     "pod_just_absolute_humidity_week_60min",
    #     "pod_just_air_temperature_week_60min",
    #     "pod_just_sunshine_duration_week_60min",
    #     "pod_just_air_pressure_week_60min",
    #     "pod_just_precipitation_duration_week_60min",
    #     "pod_just_wind_speed_week_60min",
    #     "pod_just_precipitation_duration_sunshine_duration_week_60min",
    # ]

    results: Dict[str, float] = {}
    for pod in pods_to_be_evaluated:
        tf.reset_default_graph()
        with tf.Session() as sess:
            print(f"\nEvaluating pod: {pod}")
            pod_path = os.path.join(base_path, pod)
            if not os.path.isdir(pod_path):
                print(f"Pod directory'{pod}' doesn't exist: {pod_path}")
                continue
            model_loader = TFTModelLoader(
                data_formatter_class=VisitorsFormatter,
                base_path=pod_path
            )
            evaluator = TFTModelEvaluator(model_loader=model_loader)
            try:
                mae = evaluator.run(pod_name=pod)
                results[pod] = mae
                print(f"Pod '{pod}' has MAE: {mae}")
            except Exception as e:
                print(f"Error occured while evaluation pod: '{pod}' following occured: {e}")
    print("Evaluation results:")
    for pod_name, mae_value in results.items():
        print(f"Pod: {pod_name}, MAE: {mae_value}")

if __name__ == "__main__":
    main()
