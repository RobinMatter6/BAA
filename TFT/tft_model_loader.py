import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #select GPU
import ast
import re
import csv
from typing import List
import pandas as pd
import tensorflow.compat.v1 as tf
from data_formatters.base import DataTypes, InputTypes
from data_formatters.visitors import VisitorsFormatter
import libs.tft_model

class TFTModelLoader:
    def __init__(self, data_formatter_class, base_path: str):
        self.ModelClass = libs.tft_model.TemporalFusionTransformer
        self.DataFormatterClass = data_formatter_class
        self.base_path = base_path.rstrip('/')
        self.configs = []
        self.train = None
        self.valid = None
        self.test = None

        self.config_csv_path = os.path.join(
            self.base_path,
            "saved_models",
            "visitors",
            "main",
            "params.csv"
        )
        self.data_csv_path = os.path.join(
            self.base_path,
            "data",
            "visitors",
            "Rathausquai.csv"
        )
        self.results_csv_path = os.path.join(
            self.base_path,
            "saved_models",
            "visitors",
            "main",
            "results.csv"
        )

    def parse_config(self):
        try:
            df = pd.read_csv(self.config_csv_path, index_col=0)
            df = df.transpose()
            df.reset_index(drop=True, inplace=True)
            for idx, row in df.iterrows():
                parsed_dict = {}
                for key, value in row.items():
                    if pd.isna(value):
                        continue
                    value = str(value).strip('"')
                    try:
                        if key == "column_definition":
                            sanitized_value = re.sub(r"<([^>]+)>", r"'\1'", value)
                            raw_definitions = ast.literal_eval(sanitized_value)
                            column_definitions = []
                            for col in raw_definitions:
                                name, dtype_str, input_type_str = col
                                dtype_member = dtype_str.split(':')[0].split('.')[-1]
                                input_type_member = input_type_str.split(':')[0].split('.')[-1]
                                dtype = DataTypes[dtype_member]
                                input_type = InputTypes[input_type_member]
                                column_definitions.append((name, dtype, input_type))
                            parsed_dict[key] = column_definitions
                        else:
                            if isinstance(value, str) and (value.startswith('[') or value.startswith('(')):
                                parsed_value = ast.literal_eval(value)
                            elif isinstance(value, str) and re.match(r'^-?\d+\.\d*$', value):
                                parsed_value = float(value)
                            elif isinstance(value, str) and re.match(r'^-?\d+$', value):
                                parsed_value = int(value)
                            else:
                                parsed_value = value
                            parsed_dict[key] = parsed_value
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing key '{key}' in config {idx}: {e}")
                        parsed_dict[key] = value

                self.configs.append(parsed_dict)

            print(f"Parsed {len(self.configs)} configurations successfully.")

        except FileNotFoundError:
            print(f"Configuration file not found at {self.config_csv_path}.")
            raise
        except Exception as e:
            print(f"Failed to parse configuration file: {e}")
            raise

    def load_data(self):
        try:
            raw_data = pd.read_csv(self.data_csv_path, index_col=0)
            data_formatter = self.DataFormatterClass()
            self.train, self.valid, self.test = data_formatter.split_data(raw_data)
            print("Data loaded and split successfully.")
        except FileNotFoundError:
            print(f"Data file not found at {self.data_csv_path}.")
            raise
        except Exception as e:
            print(f"Failed to load and split data: {e}")
            raise

    def load_model(self, model_id: int):
        try:
            config = self.configs[model_id]
            idx = model_id
            model = self.ModelClass(config, use_cudnn="no")
            model_folder = config.get('model_folder')
            if model_folder:
                if not os.path.isabs(model_folder):
                    model_folder = os.path.join(self.base_path, model_folder)
                model.load(model_folder)
                print(f"Model {idx} loaded successfully from {model_folder}.")
            else:
                print(f"No 'model_folder' specified for config {idx}. Skipping weight loading.")
        except IndexError:
            print(f"Model ID {model_id} is out of range. Available models: {len(self.configs)}")
            raise
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            raise
        return model


    def find_lowest_loss_position(self, csv_file_path: str) -> int:
        loss_values = []
        try:
            with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if not row:
                        continue 
                    if row[0].strip().lower() == 'loss':
                        try:
                            loss_values = [float(value) for value in row[1:] if value]
                        except ValueError as e:
                            raise ValueError(f"Error converting loss values to float: {e}")
                        break

            if not loss_values:
                raise ValueError("No loss values found in the CSV.")
            min_loss = min(loss_values)
            position = loss_values.index(min_loss)
            print(f"Lowest loss value: {min_loss} at position: {position}")
            return position
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {csv_file_path} does not exist.")
        except Exception as e:
            raise e


def main():
    test_base_path = "/opt/BAA/TFT/models/hyper_parameter_tuning/week_60min/pod_no_ex_week_60min"
    model_loader = TFTModelLoader(
        data_formatter_class=VisitorsFormatter,
        base_path=test_base_path
    )
    try:
        print("Parsing configuration...")
        model_loader.parse_config()
    except Exception as e:
        print(f"An error occurred while parsing configurations: {e}")
        return
if __name__ == "__main__":
    main()