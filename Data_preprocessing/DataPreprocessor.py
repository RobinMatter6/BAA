import pandas as pd
from typing import List, Dict
import os
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass

    def convert_to_datetime(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        date_formats = [
            '%Y%m%d%H%M',
            '%d.%m.%Y %H:%M',
            '%Y-%m-%d',
            '%y%m%d',
            '%d-%m-%y',
            '%m/%d/%y',
            '%m-%d-%y',
            '%d/%m/%y',
            '%b %d, %y',
            '%y.%m.%d'
        ]
        for col in dataframe.columns:
            for date_format in date_formats:
                try:
                    dataframe[col] = pd.to_datetime(dataframe[col], format=date_format)
                    break
                except (ValueError, TypeError):
                    continue
        return dataframe

    def ensure_unique_single_word_values(self, dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
        dataframe[column_name] = dataframe[column_name].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else x)
        return dataframe

    def create_category_dictionaries(self, dataframe: pd.DataFrame, column_name: str) -> Dict[str, pd.DataFrame]:
        dataframe_grouped = dataframe.groupby(column_name)
        dataframe_dict = {name: group.reset_index(drop=True) for name, group in dataframe_grouped}
        for df in dataframe_dict.values():
            df.drop(columns=[column_name], inplace=True)
        return dataframe_dict

    def prepare_visitor_dataframe(self, input_dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        input_dataframe = input_dataframe[(input_dataframe['period'] >= 710) & (input_dataframe['period'] <= 740)]
        input_dataframe = input_dataframe[["DATUM UTC", "counter", "internal_sensor_id"]]
        input_dataframe = input_dataframe.rename(columns={"DATUM UTC": "time"})
        input_dataframe = input_dataframe.rename(columns={"counter": "visitors"})
        input_dataframe = self.convert_to_datetime(input_dataframe)
        input_dataframe = self.ensure_unique_single_word_values(input_dataframe, "internal_sensor_id")
        return self.create_category_dictionaries(input_dataframe, "internal_sensor_id")

    def load_multiple_dfs(self, csv_file_paths: List[str], csv_delimiter: str) -> List[pd.DataFrame]:
        dataframe_list = []
        for path in csv_file_paths:
            dataframe_list.append(pd.read_csv(path, delimiter=csv_delimiter))
        return dataframe_list
    
    def remove_unnecessary_meteo_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        data_quality_columns = [col for col in dataframe.columns if col.startswith('q')]
        dataframe = dataframe.drop(columns=data_quality_columns, errors='ignore')
        dataframe = dataframe.drop(columns=["stn"], errors='ignore')
        return dataframe
    
    def rename_meteo_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        unit_code_map = {
            "Absolute Humidity (g/m³)": "uto200s0", 
            "Air Pressure (hPa)": "prestas0", 
            "Air Temperature (°C)": "tre200s0",
            "Precipitation (mm)": "rre150z0",
            "Precipitation Duration (min)": "rco150z0",
            "Snow Depth (cm)": "htoauts0",
            "Sunshine Duration (min)": "sre000z0",
            "Wind Speed (km/h)": "fu3010z0"
        }
        reverse_map = {v: k for k, v in unit_code_map.items()}
        dataframe.rename(columns=reverse_map, inplace=True)
        return dataframe
    
    def align_time_ranges(self, dataframes: List[pd.DataFrame], on_column: str) -> List[pd.DataFrame]:
        start_times = [df[on_column].iloc[0] for df in dataframes]
        end_times = [df[on_column].iloc[-1] for df in dataframes]
        common_start = max(start_times)
        common_end = min(end_times)
        for i in range(len(dataframes)):
            dataframes[i] = dataframes[i][(dataframes[i][on_column] >= common_start) & 
                                        (dataframes[i][on_column] <= common_end)]
        return dataframes

    
    def merge_multiple_dfs(self, dataframes: List[pd.DataFrame], on_column: str) -> pd.DataFrame:
        if not dataframes:
            return pd.DataFrame()
        dataframes = self.align_time_ranges(dataframes, on_column)
        merged_dataframe = dataframes[0]
        for dataframe in dataframes[1:]:
            merged_dataframe = pd.merge_asof(
                merged_dataframe.sort_values(on_column),
                dataframe.sort_values(on_column),
                on=on_column,
                direction='nearest',
                tolerance=pd.Timedelta('10min')
            )
        return merged_dataframe

    def merge_visitor_with_meteo(self, visitor_dataframe: pd.DataFrame, meteo_dataframe: pd.DataFrame) -> pd.DataFrame:
        merged_dataframe = self.merge_multiple_dfs([meteo_dataframe, visitor_dataframe], "time")
        column_order = ['time', 'visitors'] + [col for col in merged_dataframe.columns if col not in ['time', 'visitors']]
        return merged_dataframe.reindex(columns=column_order)

    def prepare_meteo_dataframes(self, meteo_csv_file_paths: List[str], csv_delimiter: str) -> pd.DataFrame:
        dataframe_list = self.load_multiple_dfs(meteo_csv_file_paths, csv_delimiter)
        for i, dataframe in enumerate(dataframe_list):
            dataframe_list[i] = self.rename_meteo_columns(dataframe_list[i])
            dataframe_list[i] = self.remove_unnecessary_meteo_columns(dataframe) 
            dataframe_list[i] = self.convert_to_datetime(dataframe_list[i])
        return self.merge_multiple_dfs(dataframe_list, "time")

    def list_csv_files_in_directory(self, directory: str) -> List[str]:
        csv_file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    csv_file_paths.append(full_path)
        return csv_file_paths

    def convert_all_to_numeric(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for col in dataframe.columns:
            if not pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        return dataframe
    
    
    def standardize_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        print(dataframe.dtypes)
        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
        standardized_dataframe = dataframe.copy()
        standardized_dataframe[numeric_columns] = (dataframe[numeric_columns] - dataframe[numeric_columns].mean()) / dataframe[numeric_columns].std()
        return standardized_dataframe
    
    def remove_outliers_z_scores(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            mean = dataframe[col].mean()
            std = dataframe[col].std()
            z_scores = (dataframe[col] - mean) / std
            dataframe = dataframe[(z_scores >= -3) & (z_scores <= 3)]
        return dataframe

    #IQR is to aggressive
    def remove_outliers(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            dataframe = dataframe[(dataframe[col] >= lower_bound) & (dataframe[col] <= upper_bound)]
        return dataframe

    def log_transform_skewed_columns(self, dataframe: pd.DataFrame, skew_threshold: float = 0.5) -> pd.DataFrame:
        for col in dataframe.select_dtypes(include=['float64', 'int64']).columns:
            skewness = dataframe[col].skew()
            if skewness > skew_threshold:
                if (dataframe[col] <= 0).any():
                    dataframe[col] = np.log1p(dataframe[col] - dataframe[col].min() + 1)
                else:
                    dataframe[col] = np.log(dataframe[col])
        return dataframe

    def prepare_data_for_ml(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.remove_outliers_z_scores(dataframe)
        dataframe = self.log_transform_skewed_columns(dataframe)
        dataframe = self.remove_outliers_z_scores(dataframe)
        dataframe = self.standardize_dataframe(dataframe)
        return dataframe

    def get_merged_visitor_meteo_data(self, visitor_dataframe: pd.DataFrame, 
                                   meteo_csv_file_paths: List[str] = None, 
                                   meteo_csv_directory: str = None,
                                   prepare_for_ml: bool = False) -> Dict[str, pd.DataFrame]:
        if (meteo_csv_file_paths is None) == (meteo_csv_directory is None):
            raise ValueError("Provide either meteo_csv_file_paths or meteo_csv_directory, but not both.")
        visitor_dataframes = self.prepare_visitor_dataframe(visitor_dataframe)
        meteo_csv_file_paths = meteo_csv_file_paths or self.list_csv_files_in_directory(meteo_csv_directory)
        meteo_dataframe = self.prepare_meteo_dataframes(meteo_csv_file_paths, ";")
        merged_data = {
            key: self.convert_all_to_numeric(self.merge_visitor_with_meteo(visitor_df, meteo_dataframe))
            for key, visitor_df in visitor_dataframes.items()
        }
        return {key: self.prepare_data_for_ml(df) for key, df in merged_data.items()} if prepare_for_ml else merged_data


if __name__ == "__main__":
    preprocessing = DataPreprocessor()
    visitor_data = pd.read_csv("BesucherMessungExport.csv")
    meteo_csv_directory = "./exogen_data"
    merged_visitor_meteo_dataframes = preprocessing.get_merged_visitor_meteo_data(visitor_data, meteo_csv_directory=meteo_csv_directory, prepare_for_ml=True)
    print(merged_visitor_meteo_dataframes)

    for key, df in merged_visitor_meteo_dataframes.items():
            print(f"Testing DataFrame for {key}...")
            
            # Check for skewness
            skewness = df.select_dtypes(include=['float64', 'int64']).skew()
            print(f"Skewness for {key}:\n{skewness}\n")
            
            # Check for outliers
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            outlier_counts = {}
            
            for col in numeric_columns:
                mean = df[col].mean()
                std = df[col].std()
                z_scores = (df[col] - mean) / std
                outlier_counts[col] = ((z_scores < -3) | (z_scores > 3)).sum()
            
            print(f"Outlier counts for {key}:\n{outlier_counts}\n")
            
            # Check for standardization
            means = df[numeric_columns].mean()
            stds = df[numeric_columns].std()
            is_standardized = all(np.isclose(means, 0, atol=1e-1)) and all(np.isclose(stds, 1, atol=1e-1))
            print(f"Is {key} standardized? {is_standardized}\n")