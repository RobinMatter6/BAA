import pandas as pd
from typing import List, Dict
import os
import numpy as np
from datetime import timedelta
import numpy as np
import pytz
from sklearn.preprocessing import PowerTransformer

class DataPreprocessor:
    def __init__(self):
        pass

    def convert_to_datetime(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        date_formats = [
            '%Y%m%d%H%M',
            '%d.%m.%Y %H:%M',
        ]
        for col in dataframe.columns:
            for date_format in date_formats:
                try:
                    dataframe[col] = pd.to_datetime(dataframe[col], format=date_format)
                    break
                except (ValueError, TypeError):
                    continue
        return dataframe

    def convert_utc_to_local_time(self, dataframe: pd.DataFrame, time_column: str = "time") -> pd.DataFrame:
        dataframe[time_column] = dataframe[time_column].dt.tz_localize(pytz.utc)
        local_tz = pytz.timezone("Europe/Zurich")
        dataframe[time_column] = dataframe[time_column].dt.tz_convert(local_tz)
        dataframe[time_column] = dataframe[time_column].dt.tz_localize(None)
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

    def prepare_visitor_dataframe(self, input_dataframe: pd.DataFrame, prepare_for_tft: bool) -> Dict[str, pd.DataFrame]:
        input_dataframe = input_dataframe[(input_dataframe['period'] >= 710) & (input_dataframe['period'] <= 740)]
        input_dataframe = input_dataframe[["DATUM UTC", "counter", "internal_sensor_id"]]
        input_dataframe = input_dataframe.rename(columns={"DATUM UTC": "time"})
        input_dataframe = input_dataframe.rename(columns={"counter": "visitors"})
        input_dataframe = self.convert_to_datetime(input_dataframe)
        input_dataframe = self.convert_utc_to_local_time(input_dataframe)
        input_dataframe = self.ensure_unique_single_word_values(input_dataframe, "internal_sensor_id")
        dataframe_dict = self.create_category_dictionaries(input_dataframe, "internal_sensor_id")
        for key in dataframe_dict:
            print(key)
            if prepare_for_tft:
                 dataframe_dict[key] = self.remove_outliers_with_rolling_window(dataframe_dict[key], time_column="time", value_column="visitors")
            dataframe_dict[key] = self.adjust_dataframe_time_series_nearest(
                dataframe_dict[key], time_column='time',
                expected_interval_minutes=12
            )
            print(dataframe_dict[key])
        return dataframe_dict


    def adjust_dataframe_time_series_nearest(self, df: pd.DataFrame, time_column: str, expected_interval_minutes: int) -> pd.DataFrame:
        import numpy as np
        from datetime import timedelta
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column).reset_index(drop=True)
        first_timestamp = df[time_column].iloc[0]
        last_timestamp = df[time_column].iloc[-1]
        start_time = first_timestamp - timedelta(minutes=(expected_interval_minutes / 2))
        parallel_series = pd.date_range(
            start=start_time,
            end=last_timestamp + timedelta(minutes=expected_interval_minutes),
            freq=f'{expected_interval_minutes}min'
        )
        times = df[time_column].values
        indices = np.searchsorted(parallel_series, times, side='left')
        indices_minus_one = np.maximum(indices - 1, 0)
        indices_plus_one = np.minimum(indices, len(parallel_series) - 1)
        times_lower = parallel_series[indices_minus_one]
        times_upper = parallel_series[indices_plus_one]
        diff_lower = np.abs(times - times_lower)
        diff_upper = np.abs(times - times_upper)
        choose_lower = diff_lower <= diff_upper
        final_indices = np.where(choose_lower, indices_minus_one, indices_plus_one)
        df['parallel_time'] = parallel_series[final_indices].values
        df_aggregated = df.groupby('parallel_time').agg({
            'visitors': 'last',
        }).reset_index().rename(columns={'parallel_time': 'time'})
        df_parallel = pd.DataFrame({'time': parallel_series})
        df_final = pd.merge(df_parallel, df_aggregated, on='time', how='left')
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns
        df_final[numeric_cols] = df_final[numeric_cols].interpolate(method='linear')
        return df_final


    def resample_to_resolution(self, df, resolution_minutes):
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        resolution = pd.Timedelta(minutes=resolution_minutes)
        half_resolution = resolution / 2
        start_time = df.index[0].ceil(f'{resolution_minutes}min')
        end_time = df.index[-1].floor(f'{resolution_minutes}min')
        if end_time < start_time:
            return pd.DataFrame()
        times = pd.date_range(start=start_time, end=end_time, freq=f'{resolution_minutes}min')
        results = []
        
        for t in times:
            window_start = t - half_resolution
            window_end = t + half_resolution
            mask = (df.index > window_start) & (df.index <= window_end)
            data_in_window = df.loc[mask]
            if not data_in_window.empty:
                averaged = data_in_window.mean()
                averaged['time'] = t
                results.append(averaged)
        result_df = pd.DataFrame(results)
        result_df.sort_values('time', inplace=True)
        result_df = result_df.reset_index(drop=True)
        cols = result_df.columns.tolist()
        cols.remove('time')
        cols.insert(0, 'time')
        result_df = result_df[cols]
        return result_df

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
            "Absolute Humidity": "uto200s0", 
            "Air Pressure": "prestas0", 
            "Air Temperature": "tre200s0",
            "Precipitation": "rre150z0",
            "Precipitation Duration": "rco150z0",
            "Snow Depth": "htoauts0",
            "Sunshine Duration": "sre000z0",
            "Wind Speed": "fu3010z0"
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

    def prepare_meteo_dataframes(self, meteo_csv_file_paths: List[str], csv_delimiter: str, prepare_for_tft) -> pd.DataFrame:
        dataframe_list = self.load_multiple_dfs(meteo_csv_file_paths, csv_delimiter)
        for i, dataframe in enumerate(dataframe_list):
            dataframe_list[i] = self.rename_meteo_columns(dataframe_list[i])
            dataframe_list[i] = self.remove_unnecessary_meteo_columns(dataframe) 
            dataframe_list[i] = self.convert_to_datetime(dataframe_list[i])
            dataframe_list[i] = self.convert_utc_to_local_time(dataframe_list[i])
            if prepare_for_tft:
                value_column = self.get_meteo_column(dataframe_list[i])
                dataframe_list[i] = self.remove_outliers_with_rolling_window(dataframe_list[i], time_column="time", value_column=value_column)
        return self.merge_multiple_dfs(dataframe_list, "time")

    def get_meteo_column(self, dataframe):
        meteo_columns = [
            "Absolute Humidity",
            "Air Pressure",
            "Air Temperature",
            "Precipitation",
            "Precipitation Duration",
            "Snow Depth",
            "Sunshine Duration",
            "Wind Speed"
        ]
        columns = dataframe.columns
        for column in columns:
            if column in meteo_columns:
                return column
        return None

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
                try:
                    dataframe[col] = pd.to_numeric(dataframe[col], downcast='integer', errors='raise')
                except (ValueError, TypeError):
                    dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        return dataframe
    
    def standardize_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
        standardized_dataframe = dataframe.copy()
        standardized_dataframe[numeric_columns] = (dataframe[numeric_columns] - dataframe[numeric_columns].mean()) / dataframe[numeric_columns].std()
        return standardized_dataframe
    
    def remove_outliers_with_rolling_window(self, dataframe: pd.DataFrame, time_column: str, value_column: str,
                                            multiplier: float = 0.5, mark_outliers: bool = False) -> pd.DataFrame:
        dataframe[time_column] = pd.to_datetime(dataframe[time_column])
        dataframe[value_column] = pd.to_numeric(dataframe[value_column], errors='coerce')
        dataframe['weekday'] = dataframe[time_column].dt.weekday
        dataframe['time_30min'] = dataframe[time_column].dt.floor('30min')
        outlier_flags = pd.Series(False, index=dataframe.index)
        outlier_count = 0
        for weekday in range(7):
            weekday_data = dataframe[dataframe['weekday'] == weekday]
            weekday_data_grouped = weekday_data.groupby('time_30min', as_index=True)[value_column]
            for time_window, group in weekday_data_grouped:
                Q1 = group.quantile(0.25)
                Q3 = group.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outliers = (group < lower_bound) | (group > upper_bound)
                outlier_flags.loc[group.index] = outliers
        if mark_outliers:
            dataframe['outlier'] = outlier_flags
            outlier_count = outlier_count + 1
        else:
            dataframe = dataframe[~outlier_flags]
        dataframe = dataframe.drop(columns=['weekday', 'time_30min'])
        print(f"Outliers: {outlier_count}")
        return dataframe
    
    # This method was used to check if the analysis results were affected because unskewing was not possible using log_transform_skewed_columns
    # It showed that the results did not improve with this method compared to log_transform_skewed_columns.
    def yeo_johnson_transform_skewed_columns(self, dataframe: pd.DataFrame, skew_threshold: float = 0.5) -> pd.DataFrame:
        pt = PowerTransformer(method='yeo-johnson')
        for col in dataframe.select_dtypes(include=['float64', 'int64']).columns:
            dataframe[col] = pt.fit_transform(dataframe[[col]])
        return dataframe



    def log_transform_skewed_columns(self, dataframe: pd.DataFrame, skew_threshold: float = 0.5, log_base: int = 10) -> pd.DataFrame:
        high_log_columns = ["Precipitation Duration", "Precipitation", "Snow Depth"]
        for col in dataframe.select_dtypes(include=['float64', 'int64']).columns:
            skewness = dataframe[col].skew()
            if skewness > skew_threshold:
                if col in high_log_columns:
                    print("high log: " + col)
                    if (dataframe[col] <= 0).any():
                        dataframe[col] = np.log1p(np.log10((dataframe[col] - dataframe[col].min() + 1)))
                    else:
                        dataframe[col] = np.log1p(np.log10((dataframe[col])))
                else:
                    if (dataframe[col] <= 0).any():
                        dataframe[col] = np.log1p(dataframe[col] - dataframe[col].min() + 1)
                    else:
                        dataframe[col] = np.log1p(dataframe[col])
        return dataframe


    def add_time_and_dummy_columns(self, dataframe: pd.DataFrame, time_column: str = "time") -> pd.DataFrame:
        dataframe["minute_of_hour"] = dataframe[time_column].dt.minute
        dataframe["hour_of_day"] = dataframe[time_column].dt.hour
        dataframe["day_of_week"] = dataframe[time_column].dt.dayofweek
        dataframe["day_of_month"] = dataframe[time_column].dt.day
        dataframe["week_of_year"] = dataframe[time_column].dt.isocalendar().week
        dataframe["month_of_year"] = dataframe[time_column].dt.month
        dataframe["dummy_id"] = 1
        dataframe["region"] = "Rathausquai"
        dataframe["date"] = dataframe["time"]
        return dataframe

    

    def prepare_data_for_tft(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.drop(["Snow Depth", "Precipitation"], axis=1, inplace=True)
        dataframe.dropna(inplace=True)
        dataframe = self.log_transform_skewed_columns(dataframe)
        #dataframe = self.standardize_dataframe(dataframe) Was deactivated because TFT has standartisation in DataFormatter
        dataframe = self.add_time_and_dummy_columns(dataframe)
        return dataframe


    def get_merged_visitor_meteo_data(self, visitor_dataframe: pd.DataFrame, 
                                   meteo_csv_file_paths: List[str] = None, 
                                   meteo_csv_directory: str = None,
                                   resolution_in_minutes: str = None,
                                   prepare_for_tft: bool = False) -> Dict[str, pd.DataFrame]:
        if (meteo_csv_file_paths is None) == (meteo_csv_directory is None):
            raise ValueError("Provide either meteo_csv_file_paths or meteo_csv_directory, but not both.")
        visitor_dataframes = self.prepare_visitor_dataframe(visitor_dataframe, prepare_for_tft)
        meteo_csv_file_paths = meteo_csv_file_paths or self.list_csv_files_in_directory(meteo_csv_directory)
        meteo_dataframe = self.prepare_meteo_dataframes(meteo_csv_file_paths, ";", prepare_for_tft)
        merged_data = {
            key: self.convert_all_to_numeric(self.merge_visitor_with_meteo(visitor_df, meteo_dataframe))
            for key, visitor_df in visitor_dataframes.items()
        }
        for key, _ in merged_data.items():
            merged_data[key] = self.resample_to_resolution(merged_data[key],resolution_in_minutes)
        return {key: self.prepare_data_for_tft(df) for key, df in merged_data.items()} if prepare_for_tft else merged_data


if __name__ == "__main__":
    preprocessing = DataPreprocessor()
    visitor_data = pd.read_csv("BesucherMessungExport.csv")
    meteo_csv_directory = "./exogen_data"
    merged_visitor_meteo_dataframes = preprocessing.get_merged_visitor_meteo_data(
        visitor_data, 
        meteo_csv_directory=meteo_csv_directory,
        resolution_in_minutes=60,
        prepare_for_tft=True
    )
    
    print(merged_visitor_meteo_dataframes)
    merged_visitor_meteo_dataframes["Rathausquai"].to_csv("Rathausquai.csv", index=False)
    merged_visitor_meteo_dataframes["Kapellbrücke"].to_csv("Kapellbrücke.csv", index=False)
    merged_visitor_meteo_dataframes["Hertensteinstrasse"].to_csv("Hertensteinstrasse.csv", index=False)
    merged_visitor_meteo_dataframes["Löwendenkmal"].to_csv("Löwendenkmal.csv", index=False)
    merged_visitor_meteo_dataframes["Schwanenplatz"].to_csv("Schwanenplatz.csv", index=False)

    for key, df in merged_visitor_meteo_dataframes.items():
        print(f"\n--- Testing DataFrame for {key} ---")
        print("Datentypen der Spalten:")
        print(df.dtypes)
        print()
        
        print("Überprüfung auf NaN-Werte:")
        nan_counts = df.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]
        if not nan_columns.empty:
            print("Spalten mit NaN-Werten:")
            print(nan_columns)
        else:
            print("Keine NaN-Werte in den Spalten gefunden.")
        print()
        
        print("Skewness der numerischen Spalten:")
        skewness = df.select_dtypes(include=['float64', 'int64']).skew()
        print(skewness)
        print()
        
        print("Anzahl der Ausreisser pro numerischer Spalte:")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        outlier_counts = {}
        for col in numeric_columns:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                outlier_counts[col] = 0
                continue
            z_scores = (df[col] - mean) / std
            outlier_counts[col] = ((z_scores < -3) | (z_scores > 3)).sum()
        print(outlier_counts)
        print()
        
        print("Überprüfung der Standardisierung der numerischen Spalten:")
        means = df[numeric_columns].mean()
        stds = df[numeric_columns].std()
        is_standardized = all(np.isclose(means, 0, atol=1e-1)) and all(np.isclose(stds, 1, atol=1e-1))
        print(f"Ist das DataFrame '{key}' standardisiert? {'Ja' if is_standardized else 'Nein'}")
        print()