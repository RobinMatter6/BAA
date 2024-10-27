import pandas as pd
from typing import List, Dict
import os
import sys
import matplotlib.pyplot as plt
import re

sys.path.append('../Data_preprocessing')
from DataPreprocessor import DataPreprocessor

class DataAnalyser:
    def __init__(self):
        pass

    def load_data(self, visitor_path: str, exogen_data_directory_path: str) -> Dict[str, pd.DataFrame]:
        preprocessing = DataPreprocessor()
        visitor_data = pd.read_csv(visitor_path)
        meteo_csv_directory = exogen_data_directory_path
        return preprocessing.get_merged_visitor_meteo_data(visitor_dataframe=visitor_data, meteo_csv_directory=meteo_csv_directory)
    
    def strip_column_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        return df

    def convert_object_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def split_by_weekday(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        weekday_map = {0: 'MO', 1: 'TU', 2: 'WE', 3: 'TH', 4: 'FR', 5: 'SA', 6: 'SU'}
        return {day: df[df['time'].dt.weekday == day_num].reset_index(drop=True)
                for day_num, day in weekday_map.items()}

    def process_data(self, visitor_path: str, exogen_data_directory_path: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        df_dict = self.load_data(visitor_path=visitor_path, exogen_data_directory_path=exogen_data_directory_path)
        return {location: self.split_by_weekday(df=self.convert_object_to_numeric(df=self.strip_column_spaces(df=df)))
                for location, df in df_dict.items()}

    def average_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = df['time'].dt.hour
        hourly_mean_df = df.groupby('hour').mean(numeric_only=True).reset_index()
        return hourly_mean_df

    def create_hourly_avg_for_all(self, location_weekdays_dfs: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        return {location: {day: self.average_hourly_data(df=df) for day, df in weekdays_df.items()}
                for location, weekdays_df in location_weekdays_dfs.items()}

    def get_exog_categories(self, exog_variable: str) -> List[tuple]:
        exog_categories = {
            'Air Temperature (°C)': [(-float('inf'), 0, '< 0°C'), (0, 10, '0-10°C'), (10, 20, '10-20°C'), (20, 30, '20-30°C'), (30, float('inf'), '> 30°C')],
            'Sunshine Duration (min)': [(0, 0, 'No Sunshine'), (1, 2, '1-2 min'), (2, 5, '2-5 min'), (5, 10, '5-10 min')],
            'Air Pressure (hPa)': [(-float('inf'), 980, '< 980 hPa'), (980, 1000, '980-1000 hPa'), (1000, 1020, '1000-1020 hPa'), (1020, 1040, '1020-1040 hPa'), (1040, float('inf'), '> 1040 hPa')],
            'Precipitation Duration (min)': [(0, 0, 'No Precipitation'), (0.1, 2, '1-2 min'), (2.1, 5, '2-5 min'), (5.1, 10, '5-10 min')],
            'Absolute Humidity (g/m³)': [(0, 5, '0-5 g/m³'), (5, 10, '5-10 g/m³'), (10, 15, '10-15 g/m³'), (15, float('inf'), '> 15 g/m³')],
            'Precipitation (mm)': [(0, 0, 'No Rain'), (0.1, 2, '0.1-2 mm'), (2, 5, '2-5 mm'), (5, 10, '5-10 mm')],
            'Snow Depth (cm)': [(0, 0, 'No Snow'), (1, 2, '1-2 cm'), (2, 5, '2-5 cm'), (5, 10, '5-10 cm')],
            'Wind Speed (km/h)': [(0, 10, '0-10 km/h'), (10, 20, '10-20 km/h'), (20, 30, '20-30 km/h'), (30, float('inf'), '> 30 km/h')]
        }
        return exog_categories.get(exog_variable, [])

    def calculate_deltas(self, overall_avg: pd.DataFrame, df: pd.DataFrame, exog_variable: str, categories: List[tuple]) -> pd.DataFrame:
        delta_frames = []
        for value_range in categories:
            df_filtered = df[(df[exog_variable] >= value_range[0]) & (df[exog_variable] < value_range[1])].copy()
            df_filtered['hour'] = df_filtered['time'].dt.hour
            hourly_avg_filtered = df_filtered.groupby('hour').mean(numeric_only=True).reset_index()
            delta = pd.merge(overall_avg[['hour', 'visitors']], hourly_avg_filtered[['hour', 'visitors']],
                             on='hour', how='left', suffixes=('_overall', '_filtered'))
            delta['visitors_filtered'] = delta['visitors_filtered'].fillna(overall_avg['visitors'])
            delta['visitors_delta'] = delta['visitors_filtered'] - delta['visitors_overall']
            delta['category'] = value_range[2]
            delta_frames.append(delta)
        return pd.concat(delta_frames)

    def process_and_plot_all_exog_variables(self, location_weekdays_dfs: Dict[str, Dict[str, pd.DataFrame]], directory: str = './plots') -> None:
        weekday_number_map = {'MO': '01', 'TU': '02', 'WE': '03', 'TH': '04', 'FR': '05', 'SA': '06', 'SU': '07'}
        for location, weekdays_dfs in location_weekdays_dfs.items():
            for day, df in weekdays_dfs.items():
                day_number = weekday_number_map.get(day[:2], '')
                exogenous_columns = [col for col in df.columns if col not in ['visitors', 'time']]
                overall_avg = self.average_hourly_data(df=df)
                for exog_variable in exogenous_columns:
                    if not df[exog_variable].isna().all():
                        print(f"Plotting {exog_variable} for {location} on {day}")
                        self.plot_exog_vs_visitors_percentage_change(df=df, exog_variable=exog_variable, location=location, day=day, day_number=day_number, directory=directory)
                    categories = self.get_exog_categories(exog_variable=exog_variable)
                    if categories:
                        delta_df = self.calculate_deltas(overall_avg=overall_avg, df=df, exog_variable=exog_variable, categories=categories)
                        self.plot_exog_vs_visitors_percentage_change_on_hour(df=delta_df, x_column='hour', y_column='visitors_delta', category_column='category', exog_variable=exog_variable, location=location, day=day, day_number=day_number, directory=directory)
                    self.plot_correlation_of_exog_variable(df=df, exog_variable=exog_variable, location=location, day=day, day_number=day_number, directory=directory)

    def clean_variable_name(self, var_name: str) -> str:
        return re.sub(r'[^\w\s-]', '_', var_name)

    def calculate_percentage_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        avg_visitors = df['visitors'].mean()
        df['visitor_percentage_deviation'] = ((df['visitors'] - avg_visitors) / avg_visitors) * 100
        return df

    def get_and_create_exog_directory(self, plot_category: str, base_directory: str, location: str, exog_variable: str) -> str:
        exog_directory = os.path.join(base_directory, plot_category, location, exog_variable)
        os.makedirs(exog_directory, exist_ok=True)
        return exog_directory

    def calculate_correlation(self, df: pd.DataFrame, exog_variable: str) -> pd.DataFrame:
        correlation_data = []
        for hour in range(24):
            hourly_data = df[df['hour'] == hour]
            hourly_data = hourly_data.dropna(subset=[exog_variable, 'visitors'])
            if len(hourly_data) > 1:
                if hourly_data[exog_variable].std() == 0 or hourly_data['visitors'].std() == 0:
                    correlation = None
                else:
                    correlation = hourly_data[exog_variable].corr(hourly_data['visitors'])
                correlation_data.append((hour, correlation * 100 if correlation is not None else None))
            else:
                correlation_data.append((hour, None))
        correlation_df = pd.DataFrame(correlation_data, columns=['Hour of Day', 'Correlation Factor (%)'])
        correlation_df.dropna(inplace=True)
        return correlation_df

    def plot_correlation_of_exog_variable(self, df: pd.DataFrame, exog_variable: str, location: str, day: str, day_number: str, directory: str = './plots') -> None:
        clean_exog_variable = self.clean_variable_name(var_name=exog_variable)
        exog_directory = self.get_and_create_exog_directory(plot_category="correlation_of_exog_variable_and_visitors", base_directory=directory, location=location, exog_variable=exog_variable)
        correlation_df = self.calculate_correlation(df=df, exog_variable=exog_variable)

        plt.figure(figsize=(10, 6))
        plt.plot(correlation_df['Hour of Day'], correlation_df['Correlation Factor (%)'], marker='o')
        plt.title(f'Correlation between Visitors and {exog_variable} for {location} on {day}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Correlation Factor (%)')
        plt.grid(True)
        plt.savefig(os.path.join(exog_directory, f'correlation_visitors_vs_{clean_exog_variable}_{location}_{day_number}.png'))
        plt.close()

    def plot_exog_vs_visitors_percentage_change_on_hour(self, df: pd.DataFrame, x_column: str, y_column: str, category_column: str, exog_variable: str, location: str, day: str, day_number: str, directory: str = './plots') -> None:
        clean_exog_variable = self.clean_variable_name(var_name=exog_variable)
        exog_directory = self.get_and_create_exog_directory(plot_category="exog_vs_visitors_percentage_change_on_hour", base_directory=directory, location=location, exog_variable=exog_variable)
        plt.figure(figsize=(10, 6))
        for category in df[category_column].unique():
            category_df = df[df[category_column] == category]
            plt.plot(category_df[x_column], category_df[y_column], marker='o', label=category)
        plt.title(f'{y_column} vs {x_column} ({exog_variable}) for {location} on {day}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.legend()
        plt.grid(True)
        filename = os.path.join(exog_directory, f'percentage_deviation_visitors_vs_{clean_exog_variable}_on_hour_{location}_{day_number}.png')
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as {filename}")

    def plot_exog_vs_visitors_percentage_change(self, df: pd.DataFrame, exog_variable: str, location: str, day: str, day_number: str, directory: str = './plots') -> None:
        clean_exog_variable = self.clean_variable_name(var_name=exog_variable)
        exog_directory = self.get_and_create_exog_directory(plot_category="exog_vs_visitors_percentage_change", base_directory=directory, location=location, exog_variable=exog_variable)
        df = self.calculate_percentage_deviation(df=df)
        df_grouped = df.groupby(exog_variable)['visitor_percentage_deviation'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.scatter(df_grouped[exog_variable], df_grouped['visitor_percentage_deviation'], marker='o', color='b')
        plt.title(f'Percentage Deviation in Visitors vs {exog_variable} for {location} on {day}')
        plt.xlabel(exog_variable)
        plt.ylabel('Percentage Deviation (%)')
        plt.grid()
        plt.savefig(os.path.join(exog_directory, f'percentage_deviation_visitors_vs_{clean_exog_variable}_{location}_{day_number}.png'))
        plt.close()

if __name__ == "__main__":
    analyser = DataAnalyser()
    location_weekdays_dfs = analyser.process_data(visitor_path="../Data_preprocessing/BesucherMessungExport.csv", exogen_data_directory_path="../Data_preprocessing/exogen_data")

    analyser.process_and_plot_all_exog_variables(location_weekdays_dfs=location_weekdays_dfs, directory='./delta_plots_exogen_data')
