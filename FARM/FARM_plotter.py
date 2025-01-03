import os
import json
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rpy2 import robjects
from rpy2.robjects.packages import importr, PackageNotInstalledError
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri, numpy2ri

pandas2ri.activate()
numpy2ri.activate()


def setup_matplotlib(style: str = 'ggplot') -> None:
    plt.style.use(style)
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (14, 8),
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })


def install_r_package(package_name: str, utils_pkg) -> None:
    try:
        importr(package_name)
        print(f"R package '{package_name}' is already installed.")
    except PackageNotInstalledError:
        print(f"R package '{package_name}' not found. Installing...")
        utils_pkg.install_packages(StrVector([package_name]))
        importr(package_name)
        print(f"R package '{package_name}' installed successfully.")


def load_r_packages(lib_path: str = "/usr/lib64/R/library") -> robjects.packages.Package:
    utils = importr('utils')
    robjects.r(f'.libPaths("{lib_path}")')
    lib_paths = list(robjects.r('.libPaths()'))
    print("R library paths:", lib_paths)
    return utils


def source_r_scripts(script_paths: List[str]) -> None:
    r_source = robjects.r['source']
    for script in script_paths:
        r_source(script)
        print(f"Sourced R script: {script}")


def read_csv_with_r(csv_path: str) -> Tuple[robjects.vectors.DataFrame, pd.DataFrame]:
    r_df = robjects.r['read.csv'](csv_path, stringsAsFactors=False)
    print("CSV Columns:", list(r_df.names))
    df_pd = pandas2ri.rpy2py_dataframe(r_df)
    df_pd['time'] = pd.to_datetime(df_pd['time'])
    df_pd['Hour'] = df_pd['time'].dt.hour
    return r_df, df_pd


def run_farm_function(farm_func, r_df: robjects.vectors.DataFrame, weather_column: str) -> Tuple[np.ndarray, float]:
    ref_ts = robjects.FloatVector(r_df.rx2(weather_column))
    qry_ts = robjects.FloatVector(r_df.rx2('visitors'))
    result = farm_func(
        ref_ts,
        qry_ts,
    )
    print("FARM Function Output Structure:", list(result.names))
    rel_local = np.array(result.rx2('rel.local'))
    rel_global = float(result.rx2('rel.global')[0])
    return rel_local, rel_global


def plot_local_relevance(df: pd.DataFrame, rel_global: float, weather_display: str) -> None:
    average_relevance = df.groupby('Hour')['rel_local'].mean().reset_index()

    plt.figure()
    plt.plot(
        average_relevance['Hour'],
        average_relevance['rel_local'],
        label=f'Local Relevance ({weather_display})',
        color='#001cbd',
        marker='o'
    )
    plt.axhline(
        y=rel_global,
        color='#bd1900',
        linestyle='--',
        linewidth=2,
        label='Global Relevance'
    )
    plt.title(f'Local Relevance Over the Average Day ({weather_display})', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Local Relevance', fontsize=14)
    plt.xticks(range(0, 24))
    plt.ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plot_filename = f'local_relevance_{weather_display.replace(" ", "_")}.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")
    plt.close()


def plot_weekly_relevance(df: pd.DataFrame, rel_global: float, weather_display: str, average_weeks: int = 20) -> pd.DataFrame:
    df = df.copy()
    df['day_of_week'] = df['time'].dt.dayofweek
    df['week_hour'] = df['day_of_week'] * 24 + df['Hour']

    iso_calendar = df['time'].dt.isocalendar()
    df['year'] = iso_calendar.year
    df['week'] = iso_calendar.week

    unique_weeks = df[['year', 'week']].drop_duplicates().reset_index(drop=True)
    unique_weeks['sequential_week'] = unique_weeks.index + 1

    df = df.merge(unique_weeks, on=['year', 'week'], how='left')

    df['group'] = ((df['sequential_week'] - 1) // average_weeks) + 1

    plt.figure()

    grouped = df.groupby(['group', 'week_hour'])
    avg_per_group = grouped['rel_local'].mean().reset_index()

    for group_num in avg_per_group['group'].unique():
        group_data = avg_per_group[avg_per_group['group'] == group_num]
        plt.plot(
            group_data['week_hour'],
            group_data['rel_local'],
            color='#001cbd',
            alpha=0.3,
            label=f'{average_weeks}-Week Avg Group {group_num}' if group_num == 1 else ""
        )

    avg_week = df.groupby('week_hour')['rel_local'].mean().reset_index()
    plt.plot(
        avg_week['week_hour'],
        avg_week['rel_local'],
        color='#bd1900',
        linewidth=3,
        label='Average Week'
    )

    plt.axhline(y=rel_global, color='#7f0000', linestyle='--', linewidth=2, label='Global Relevance')

    tick_positions = np.arange(0, 168, 12)
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tick_labels = [
        f"{weekdays[(pos // 24) % 7]} {pos % 24:02d}" for pos in tick_positions
    ]

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Local Relevance Over Monday to Sunday for All Weeks ({average_weeks}-Week Averages) ({weather_display})', fontsize=16)
    plt.xlabel('Time (Weekday Hour)', fontsize=14)
    plt.ylabel('Local Relevance', fontsize=14)
    plt.xlim([0, 168])
    plt.ylim(bottom=0)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    plot_filename_weekly = f'all_weeks_local_relevance_avg{average_weeks}_{weather_display.replace(" ", "_")}.png'
    plt.savefig(plot_filename_weekly, dpi=300)
    print(f"All-weeks plot saved as {plot_filename_weekly}")
    plt.close()
    temp_columns = ['rel_local', 'day_of_week', 'week_hour', 'year', 'week', 'sequential_week', 'group']
    df.drop(columns=temp_columns, inplace=True)
    return df


def save_importance_to_json(local_importance: Dict[str, List[float]], output_file: str = "FARM_importance.json") -> None:
    try:
        with open(output_file, 'w') as f:
            json.dump(local_importance, f, indent=4)
        print(f"\nLocal importance values saved to '{output_file}'.")
    except IOError as e:
        print(f"Failed to save local importance to JSON. Error: {e}")


def process_weather_column(farm_func, r_df: robjects.vectors.DataFrame, df_pd: pd.DataFrame,
                          weather_column: str, display_name: str) -> List[float]:
    print(f"\nProcessing weather column: {display_name}")

    rel_local, rel_global = run_farm_function(farm_func, r_df, weather_column)

    local_importance = rel_local.tolist()

    df_plot = df_pd.copy()
    df_plot['rel_local'] = rel_local

    plot_local_relevance(df_plot, rel_global, display_name)
    plot_weekly_relevance(df_plot, rel_global, display_name, average_weeks=20)
    return local_importance


def main():
    setup_matplotlib(style='ggplot')

    working_dir = "/opt/BAA/FARM"
    try:
        os.chdir(working_dir)
        print(f"Changed working directory to {working_dir}")
    except FileNotFoundError:
        print(f"Working directory '{working_dir}' not found.")
        return

    utils_pkg = load_r_packages()
    required_packages = ['dplyr', 'magrittr']
    for pkg in required_packages:
        install_r_package(pkg, utils_pkg)

    r_scripts = [
        "FARM_Skript/R/farm.dist.R",
        "FARM_Skript/R/farm.R",
        "FARM_Skript/R/tw.decomp.R"
    ]
    source_r_scripts(r_scripts)
    farm_func = robjects.globalenv['farm']

    csv_path = "./data/Rathausquai.csv"
    if not os.path.isfile(csv_path):
        print(f"CSV file '{csv_path}' does not exist.")
        return
    
    r_df, df_pd = read_csv_with_r(csv_path)
    columns_to_drop = {'time', 'visitors', 'dummy_id', 'region', 'date'}
    columns_to_analyse = [col for col in r_df.names if col not in columns_to_drop]
    display_column_names = { 
        'Wind.Speed': 'Wind Speed',
        'Sunshine.Duration': 'Sunshine Duration',
        'Air.Pressure': 'Air Pressure',
        'Absolute.Humidity': 'Absolute Humidity',
        'Precipitation.Duration': 'Precipitation Duration',
        'Air.Temperature': 'Air Temperature',
        'minute_of_hour': 'minute_of_hour',
        'hour_of_day': 'hour_of_day',
        'day_of_week': 'day_of_week',
        'week_of_year': 'week_of_year',
        'month_of_year': 'month_of_year',
        'day_of_month': 'day_of_month'
    }
    local_importance_dict = {}
    print(list(r_df.names))
    for original, display in display_column_names.items():
        if original not in columns_to_analyse:
            print(f"Warning: '{original}' is not in the list of original weather columns.")
            continue
        importance = process_weather_column(farm_func, r_df, df_pd, original, display)
        local_importance_dict[display] = importance
    save_importance_to_json(local_importance_dict, output_file="FARM_importance.json")
    print("\nProcessing completed.")

if __name__ == "__main__":
    main()