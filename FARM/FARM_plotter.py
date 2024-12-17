import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri, numpy2ri

def setup_matplotlib(style='ggplot'):
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

def install_if_missing(package_name, utils):
    try:
        importr(package_name)
        print(f"Package '{package_name}' is already installed.")
    except Exception:
        print(f"Package '{package_name}' not found. Installing...")
        utils.install_packages(StrVector([package_name]))
        importr(package_name)
        print(f"Package '{package_name}' installed successfully.")

def load_r_packages():
    base = importr('base')
    utils = importr('utils')
    robjects.r('.libPaths("/usr/lib64/R/library")')
    lib_paths = robjects.r('.libPaths()')
    print("R library paths:", list(lib_paths))
    return utils

def source_r_scripts(script_paths):
    r_source = robjects.r['source']
    for script in script_paths:
        r_source(script)
        print(f"Sourced R script: {script}")

def read_csv_with_r(csv_path):
    r_df = robjects.r['read.csv'](csv_path, stringsAsFactors=False)
    print("CSV Columns:", list(r_df.names))
    df_pd = pandas2ri.rpy2py_dataframe(r_df)
    df_pd['time'] = pd.to_datetime(df_pd['time'])
    df_pd['Hour'] = df_pd['time'].dt.hour
    return r_df, df_pd

def run_farm_function(farm_func, r_df, weather_column):
    refTS = robjects.FloatVector(r_df.rx2(weather_column))
    qryTS = robjects.FloatVector(r_df.rx2('visitors'))
    result = farm_func(
        refTS, 
        qryTS,
        lcwin=3,
        rel_th=10,
        fuzzyc=1, 
        metric_space=True
    )
    print("Structure:")
    print("List element names:", list(result.names))
    rel_local = np.array(result.rx2('rel.local'))
    rel_global = np.array(result.rx2('rel.global'))[0]
    return rel_local, rel_global

def plot_local_relevance(df, rel_global, weather_column):
    average_relevance = df.groupby('Hour')['rel_local'].mean().reset_index()
    plt.figure()
    plt.plot(
        average_relevance['Hour'], 
        average_relevance['rel_local'], 
        label=f'Local Relevance ({weather_column})', 
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
    plt.title(f'Local Relevance Over the Average Day ({weather_column})', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Local Relevance', fontsize=14)
    plt.xticks(range(0, 24))
    plt.ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = f'local_relevance_{weather_column.replace(".", "_")}.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")
    plt.close()

def plot_weekly_relevance(df, rel_global, weather_column, average_weeks=20):
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
    tick_positions = np.arange(0, 169, 12)
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tick_labels = [
        f"{weekdays[(pos // 24) % 7]} {pos % 24:02d}" for pos in tick_positions
    ]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Local Relevance Over Monday to Sunday for All Weeks ({average_weeks}-Week Averages) ({weather_column})', fontsize=16)
    plt.xlabel('Time (Weekday Hour)', fontsize=14)
    plt.ylabel('Local Relevance', fontsize=14)
    plt.xlim([0, 168])
    plt.ylim(bottom=0)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plot_filename_weekly = f'all_weeks_local_relevance_avg{average_weeks}_{weather_column.replace(".", "_")}.png'
    plt.savefig(plot_filename_weekly, dpi=300)
    print(f"All-weeks plot saved as {plot_filename_weekly}")
    plt.close()
    df.drop(columns=['rel_local', 'day_of_week', 'week_hour', 'year', 'week', 'sequential_week', 'group'], inplace=True)
    return df

def main():
    setup_matplotlib(style='ggplot')
    pandas2ri.activate()
    numpy2ri.activate()
    working_dir = "/home/robin/clone_box/BAA/FARM"
    os.chdir(working_dir)
    utils = load_r_packages()
    required_packages = ['dplyr', 'magrittr']
    for pkg in required_packages:
        install_if_missing(pkg, utils)
    r_scripts = [
        "FARM_Skript/R/farm.dist.R",
        "FARM_Skript/R/farm.R",
        "FARM_Skript/R/tw.decomp.R"
    ]
    source_r_scripts(r_scripts)
    farm_func = robjects.globalenv['farm']
    csv_path = "./data/Rathausquai.csv"
    r_df, df_pd = read_csv_with_r(csv_path)
    weather_columns = [
        'Wind.Speed', 'Sunshine.Duration', 'Air.Pressure', 
        'Absolute.Humidity', 'Precipitation.Duration', 'Air.Temperature'
    ]
    for weather_column in weather_columns:
        print(f"\nProcessing weather column: {weather_column}")
        rel_local, rel_global = run_farm_function(farm_func, r_df, weather_column)
        df_iteration = df_pd.copy()
        df_iteration['rel_local'] = rel_local
        plot_local_relevance(df_iteration, rel_global, weather_column)
        df_iteration = plot_weekly_relevance(df_iteration, rel_global, weather_column, average_weeks=20)
    print("\nProcessing completed.")

if __name__ == "__main__":
    main()
