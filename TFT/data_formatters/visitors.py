import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class VisitorsFormatter(GenericDataFormatter):
  """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """


  # _column_definition = [
  #     ('date', DataTypes.DATE, InputTypes.TIME),
  #     ('minute_of_hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),             # Minute within the hour
  #     ('hour_of_day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Hour within the day
  #     ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Day of the week
  #     ('week_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Week of the year
  #     ('month_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),              # Month of the year
  #     ('day_of_month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Day of the month
  #     ('visitors', DataTypes.REAL_VALUED, InputTypes.TARGET),                        # Target variable                    # Static Categorical Input
  #     ('dummy_id', DataTypes.REAL_VALUED, InputTypes.ID),                            # Dummy ID
  #     ('region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT) 
  # ]

  

  _column_definition = [
      ('date', DataTypes.DATE, InputTypes.TIME),
      ('minute_of_hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),             # Minute within the hour
      ('hour_of_day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Hour within the day
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Day of the week
      ('week_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Week of the year
      ('month_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),              # Month of the year
      ('day_of_month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Day of the month
      ('Absolute Humidity', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),    # Absolute Humidity
      ('visitors', DataTypes.REAL_VALUED, InputTypes.TARGET),                        # Target variable                    # Static Categorical Input
      ('dummy_id', DataTypes.REAL_VALUED, InputTypes.ID),                            # Dummy ID
      ('region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT) 
  ]

  # _column_definition = [
  #     ('date', DataTypes.DATE, InputTypes.TIME),
  #     ('minute_of_hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),             # Minute within the hour
  #     ('hour_of_day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Hour within the day
  #     ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Day of the week
  #     ('week_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Week of the year
  #     ('month_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),              # Month of the year
  #     ('day_of_month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Day of the month
  #     ('Wind Speed', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),           # Wind Speed
  #     ('Sunshine Duration', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),    # Sunshine Duration
  #     ('Air Pressure', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),          # Air Pressure
  #     ('Absolute Humidity', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),    # Absolute Humidity
  #     ('Precipitation Duration', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT), # Precipitation Duration
  #     ('Air Temperature', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),        # Air Temperature
  #     ('visitors', DataTypes.REAL_VALUED, InputTypes.TARGET),                        # Target variable                    # Static Categorical Input
  #     ('dummy_id', DataTypes.REAL_VALUED, InputTypes.ID),                            # Dummy ID
  #     ('region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT) 
  # ]

  # _column_definition = [
  #     ('date', DataTypes.DATE, InputTypes.TIME),
  #     ('minute_of_hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),             # Minute within the hour
  #     ('hour_of_day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Hour within the day
  #     ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),                # Day of the week
  #     ('week_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Week of the year
  #     ('month_of_year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),              # Month of the year
  #     ('day_of_month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),               # Day of the month
  #     ('Wind Speed (km/h)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),           # Wind Speed
  #     ('Sunshine Duration (min)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),    # Sunshine Duration
  #     ('Air Pressure (hPa)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),          # Air Pressure
  #     ('Absolute Humidity (g/m³)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),    # Absolute Humidity
  #     ('Precipitation Duration (min)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT), # Precipitation Duration
  #     ('Air Temperature (°C)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),        # Air Temperature
  #     ('visitors', DataTypes.REAL_VALUED, InputTypes.TARGET),                        # Target variable
  #     ('dummy_id', DataTypes.REAL_VALUED, InputTypes.ID),                            # Dummy ID
  #     ('region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)                     # Static Categorical Input
  # ]


  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df):
      """Splits data frame into training-validation-test data frames using a 70:20:10 split.

      This also calibrates scaling object, and transforms data for each split.

      Args:
        df: Source data frame to split.

      Returns:
        Tuple of transformed (train, valid, test) data.
      """
      
      print("Formatting train-valid-test splits.")
      
      # Calculate split indices
      total_rows = len(df)
      train_end = int(total_rows * 0.7)  # End index for training set
      valid_end = train_end + int(total_rows * 0.2)  # End index for validation set

      # Split data
      train = df.iloc[:train_end]
      valid = df.iloc[train_end:valid_end]
      test = df.iloc[valid_end:]
     
      # Set scalers for the training set
      self.set_scalers(train)

      # Return transformed datasets
      return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')
    
    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
        df[[target_column]].values)  # used for predictions

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  # def format_predictions(self, predictions):
  #   """Reverts any normalisation to give predictions in original scale.

  #   Args:
  #     predictions: Dataframe of model predictions.

  #   Returns:
  #     Data frame of unnormalised predictions.
  #   """
  #   output = predictions.copy()

  #   column_names = predictions.columns

  #   for col in column_names:
  #     if col not in {'forecast_time', 'identifier'}:
  #           reshaped_data = predictions[col].values.reshape(-1, 1)
  #           # Apply inverse transform
  #           inversed_data = self._target_scaler.inverse_transform(reshaped_data)
  #           # Flatten back to 1D and assign
  #           output[col] = inversed_data.flatten()
  #     return output


  def format_predictions(self, predictions):
      """Reverts any normalization to give predictions in original scale and applies PowerTransformer.

      Args:
          predictions: DataFrame of model predictions.

      Returns:
          DataFrame of unnormalized and power-transformed predictions.
      """
      # Kopiere die Vorhersagen, um die Originaldaten nicht zu verändern
      output = predictions.copy()

      # Erhalte die Spaltennamen
      column_names = predictions.columns

      # Rücktransformation der Normalisierung
      for col in column_names:
          if col not in {'forecast_time', 'identifier'}:
              reshaped_data = predictions[col].values.reshape(-1, 1)
              # Inverse Transform anwenden
              inversed_data = self._target_scaler.inverse_transform(reshaped_data)
              # Zurück in 1D umformen und zuweisen
              output[col] = inversed_data.flatten()

      # Identifiziere numerische Spalten für die PowerTransformation
      numeric_cols = output.select_dtypes(include=['float64', 'int64']).columns

      # Initialisiere den PowerTransformer
      pt = PowerTransformer(method='yeo-johnson', standardize=False)

      # Wende den PowerTransformer auf die numerischen Spalten an
      output[numeric_cols] = pt.fit_transform(output[numeric_cols])

      return output
    
  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 8 * 24,
        'num_encoder_steps': 24,
        'num_epochs': 100,
        'early_stopping_patience': 3,
        'multiprocessing_workers': 5,
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.3, # tiefer versuchen
        'hidden_layer_size': 160, # tiefer versuchen
        'learning_rate': 0.001, #  von 0.0001 hoch
        'minibatch_size': 64, # 128 oder 64
        'max_gradient_norm': 0.01, # 100 oder 0.01
        'num_heads': 1,
        'stack_size': 1
    }


    return model_params
#learning_rate, minibatch_size, hidden_layer_size, dropout_rate, max_gradient_norm, num_heads,stack_size


