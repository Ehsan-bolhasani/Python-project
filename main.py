import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from scipy.optimize import curve_fit
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot


class DatabaseManager:
    """
    A class for managing the SQLite database.

    Parameters:
        db_name (str): The name of the SQLite database.

    Attributes:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine for database connection.
    """
    def __init__(self, db_name):
        self.engine = create_engine(f"sqlite:///{db_name}.db")

    def create_tables(self, table_name, columns):
        """
        Create tables in the database.

        Parameters:
            table_name (str): The name of the table.
            columns (list): A list of column names for the table.
        """
        data_df = pd.DataFrame(columns=columns)
        data_df.to_sql(table_name, con=self.engine, index_label='index', if_exists='replace')

    def load_data(self, table_name, file_name):
        """
        Load data from a CSV file into the specified table in the database.

        Parameters:
            table_name (str): The name of the table.
            file_name (str): The name of the CSV file.
        """
        data_df = pd.read_csv(file_name)
        data_df.to_sql(table_name, con=self.engine, index_label='index', if_exists='replace')


class IdealFunctionCalculator:
    """
    A class for calculating ideal functions.

    Parameters:
        db_name (str): The name of the SQLite database.

    Attributes:
        db_name (str): The name of the SQLite database.
    """
    def __init__(self, db_name):
        self.db_name = db_name

    def calculate_ideal_functions(self):
        """
        Calculate ideal functions based on training data.
        """
        with sqlite3.connect(self.db_name + '.db') as connection:
            query = 'SELECT * FROM training_data'
            training_data_df = pd.read_sql_query(query, con=connection, index_col='index')

        ideal_functions_df = training_data_df.groupby('x').mean().reset_index()

        with sqlite3.connect(self.db_name + '.db') as connection:
            ideal_functions_df.to_sql('ideal_functions', con=connection, index=False, if_exists='replace')

    def map_to_ideal_functions(self, x, y):
        """
        Map test data to the nearest ideal function.

        Parameters:
            x (float): The X value of the test data.
            y (float): The Y value of the test data.

        Returns:
            tuple: A tuple containing the assigned function and deviation.
        """
        with sqlite3.connect(self.db_name + '.db') as connection:
            query = 'SELECT * FROM ideal_functions'
            ideal_functions_df = pd.read_sql_query(query, con=connection)

        ideal_function_columns = ideal_functions_df.columns[1:]
        best_fit_function = None
        best_deviation = float('inf')

        for column in ideal_function_columns:
            try:
                x_data = ideal_functions_df['x']
                y_data = ideal_functions_df[column]

                if x_data.empty:
                    continue

                initial_guess = [1.0] * len(ideal_function_columns)
                popt, _ = curve_fit(self.ideal_function, x_data, y_data, p0=initial_guess)

                predicted_values = self.ideal_function(x, *popt)
                deviation = abs(y - predicted_values)

                if deviation < best_deviation:
                    best_fit_function = column
                    best_deviation = deviation

            except Exception as e:
                print(f"Error fitting function {column}: {e}")

        return best_fit_function, best_deviation

    @staticmethod
    def ideal_function(x, *params):
        """
        The ideal function for curve fitting.

        Parameters:
            x (float): The X value.
            *params (float): Coefficients of the function.

        Returns:
            float: The calculated Y value.
        """
        return sum(p * x ** i for i, p in enumerate(params))


class TestDataMapper:
    """
    A class for mapping test data to ideal functions.

    Parameters:
        db_name (str): The name of the SQLite database.
        ideal_calculator (IdealFunctionCalculator): An instance of the IdealFunctionCalculator class.
    """
    def __init__(self, db_name, ideal_calculator):
        self.db_name = db_name
        self.ideal_calculator = ideal_calculator

    def map_test_data(self):
        """
        Map test data to ideal functions and save the results.
        """
        with sqlite3.connect(self.db_name + '.db') as connection:
            query = 'SELECT * FROM test_data'
            test_data_df = pd.read_sql_query(query, con=connection, index_col='index')

        results = []
        for index, row in test_data_df.iterrows():
            x, y = row['x'], row['y']
            assigned_function, deviation = self.ideal_calculator.map_to_ideal_functions(x, y)
            results.append({'X': x, 'Y': y, 'Assigned_Function': assigned_function, 'Deviation': deviation})

        result_df = pd.DataFrame(results)
        with sqlite3.connect(self.db_name + '.db') as connection:
            result_df.to_sql('test_results', con=connection, index=False, if_exists='replace')


class DataVisualizer:
    """
    A class for visualizing data.

    Parameters:
        db_name (str): The name of the SQLite database.
    """
    def __init__(self, db_name):
        self.db_name = db_name

    def visualize_data(self):
        """
        Visualize training data, test results, and ideal functions using Bokeh.
        """
        with sqlite3.connect(self.db_name + '.db') as connection:
            training_data_query = 'SELECT * FROM training_data'
            training_data_df = pd.read_sql_query(training_data_query, con=connection, index_col='index')

            test_results_query = 'SELECT * FROM test_results'
            test_results_df = pd.read_sql_query(test_results_query, con=connection)

            ideal_functions_query = 'SELECT * FROM ideal_functions'
            ideal_functions_df = pd.read_sql_query(ideal_functions_query, con=connection)

        print("Training Data Columns:", training_data_df.columns)
        print("Test Results Columns:", test_results_df.columns)
        print("Ideal Functions Columns:", ideal_functions_df.columns)

        training_data_plot = self.create_scatter_plot(training_data_df, 'x', 'y1', 'Training Data')
        test_results_plot = self.create_scatter_plot(test_results_df, 'X', 'Y', 'Test Results', color='red')
        ideal_functions_plot = self.create_line_plot(ideal_functions_df, 'x', ideal_functions_df.columns[1:],
                                                     'Ideal Functions')

        grid = gridplot([[training_data_plot, test_results_plot], [ideal_functions_plot]])

        show(grid)

    def create_scatter_plot(self, data, x_col, y_col, title, color='blue'):
        """
        Create a scatter plot using Bokeh.

        Parameters:
            data (pd.DataFrame): The data for the plot.
            x_col (str): The name of the X column.
            y_col (str): The name of the Y column.
            title (str): The title of the plot.
            color (str): The color of the plot points.

        Returns:
            bokeh.plotting.Figure: The Bokeh scatter plot.
        """
        source = ColumnDataSource(data)
        plot = figure(title=title, x_axis_label=x_col, y_axis_label=y_col, width=400, height=400)
        plot.scatter(x=x_col, y=y_col, source=source, color=color)
        return plot

    def create_line_plot(self, data, x_col, y_cols, title):
        """
        Create a line plot using Bokeh.

        Parameters:
            data (pd.DataFrame): The data for the plot.
            x_col (str): The name of the X column.
            y_cols (list): The names of the Y columns.
            title (str): The title of the plot.

        Returns:
            bokeh.plotting.Figure: The Bokeh line plot.
        """
        source = ColumnDataSource(data)
        plot = figure(title=title, x_axis_label=x_col, y_axis_label='Y', width=400, height=400)
        for col in y_cols:
            plot.line(x=x_col, y=col, source=source, legend_label=col)
        plot.legend.click_policy = 'hide'
        return plot


if __name__ == "__main__":
    db_manager = DatabaseManager('My_database')

    columns_training_data = ['x', 'y1', 'y2', 'y3', 'y4']
    columns_ideal_functions = ['x'] + [f'y{i}' for i in range(1, 51)]
    columns_test_data = ['x', 'y']

    # Create tables
    db_manager.create_tables('training_data', columns_training_data)
    db_manager.create_tables('ideal_functions', columns_ideal_functions)
    db_manager.create_tables('test_data', columns_test_data)

    # Load training data
    db_manager.load_data('training_data', 'train.csv')

    # Load test data
    db_manager.load_data('test_data', 'test.csv')

    # Load ideal functions
    db_manager.load_data('ideal_functions', 'ideal.csv')

    # Calculate ideal functions
    ideal_calculator = IdealFunctionCalculator('My_database')
    ideal_calculator.calculate_ideal_functions()

    # Map test data
    test_data_mapper = TestDataMapper('My_database', ideal_calculator)
    test_data_mapper.map_test_data()

    # Visualize the data
    data_visualizer = DataVisualizer('My_database')
    data_visualizer.visualize_data()
