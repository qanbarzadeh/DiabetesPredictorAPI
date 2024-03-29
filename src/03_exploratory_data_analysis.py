# Import necessary modules
import sys
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Adjust the path to include the Helpers directory
sys.path.append('../Helpers')

# Import custom functions from data_helpers.py
from data_helpers import load_config, load_data

# Load configuration settings from a JSON file
config = load_config('../config/config.json')  # Adjust path as needed

# Ensure the config was loaded successfully
if not config:
    raise Exception("Failed to load configuration.")

# Load the dataset based on the path specified in the configuration
data_path = config['data_path']
diabetes_data = load_data(data_path)

# Ensure the data was loaded successfully
if diabetes_data is None:
    raise Exception("Failed to load the data.")

# Function to plot the distribution for each numerical feature
def plot_feature_distributions(df):
    """Plot histograms and box plots for each numerical feature in the dataframe."""
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig = px.histogram(df, x=column, marginal="box", title=f'Distribution of {column}')
        fig.show()

# Function to visualize the correlation matrix of features
def plot_correlation_matrix(df):
    """Generate a heatmap representing the correlation matrix of the dataframe's features."""
    corr_matrix = df.corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.to_numpy(),
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).to_numpy(),
        colorscale='Viridis',
        showscale=True
        )
    fig.update_layout(title_text='Feature Correlation Matrix', title_x=0.5)
    fig.show()

# Function to compare the distribution of variables across different categories
def plot_variable_relationships(df, target):
    """Plot box plots for each numerical feature across different categories of the target variable."""
    features = df.select_dtypes(include=['float64', 'int64']).columns.drop(target)
    for feature in features:
        fig = px.box(df, x=target, y=feature, color=target, title=f'{feature} Distribution by {target}')
        fig.show()

# Function to analyze the distribution of the target variable
def plot_target_distribution(df, target):
    """Plot a histogram to show the distribution of the target variable."""
    fig = px.histogram(df, x=target, title=f'Distribution of {target}')
    fig.show()

# Conducting EDA
print("Conducting Exploratory Data Analysis...")
plot_feature_distributions(diabetes_data)
plot_correlation_matrix(diabetes_data)
plot_variable_relationships(diabetes_data, 'Outcome')
plot_target_distribution(diabetes_data, 'Outcome')
