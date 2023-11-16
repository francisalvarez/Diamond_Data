import pandas as pd
import plotly.express as px
from plotly.offline import plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import tools.eda as ed
from tools.modeling import model_data_stats_model

def load_data():
    fp = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
    return pd.read_csv(fp)

diamonds = load_data()
print(diamonds.head())
print(diamonds.describe())

## Plotting
# fig = px.scatter(
#     data_frame=diamonds,
#     x="carat",
#     y="price"
# )
# plot(fig)

## Modeling
model_data_stats_model(
    df=diamonds,
    numeric_features=['carat', 'depth', 'table', 'x', 'y', 'z'],
    categorical_features=['cut', 'color', 'clarity'],
    y_field="price"
)

print("hi")