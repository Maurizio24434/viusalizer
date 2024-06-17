import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import calendar
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Import data and set index to 'date'
df = pd.read_csv('fcc-forum-pageviews.csv', parse_dates=['date'], index_col='date')

# Clean data: filter out days when the page views were in the top 2.5% or bottom 2.5%
df_clean = df[
    (df['value'] >= df['value'].quantile(0.025)) &
    (df['value'] <= df['value'].quantile(0.975))
    ]


def clean_data():
    df_clean = df.copy()  # Make a copy to avoid modifying the original DataFrame

    # Perform data cleaning operations
    df_clean.dropna(inplace=True)  # Remove rows with missing values

    # Filter data to include only rows within a specific date range
    start_date = '2016-05-01'
    end_date = '2019-12-31'
    df_clean = df_clean[(df_clean.index >= start_date) & (df_clean.index <= end_date)]

    return df_clean

def draw_line_plot():
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the data with a rolling mean for a sinusoidal effect
    df_smoothed = df_clean.rolling(window=30, center=True).mean()
    ax.plot(df_smoothed.index, df_smoothed['value'], color='red', linewidth=1)

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Page Views')
    ax.set_title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')

    # Format x-axis ticks
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Save image and return fig
    plt.savefig('line_plot.png')
    return fig


def draw_bar_plot():
    # Prepare data for monthly bar plot
    df_bar = df_clean.groupby([df_clean.index.year, df_clean.index.month]).mean()
    df_bar = df_bar.unstack()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a custom color palette for months
    custom_palette = sns.color_palette('tab20', n_colors=12)  # Example using tab20 palette

    df_bar.plot(kind='bar', ax=ax, legend=True, color=custom_palette)

    # Set labels and title
    ax.set_xlabel('Years')
    ax.set_ylabel('Average Page Views')
    ax.set_title('Average Page Views per Month (2016-2019)')

    # Set legend
    ax.legend(title='Months', labels=[calendar.month_name[i] for i in range(1, 13)])

    # Save image and return fig
    plt.savefig('bar_plot.png')
    return fig


def draw_box_plot():
    # Prepare data for box plots
    df_box = df_clean.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box['date']]
    df_box['month'] = [d.strftime('%b') for d in df_box['date']]

    # Ordering months
    df_box['month_num'] = df_box['date'].dt.month
    df_box = df_box.sort_values('month_num')

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    sns.boxplot(x='year', y='value', data=df_box, ax=axes[0], palette='Set2').set(
        xlabel='Year',
        ylabel='Page Views',
        title='Year-wise Box Plot (Trend)'
    )

    sns.boxplot(x='month', y='value', data=df_box, ax=axes[1], palette='Set3').set(
        xlabel='Month',
        ylabel='Page Views',
        title='Month-wise Box Plot (Seasonality)'
    )

    # Set month order
    axes[1].set_xticklabels(labels=[calendar.month_abbr[i] for i in range(1, 13)])

    # Save image and return fig
    plt.savefig('box_plot.png')
    return fig


import unittest
import time_series_visualizer
import matplotlib as mpl


class DataCleaningTestCase(unittest.TestCase):
    def test_data_cleaning(self):
        actual = len(time_series_visualizer.df_clean)
        expected = 1238
        self.assertEqual(actual, expected, "Expected DataFrame count after cleaning to be 1238.")


class LinePlotTestCase(unittest.TestCase):
    def setUp(self):
        self.fig = time_series_visualizer.draw_line_plot()
        self.ax = self.fig.axes[0]

    def test_line_plot_title(self):
        actual = self.ax.get_title()
        expected = "Daily freeCodeCamp Forum Page Views 5/2016-12/2019"
        self.assertEqual(actual, expected,
                         "Expected line plot title to be 'Daily freeCodeCamp Forum Page Views 5/2016-12/2019'")

    def test_line_plot_labels(self):
        actual = self.ax.get_xlabel()
        expected = "Date"
        self.assertEqual(actual, expected, "Expected line plot xlabel to be 'Date'")
        actual = self.ax.get_ylabel()
        expected = "Page Views"
        self.assertEqual(actual, expected, "Expected line plot ylabel to be 'Page Views'")

    def test_line_plot_data_quantity(self):
        actual = len(self.ax.lines[0].get_ydata())
        expected = 1238
        self.assertEqual(actual, expected, "Expected number of data points in line plot to be 1238.")


class BarPlotTestCase(unittest.TestCase):
    def setUp(self):
        self.fig = time_series_visualizer.draw_bar_plot()
        self.ax = self.fig.axes[0]

    def test_bar_plot_legend_labels(self):
        actual = []
        for label in self.ax.get_legend().get_texts():
            actual.append(label.get_text())
        expected = [calendar.month_name[i] for i in range(1, 13)]
        self.assertEqual(actual, expected, "Expected bar plot legend labels to be months of the year.")

    def test_bar_plot_labels(self):
        actual = self.ax.get_xlabel()
        expected = "Years"
        self.assertEqual(actual, expected, "Expected bar plot xlabel to be 'Years'")
        actual = self.ax.get_ylabel()
        expected = "Average Page Views"
        self.assertEqual(actual, expected, "Expected bar plot ylabel to be 'Average Page Views'")
        actual = []
        for label in self.ax.get_xaxis().get_majorticklabels():
            actual.append(label.get_text())
        expected = ['2016', '2017', '2018', '2019']
        self.assertEqual(actual, expected, "Expected bar plot secondary labels to be '2016', '2017', '2018', '2019'")

    def test_bar_plot_number_of_bars(self):
        actual = len([rect for rect in self.ax.get_children() if isinstance(rect, mpl.patches.Rectangle)])
        expected = 49
        self.assertEqual(actual, expected, "Expected a different number of bars in bar chart.")


class BoxPlotTestCase(unittest.TestCase):
    def setUp(self):
        self.fig = time_series_visualizer.draw_box_plot()
        self.ax1 = self.fig.axes[0]
        self.ax2 = self.fig.axes[1]

    def test_box_plot_number(self):
        actual = len(self.fig.get_axes())
        expected = 2
        self.assertEqual(actual, expected, "Expected two box plots in figure.")

    def test_box_plot_labels(self):
        actual = self.ax1.get_xlabel()
        expected = "Year"
        self.assertEqual(actual, expected, "'Year'")
        actual = self.ax1.get_ylabel()
        expected = "Page Views"
        self.assertEqual(actual, expected, "'Page Views'")
        actual = self.ax2.get_xlabel()
        expected = "Month"
        self.assertEqual(actual, expected, "'Month'")
        actual = self.ax2.get_ylabel()
        expected = "Page Views"
        self.assertEqual(actual, expected, "'Page Views'")
        actual = []
        for label in self.ax1.get_xaxis().get_majorticklabels():
            actual.append(label.get_text())
        expected = ['2016', '2017', '2018', '2019']
        self.assertEqual(actual, expected, "'2016', '2017', '2018', '2019'")
        actual = []
        for label in self.ax2.get_xaxis().get_majorticklabels():
            actual.append(label.get_text())
        expected = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.assertEqual(actual, expected,
                         "'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'")

    def test_box_plot_titles(self):
        actual = self.ax1.get_title()
        expected = "Year-wise Box Plot (Trend)"
        self.assertEqual(actual, expected, "'Year-wise Box Plot (Trend)'")
        actual = self.ax2.get_title()
        expected = "Month-wise Box Plot (Seasonality)"
        self.assertEqual(actual, expected, "'Month-wise Box Plot (Seasonality)'")

    def test_box_plot_number_of_boxes(self):
        actual = len(self.ax1.lines) / 6  # Every box has 6 lines
        expected = 4
        self.assertEqual(actual, expected, "Expected four boxes in box plot 1")
        actual = len(self.ax2.lines) / 6  # Every box has 6 lines
