import numpy as np
import pandas as pd

from boto3 import Session
from botocore.exceptions import ClientError
from copy import copy
from bokeh.events import ButtonClick
from bokeh.io import curdoc, show
from bokeh.layouts import layout, widgetbox, column, row
from bokeh.models import (Plot, LegendItem, Legend, LinearAxis, DatetimeAxis, DatetimeTickFormatter,
                          TextInput, Button, Paragraph)
from bokeh.models.annotations import Title
from bokeh.models.glyphs import Line, Text
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import Dropdown


# track all necessary data objects and variables
DATA_STORE = {
    'symbol': None,
    'data': None,
    'data_source': None,
    'valid_symbols': None,
    'dropdown': None,
    'title_source': None,
    's3_bucket': None,
    's3_key': None,
    'boto_profile': None,
    'text_output': None,
    's3_bucket_input': None,
    's3_key_input': None,
    'text_button': None,
    'boto_profile_input': None,
    'default_button': None
}

DEFAULTS = {
    's3_bucket': 'mlsl-mle-ws',
    's3_key': 'training-data/crypto_test_preds.csv',
    'boto_profile': 'kyle_general'
}

DATA_STORE.update(**copy(DEFAULTS))


def retrieve_df_from_s3(s3_bucket, s3_key, profile):
    if profile == '':
        print('Boto profile not provided, defaulting to system default profile')
        profile = None

    # Instantiate new S3 Boto3 client using provided AWS CLI profile
    s3_client = Session(profile_name=profile).client('s3')
    try:
        # Retrieve object present in s3_bucket, at s3_key
        obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    except ClientError as e:
        df_ = None
        err = e
    else:
        # Attempt to read in content
        df_ = pd.read_csv(obj['Body'])
        err = None

    return df_, err


# TODO - update this to allow data pull via specified URL (shouldn't have to be s3)
def retrieve_data(initial=False):
    s3_bucket = DATA_STORE['s3_bucket']
    s3_key = DATA_STORE['s3_key']
    boto_profile = DATA_STORE['boto_profile']

    if s3_bucket is None or s3_key is None or boto_profile is None:
        stmt = 'At least one of s3_bucket, s3_key, boto_profile not provided.\n' \
               'Defaulting to load from local crypto_test_preds.csv'
        df = pd.read_csv('data/crypto_val_preds.csv').set_index(['symbol', 'index'])
    else:
        s3_url = 's3://{b}/{k}'.format(b=s3_bucket, k=s3_key)
        df, err = retrieve_df_from_s3(s3_bucket=s3_bucket, s3_key=s3_key, profile=boto_profile)
        if err is not None:
            stmt = 'Failed to read data from s3 object: \n{u}\n' \
                   'Error message: {e}'.format(u=s3_url, e=err)
            if initial:
                print('Loading data from local source crypto_val_preds.csv')
                df = pd.read_csv('data/crypto_test_preds.csv').set_index(['symbol', 'index'])
            else:
                df = None
        else:
            stmt = 'Succesfully loaded data from s3 object: \n{}'.format(s3_url)
            df = df.set_index(['symbol', 'index'])

    print(stmt)
    DATA_STORE['text_output'].text = stmt
    DATA_STORE.update({'data': df})


def update_data_source():
    # retrieve active symbol
    symbol = DATA_STORE['symbol']
    df = DATA_STORE['data']

    if df is not None:
        print('dataframe not empty')
        df_ = df.xs(symbol).copy()
        df_['symbol'] = symbol

        # Establish data for new symbol
        data = {
            'index': range(df_.shape[0]),
            'symbol': [symbol]*df_.shape[0],
            'date': pd.to_datetime(df_.loc[:, 'date'].values),
            'pred_open': df_.loc[:, 'pred_open'].values,
            'val_open': df_.loc[:, 'val_open'].values
        }

        if DATA_STORE['data_source'] is None:
            DATA_STORE.update({'data_source': ColumnDataSource()})

        if DATA_STORE['title_source'] is None:
            DATA_STORE.update({'title_source': ColumnDataSource()})

        DATA_STORE['data_source'].data = data

        mad = np.round(np.mean(np.abs(df_['pred_open'] - df_['val_open'])), 2)
        title_text = 'Symbol: {}, MAD: {}'.format(symbol, mad)

        max_y = max(max(data['pred_open']), max(data['val_open']))
        title_data = {'text': [title_text], 'x': [data['date'][len(data['date'])//3]], 'y': [max_y]}

        DATA_STORE['title_source'].data = title_data
    else:
        print('dataframe is empty. clearing data sources...')
        # Clear out data contents so plot will go blank
        data = {
            'index': [],
            'symbol': [],
            'date': [],
            'pred_open': [],
            'val_open': []
        }
        DATA_STORE['data_source'].data = data
        DATA_STORE['title_source'].data = {'text': [], 'x': [], 'y': []}


def set_data_symbol(symbol=None):
    if symbol is None:
        symbol = DATA_STORE['valid_symbols'][0]

    print('Setting active symbol to {}'.format(symbol))
    DATA_STORE.update({'symbol': symbol})


def dropdown_callback(attr, old, new):
    set_data_symbol(new)
    update_data_source()


def text_button_callback(_):
    DATA_STORE['s3_bucket'] = DATA_STORE['s3_bucket_input'].value
    DATA_STORE['s3_key'] = DATA_STORE['s3_key_input'].value
    DATA_STORE['boto_profile'] = DATA_STORE['boto_profile_input'].value

    s3_bucket = DATA_STORE['s3_bucket']
    s3_key = DATA_STORE['s3_key']
    boto_profile = DATA_STORE['boto_profile']

    if 'preds' in s3_key:
        print('S3 bucket: {b}\nS3 key: {k}\nAWS Profile: {p}'.format(b=s3_bucket, k=s3_key, p=boto_profile))
        retrieve_data()
        update_data_source()
    else:
        print('string "preds" must be present in s3 key')
        DATA_STORE['text_output'].text = 'String "preds" not found in s3 key. Did not update data sources'


def default_button_callback(_):
    print('Resetting default text inputs...')

    DATA_STORE.update(**copy(DEFAULTS))

    DATA_STORE['s3_bucket_input'].value = DATA_STORE['s3_bucket']
    DATA_STORE['s3_key_input'].value = DATA_STORE['s3_key']
    DATA_STORE['boto_profile_input'].value = DATA_STORE['boto_profile']

    retrieve_data()
    update_data_source()


# Plot the predicted vs actual of a given symbol
def gen_bokeh_plot():
    # retrieve current active symbol and data source
    data_source = DATA_STORE['data_source']
    title_source = DATA_STORE['title_source']

    # calculate mean absolute deviation (MAD)
    title = Title(text='Predicted vs Actual Market Open Price')

    plot = Plot(title=title, plot_width=1200, plot_height=800, h_symmetry=False, v_symmetry=False,
                min_border=0, toolbar_location='right')

    val_glyph = Line(x='date', y='val_open', line_color='black', line_width=1.5)
    pred_glyph = Line(x='date', y='pred_open', line_color='blue', line_width=1.5, line_dash='dashed')
    plot.add_glyph(data_source, val_glyph)  # renderer 0
    plot.add_glyph(data_source, pred_glyph)  # renderer 1

    text_glyph = Text(x='x', y='y', text='text')
    plot.add_glyph(title_source, text_glyph)

    # Format plot
    li1 = LegendItem(label='actual', renderers=[plot.renderers[0]])
    li2 = LegendItem(label='predicted', renderers=[plot.renderers[1]])
    legend1 = Legend(items=[li1, li2], location='top_right')
    plot.add_layout(legend1)

    xaxis = DatetimeAxis()
    xaxis.formatter = DatetimeTickFormatter(years=['%Y'], months=['%m/%Y'], days=['%d/%m/%Y'])
    xaxis.major_label_orientation = np.pi / 4
    plot.add_layout(xaxis, 'below')

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')

    return plot


def initialize():
    text_output = Paragraph(text='Status', width=300, height=100)
    DATA_STORE.update({'text_output': text_output})

    retrieve_data(initial=True)
    df = DATA_STORE['data']

    symbols = sorted(list(set(df.index.get_level_values(level=0))))

    if DATA_STORE['valid_symbols'] is None:
        DATA_STORE.update({'valid_symbols': symbols})

    set_data_symbol()

    menu = [*zip(symbols, symbols)]
    dropdown = Dropdown(label='Select Symbol', button_type='primary', menu=menu)
    dropdown.on_change('value', dropdown_callback)

    bucket_input = TextInput(value='mlsl-mle-ws', title='S3 Bucket:')
    key_input = TextInput(value='training-data/crypto_test_preds.csv', title='S3 Key:')
    profile_input = TextInput(value='kyle_general', title='Boto Profile:')
    text_button = Button(label='Pull Data')
    text_button.on_event(ButtonClick, text_button_callback)
    default_button = Button(label='Reset Defaults')
    default_button.on_event(ButtonClick, default_button_callback)

    DATA_STORE.update({
        'dropdown': dropdown,
        's3_bucket_input': bucket_input,
        's3_key_input': key_input,
        'text_button': text_button,
        'boto_profile_input': profile_input,
        'default_button': default_button
    })


def build_plot():
    update_data_source()
    plot = gen_bokeh_plot()

    dd_box = widgetbox(DATA_STORE['dropdown'])
    text_inputs = widgetbox(
        [DATA_STORE['s3_bucket_input'], DATA_STORE['s3_key_input'], DATA_STORE['boto_profile_input']]
    )
    button_box = widgetbox(DATA_STORE['text_button'], DATA_STORE['text_output'])
    default_box = widgetbox(DATA_STORE['default_button'])

    header_row = row([dd_box, text_inputs, button_box, default_box])
    layout_ = layout(column([header_row, plot]))

    curdoc().add_root(layout_)

    show(layout_)


initialize()
build_plot()
