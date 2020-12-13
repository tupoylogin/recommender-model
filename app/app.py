import time
from datetime import datetime

from flask import Flask, render_template
import pandas as pd

from model import PortfolioOptimizer

app = Flask(__name__,
            static_url_path='/static',
            static_folder='./static',
            template_folder='./templates')


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/demo')
def demo():
    coins = ["USDT", "TUSD", "DAI", "PAX", "USDC"]
    deep = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"][:len(coins)]
    muted = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
             "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"][:len(coins)]
    po = PortfolioOptimizer(coins, 600)
    x, asset = po.optimize(0.07)
    labels_asset = asset.time.values.ravel()
    data_asset = asset.close.values.ravel()
    labels_weights = list(x.keys())
    values_weights = list(x.values())
    bar_values = [{'x': data, 'y': value} for data, value in zip(
        [
            pd.to_datetime(date).strftime('%Y-%m-%d') for
            date in
            labels_asset
        ],
        list(data_asset)
    )]
    kwds = {'bar_data': bar_values,
            'palette': deep,
            'palette_hover': muted,
            'pie_labels': x['labels'],
            'pie_data': x['data']}
    return render_template('demo.html', **kwds)
