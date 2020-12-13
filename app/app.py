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
    rate = 7
    deep = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"][:len(coins)]
    muted = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
             "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"][:len(coins)]
    po = PortfolioOptimizer(coins, 730)
    x, asset = po.optimize(rate / 100)
    kwds = {'area_labels': asset['time'],
            'area_data': asset['data'],
            'palette': deep,
            'palette_hover': muted,
            'pie_labels': x['labels'],
            'pie_data': x['data'], 
            'all_assets': ', '.join(coins),
            'projected_mean': f'{rate} %'
            }
    return render_template('demo.html', **kwds)

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0')