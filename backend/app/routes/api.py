from flask import Blueprint, render_template, abort
from flask_restx import Api, Resource, fields
from jinja2 import TemplateNotFound


blueprint = Blueprint('api',
                    __name__,
                    template_folder='templates')
api = Api(blueprint,
          version='0.1',
          title='BTC_USD API',
          description="BTC_USD API for BTC_Dash")
# namespace = api.namespace('btcusd_namespace',
#                           description="OHLCV + Predict")

# @namespace.route('/')

