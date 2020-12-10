# from flask import Blueprint, render_template, abort
# from flask_restx import Api, Namespace, Resource, fields


# blueprint = Blueprint('btcdash',
#                       __name__,
#                       template_folder='templates')
# api = Api(blueprint,
#           version='0.1',
#           title='BTC_USD API',
#           description="BTC_USD API for BTC_Dash")

# namespace = api.namespace('api',
#                           description="OHLCV + Predict")

# @namespace.route('/')
# class Health(Resource):
#     # @api.doc(description="health checkup")
#     def get(self):
#         return {'status': 'ok', 'version': api.version}
