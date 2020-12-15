from flask import Blueprint
from flask_restx import Api, Resource


blueprint = Blueprint("btcdash", __name__, template_folder="templates")
api = Api(
    blueprint,
    version="0.1",
    title="BTC Dash Machine Learning REST API",
    description="API Endpoint for BTC Dash",
    doc="/docs/",
)
ns = api.namespace("api", description="OHLCV + Predict")


@ns.route("/ping")
class Health(Resource):
    @api.doc(description="API endpoint health info")
    def get(self):

        return {"status": "ok", "version": api.version}


@ns.route("/invocations")
class Invoke(Resource):
    # @api.doc(description responses={403: "Not Authorized"})
    def get(self):
        return {"status": "wtf"}
        # api.abort(403)
