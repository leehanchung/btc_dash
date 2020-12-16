from flask import Blueprint
from flask_restx import Api, Resource, fields, reqparse


blueprint = Blueprint("btcdash", __name__, template_folder="templates")
api = Api(
    blueprint,
    version="0.1",
    title="BTC Dash Machine Learning REST API",
    description="API Endpoint for BTC Dash",
    doc="/docs/",
)
ns = api.namespace("api", description="OHLCV + Predict")


# Request argument schema
invocation_post_args_schema = reqparse.RequestParser()
invocation_post_args_schema.add_argument(
    "data", type=list, required=True, location="json"
)

# Request response schemas
invocation_response_schema = ns.model(
    "Invocation",
    {
        "greeting": fields.String(required=True),
        "id": fields.Integer(required=True),
    },
)


# API Routes
@ns.route("/ping")
class Health(Resource):
    @ns.doc(description="API endpoint health info")
    def get(self):
        return {"status": "ok", "version": api.version}


@ns.route("/invocations")
class Invoke(Resource):
    # @api.doc(description responses={403: "Not Authorized"})
    @ns.doc(description="GET invocation history")
    def get(self):
        return {"status": "wtf"}

    @ns.doc(description="POST data to invoke the model API endpoint")
    @ns.expect(invocation_post_args_schema)
    @ns.marshal_with(invocation_response_schema)
    def post(self):
        args = invocation_post_args_schema.parse_args()
        print(args)
        greeting = args["data"]  # noqa: F841

        return [{"id": 1, "greeting": "placeholder"}], 200
