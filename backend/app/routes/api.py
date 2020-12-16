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
invocation_post_args_schema.add_argument('greeting', required=True, location='json')
invocation_post_args_schema.add_argument('id', type=int, required=True, location='json')

# Request response schemas
invocation_response_schema = ns.model('Invocation', {
    'greeting': fields.String(required=True),
    'id': fields.Integer(required=True),
})

# API Routes
@ns.route("/ping")
class Health(Resource):
    @api.doc(description="API endpoint health info")
    def get(self):
        return {"status": "ok", "version": api.version}


@ns.route("/invocations")
class Invoke(Resource):
    # @api.doc(description responses={403: "Not Authorized"})
    @ns.doc("GET invocation history")
    def get(self):
        return {"status": "wtf"}

    @ns.doc("POST data to invoke the model API endpoint")
    @ns.expect(invocation_post_args_schema)
    @ns.marshal_with(invocation_response_schema)
    def post(self):
        args = invocation_post_args_schema.parse_args()
        print(args)
        greeting = args['greeting']
        idx = args['id']
        return [{"id": idx+1, "greeting": greeting}], 200
