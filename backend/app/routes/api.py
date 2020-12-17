# from flask import Blueprint
# from flask_restx import Api, Resource, fields, reqparse


# blueprint = Blueprint("btcdash", __name__, template_folder="templates")
# api = Api(
#     blueprint,
#     version="0.1",
#     title="BTC Dash Machine Learning REST API",
#     description="API Endpoint for BTC Dash",
#     doc="/docs/",
# )
# ns = api.namespace("api", description="OHLCV + Predict")


# # Request argument schema
# invocation_post_args_schema = reqparse.RequestParser()
# invocation_post_args_schema.add_argument(
#     "data", type=list, required=True, location="json"
# )

# # Request response schemas
# invocation_response_schema = ns.model(
#     "Invocation",
#     {
#         "greeting": fields.String(required=True),
#         "id": fields.Integer(required=True),
#     },
# )


from fastapi import APIRouter

from app.routes.endpoints import invocation, ping

api_router = APIRouter()
api_router.include_router(invocation.router)  # , tags=["login"])
api_router.include_router(ping.router)  # , prefix="/users", tags=["users"])
