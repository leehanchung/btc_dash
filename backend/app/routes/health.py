from flask import Blueprint
from flask_restx import Api, Resource


blueprint = Blueprint("health", __name__, template_folder="templates")
api = Api(
    blueprint,
    version="0.1",
    title="health check",
    description="API health check",
)


@api.route("/health")
class Health(Resource):
    # @api.doc(description="health checkup")
    def get(self):
        return {"status": "ok", "version": api.version}
