from typing import Any

from fastapi import APIRouter  # , Depends

# from pydantic.networks import EmailStr

# from app import models, schemas
# from app.api import deps
# from app.core.celery_app import celery_app

router = APIRouter()

# @ns.route("/invocations")
# class Invoke(Resource):
#     # @api.doc(description responses={403: "Not Authorized"})
#     @ns.doc(description="GET invocation history")
#     def get(self):
#         return {"status": "wtf"}

#     @ns.doc(description="POST data to invoke the model API endpoint")
#     @ns.expect(invocation_post_args_schema)
#     @ns.marshal_with(invocation_response_schema)
#     def post(self):
#         args = invocation_post_args_schema.parse_args()
#         print(args)
#         greeting = args["data"]  # noqa: F841

#         return [{"id": 1, "greeting": "placeholder"}], 200


@router.get("/invocation/")  # , response_model=schemas.Msg, status_code=201)
def invocation_history() -> Any:
    return {"msg": "Word received"}


@router.post("/invocation/")  # , response_model=schemas.Msg, status_code=201)
def invoke_model() -> Any:
    """
    Test emails.
    """
    return {"msg": "Test email sent"}
