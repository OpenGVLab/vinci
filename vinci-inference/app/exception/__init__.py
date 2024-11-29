from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from .handler import *

exception_handlers = {
    HTTPException: http_exception_handler,
    RequestValidationError: request_validation_exception_handler,
    Exception: last_exception_handler
}