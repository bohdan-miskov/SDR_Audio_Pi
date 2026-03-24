from dataclasses import dataclass
from typing import Any, Optional, Union
from enum import IntEnum, Enum


class DbOperation(str, Enum):
    """
    Перелік усіх можливих операцій з БД.
    """

    # Objects
    ADD_OBJECT = "add_object"
    UPDATE_OBJECT = "update_object"
    DELETE_OBJECT = "delete_object"
    GET_OBJECTS_PAGE = "get_objects_page"
    GET_ALL_OBJECTS = "get_all_objects"

    # Classes
    ADD_CLASS = "add_class"
    UPDATE_CLASS = "update_class"
    DELETE_CLASS = "delete_class"
    GET_CLASSES = "get_classes"
    RENAME_CLASS = "rename_class"

    # Errors/System
    UNKNOWN = "unknown"


class StatusCode(IntEnum):
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    NOT_FOUND = 404
    CONFLICT = 409
    INTERNAL_ERROR = 500


@dataclass
class ServiceResponse:
    status: StatusCode
    message: str
    operation: Union[DbOperation, str] = DbOperation.UNKNOWN
    data: Optional[Any] = None

    @property
    def is_success(self) -> bool:
        return 200 <= int(self.status) < 300

    @property
    def is_error(self) -> bool:
        return int(self.status) >= 400

    def to_dict(self) -> dict:
        return {
            "status": int(self.status),
            "message": self.message,
            "data": self.data,
            "operation": (
                self.operation.value
                if isinstance(self.operation, DbOperation)
                else str(self.operation)
            ),
        }

    @staticmethod
    def from_dict(data: dict) -> "ServiceResponse":
        op_str = data.get("operation", "unknown")
        try:
            op = DbOperation(op_str)
        except ValueError:
            op = op_str

        return ServiceResponse(
            status=StatusCode(data.get("status", 500)),
            message=data.get("message", ""),
            operation=op,
            data=data.get("data"),
        )
