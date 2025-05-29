# TODO(martin): Just for now it will return the type of HAR  # noqa: FIX002, TD003

from enum import Enum

from pydantic import BaseModel, Field


class HarType(Enum):
    """The type of HAR (Human Activity Recognition) activity as label."""

    WALKING = "walking"
    RUNNING = "running"
    SHUFFLING = "shuffling"
    STAIRS_UP = "stairs_up"
    STAIRS_DOWN = "stairs_down"
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    CYCLING_SIT = "cycling_sit"
    CYCLING_STAND = "cycling_stand"
    CYCLING_SIT_INACTIVE = "cycling_sit_inactive"


# Responses
class HarResponse(BaseModel):
    """The HAR model containing the type of activity."""

    type: HarType


# Requests
class HarRequest(BaseModel):
    """Request model for HAR data containing accelerometer readings.

    All the attributes represent accelerometer readings in three axes (X, Y, Z). The range of the values is set to be between -30 and 30, which is a common range for accelerometer data. Please see the [notebook](/notebook/har_clustering.ipynb) for more details on the data collection and processing.
    """

    back_x: float = Field(description="Back X acceleration", gt=-30, lt=30)
    back_y: float = Field(description="Back Y acceleration", gt=-30, lt=30)
    back_z: float = Field(description="Back Z acceleration", gt=-30, lt=30)
    thigh_x: float = Field(description="Thigh X acceleration", gt=-30, lt=30)
    thigh_y: float = Field(description="Thigh Y acceleration", gt=-30, lt=30)
    thigh_z: float = Field(description="Thigh Z acceleration", gt=-30, lt=30)
