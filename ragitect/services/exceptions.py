"""Custom exceptions for service layer operations"""


class ServiceError(Exception):
    """Base exception for all service-related errors

    Attributes:
        message: Description of the error
    """

    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class FileSizeExceededError(ServiceError):
    """Raised when uploaded file exceeds maximum allowed size

    Attributes:
        filename: Name of the file that exceeded size limit
        file_size_mb: Actual file size in megabytes
        max_size_mb: Maximum allowed size in megabytes
    """

    filename: str
    file_size_mb: float
    max_size_mb: float

    def __init__(self, filename: str, file_size_mb: float, max_size_mb: float):
        self.filename = filename
        self.file_size_mb = file_size_mb
        self.max_size_mb = max_size_mb
        message = (
            f"File '{filename}' is too large ({file_size_mb:.2f}MB). "
            f"Maximum allowed size is {max_size_mb}MB"
        )
        super().__init__(message)
