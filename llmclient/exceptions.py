from logging import LogRecord

from llmclient.messages import EMPTY_CONTENT_BASE_MSG


class JSONSchemaValidationError(ValueError):
    """Raised when the completion does not match the specified schema."""


class MalformedMessageError(ValueError):
    """Error to throw if some aspect of a Message variant is malformed."""

    @classmethod
    def common_retryable_errors_log_filter(cls, record: LogRecord) -> bool:
        """
        Filter out common parsing failures not worth looking into from logs.

        Returns:
            False if the LogRecord should be filtered out, otherwise True to keep it.
        """
        # NOTE: match both this Exception type's name and its content, to be robust
        return not all(x in record.msg for x in (cls.__name__, EMPTY_CONTENT_BASE_MSG))
