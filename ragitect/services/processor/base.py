from abc import ABC, abstractmethod


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processors.

    All processors must implement:
    1. supported_formats() - which file extensions they handle
    2. process() - how to extract text from bytes
    """

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """
        Return list of supported file extensions.

        Returns:
            List of lowercase extensions including the dot
            Example: ['.txt', '.md', '.markdown']
        """
        pass

    @abstractmethod
    def process(self, file_bytes: bytes, file_name: str) -> str:
        """
        Extract raw text from document bytes.

        Args:
            file_bytes: Raw bytes of the uploaded file
            file_name: Original filename (used for extension detection)

        Returns:
            Extracted text as string

        Raises:
            Exception: If processing fails (let it bubble up for now)
        """
        pass
