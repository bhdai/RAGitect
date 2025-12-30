"""URL validation service for secure document ingestion.

Implements SSRF prevention and URL scheme validation per NFR-S4.
"""

from ipaddress import AddressValueError, IPv4Address, IPv6Address, ip_address
from urllib.parse import urlparse


class InvalidURLSchemeError(ValueError):
    """Raised when URL uses a disallowed scheme (not HTTP/HTTPS)."""

    def __init__(self, scheme: str) -> None:
        """Initialize with the invalid scheme.

        Args:
            scheme: The URL scheme that was rejected.
        """
        self.scheme = scheme
        super().__init__(f"Only HTTP and HTTPS URLs are allowed. Got: {scheme}")


class SSRFAttemptError(ValueError):
    """Raised when URL points to a private/localhost address (SSRF prevention)."""

    def __init__(self, hostname: str) -> None:
        """Initialize with the blocked hostname.

        Args:
            hostname: The hostname or IP that was blocked.
        """
        self.hostname = hostname
        super().__init__(
            f"Private and localhost URLs are not allowed for security reasons. "
            f"Blocked: {hostname}"
        )


class URLValidator:
    """Validates URLs for secure document ingestion.

    Implements security validations per NFR-S4:
    - Only HTTP/HTTPS schemes allowed (AC2)
    - SSRF prevention: blocks localhost and private IPs (AC3)
    """

    # Hostnames that are always blocked (case-insensitive)
    BLOCKED_HOSTNAMES = frozenset({"localhost", "0.0.0.0"})

    # Allowed URL schemes
    ALLOWED_SCHEMES = frozenset({"http", "https"})

    def validate_url(self, url: str) -> None:
        """Validate a URL for secure ingestion.

        Args:
            url: The URL string to validate.

        Raises:
            InvalidURLSchemeError: If URL scheme is not HTTP/HTTPS.
            SSRFAttemptError: If URL points to localhost or private IP.
            ValueError: If URL is malformed.
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        self.validate_url_scheme(url)
        self.validate_url_hostname(url)

    def validate_url_scheme(self, url: str) -> None:
        """Validate that URL uses only HTTP or HTTPS scheme.

        Args:
            url: The URL string to validate.

        Raises:
            InvalidURLSchemeError: If scheme is not http or https.
        """
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()

        if not scheme:
            raise InvalidURLSchemeError("(empty)")

        if scheme not in self.ALLOWED_SCHEMES:
            raise InvalidURLSchemeError(scheme)

    def validate_url_hostname(self, url: str) -> None:
        """Validate URL hostname is not localhost or private IP.

        Args:
            url: The URL string to validate.

        Raises:
            SSRFAttemptError: If hostname is localhost or private IP.
        """
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            raise ValueError("URL must have a valid hostname")

        # Normalize hostname for comparison
        hostname_lower = hostname.lower()

        # Check blocked hostnames
        if hostname_lower in self.BLOCKED_HOSTNAMES:
            raise SSRFAttemptError(hostname)

        # Check if it's an IP address (IPv4 or IPv6) and if it's private
        if self.is_private_ip(hostname):
            raise SSRFAttemptError(hostname)

    def is_private_ip(self, hostname: str) -> bool:
        """Check if hostname is a private, loopback, or link-local IP address.

        Args:
            hostname: The hostname or IP address string to check.

        Returns:
            True if the address is private/loopback/link-local, False otherwise.
            Returns False for hostnames that are not valid IP addresses
            (DNS resolution is deferred to later processing stages).
        """
        try:
            ip = ip_address(hostname)
        except (AddressValueError, ValueError):
            # Not a valid IP address - it's a hostname
            # DNS resolution validation is deferred to Story 5.2 processors
            return False

        # Check all private/restricted ranges
        if isinstance(ip, (IPv4Address, IPv6Address)):
            # is_loopback: 127.0.0.0/8 for IPv4, ::1 for IPv6
            if ip.is_loopback:
                return True

            # is_private: RFC 1918 ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
            # Also includes fc00::/7 for IPv6
            if ip.is_private:
                return True

            # is_link_local: 169.254.0.0/16 for IPv4, fe80::/10 for IPv6
            # This includes the cloud metadata endpoint 169.254.169.254
            if ip.is_link_local:
                return True

            # is_reserved covers other reserved ranges
            if ip.is_reserved:
                return True

            # is_unspecified: 0.0.0.0 for IPv4, :: for IPv6
            if ip.is_unspecified:
                return True

        return False
