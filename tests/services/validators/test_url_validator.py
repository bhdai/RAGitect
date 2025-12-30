"""Unit tests for URLValidator service.

Tests URL validation logic for SSRF prevention and scheme validation (AC2, AC3, AC5).
"""

import pytest

from ragitect.services.validators.url_validator import (
    InvalidURLSchemeError,
    SSRFAttemptError,
    URLValidator,
)


class TestURLValidatorSchemeValidation:
    """Tests for URL scheme validation (AC2)."""

    def test_valid_https_url_passes(self) -> None:
        """HTTPS URLs should pass validation."""
        validator = URLValidator()
        # Should not raise any exception
        validator.validate_url("https://example.com")

    def test_valid_http_url_passes(self) -> None:
        """HTTP URLs should pass validation."""
        validator = URLValidator()
        validator.validate_url("http://wikipedia.org")

    def test_valid_url_with_path_passes(self) -> None:
        """URLs with paths should pass validation."""
        validator = URLValidator()
        validator.validate_url("https://example.com/path/to/resource")

    def test_valid_url_with_query_params_passes(self) -> None:
        """URLs with query parameters should pass validation."""
        validator = URLValidator()
        validator.validate_url("https://example.com/search?q=test&page=1")

    def test_file_scheme_blocked(self) -> None:
        """file:// scheme must be blocked (security: local file access)."""
        validator = URLValidator()
        with pytest.raises(InvalidURLSchemeError) as exc_info:
            validator.validate_url("file:///etc/passwd")
        assert "Only HTTP and HTTPS URLs are allowed" in str(exc_info.value)

    def test_ftp_scheme_blocked(self) -> None:
        """ftp:// scheme must be blocked."""
        validator = URLValidator()
        with pytest.raises(InvalidURLSchemeError) as exc_info:
            validator.validate_url("ftp://server.com/file.txt")
        assert "Only HTTP and HTTPS URLs are allowed" in str(exc_info.value)

    def test_javascript_scheme_blocked(self) -> None:
        """javascript: scheme must be blocked (XSS prevention)."""
        validator = URLValidator()
        with pytest.raises(InvalidURLSchemeError) as exc_info:
            validator.validate_url("javascript:alert(1)")
        assert "Only HTTP and HTTPS URLs are allowed" in str(exc_info.value)

    def test_data_scheme_blocked(self) -> None:
        """data: scheme must be blocked."""
        validator = URLValidator()
        with pytest.raises(InvalidURLSchemeError) as exc_info:
            validator.validate_url("data:text/html,<h1>test</h1>")
        assert "Only HTTP and HTTPS URLs are allowed" in str(exc_info.value)


class TestURLValidatorSSRFPrevention:
    """Tests for SSRF attack prevention (AC3)."""

    def test_localhost_blocked(self) -> None:
        """localhost must be blocked to prevent SSRF."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://localhost:8080")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_localhost_no_port_blocked(self) -> None:
        """localhost without port must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://localhost")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_127_0_0_1_blocked(self) -> None:
        """127.0.0.1 loopback IP must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://127.0.0.1")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_0_0_0_0_blocked(self) -> None:
        """0.0.0.0 must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://0.0.0.0")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_private_ip_192_168_blocked(self) -> None:
        """192.168.x.x private IP range must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://192.168.1.1")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_private_ip_10_blocked(self) -> None:
        """10.x.x.x private IP range must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://10.0.0.1")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_private_ip_172_16_blocked(self) -> None:
        """172.16.x.x private IP range must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://172.16.0.1")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_cloud_metadata_endpoint_blocked(self) -> None:
        """169.254.169.254 cloud metadata endpoint must be blocked (AWS/GCP/Azure)."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://169.254.169.254/latest/meta-data/")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_ipv6_localhost_blocked(self) -> None:
        """IPv6 localhost [::1] must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://[::1]")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)

    def test_ipv6_loopback_with_port_blocked(self) -> None:
        """IPv6 localhost with port must be blocked."""
        validator = URLValidator()
        with pytest.raises(SSRFAttemptError) as exc_info:
            validator.validate_url("http://[::1]:8080")
        assert "Private and localhost URLs are not allowed" in str(exc_info.value)


class TestURLValidatorEdgeCases:
    """Edge case tests for URLValidator."""

    def test_valid_public_ip_passes(self) -> None:
        """Public IP addresses should pass validation."""
        validator = URLValidator()
        # Google's public DNS
        validator.validate_url("http://8.8.8.8")

    def test_valid_international_domain_passes(self) -> None:
        """International domain names (IDN) should pass validation."""
        validator = URLValidator()
        validator.validate_url("https://中文.com")

    def test_url_with_port_passes(self) -> None:
        """URLs with non-standard ports should pass if not private."""
        validator = URLValidator()
        validator.validate_url("https://example.com:8443/api")

    def test_empty_url_raises_error(self) -> None:
        """Empty URL string should raise an error."""
        validator = URLValidator()
        with pytest.raises((InvalidURLSchemeError, ValueError)):
            validator.validate_url("")

    def test_malformed_url_raises_error(self) -> None:
        """Malformed URL should raise an error."""
        validator = URLValidator()
        with pytest.raises((InvalidURLSchemeError, ValueError)):
            validator.validate_url("not-a-valid-url")


class TestURLValidatorHelperMethods:
    """Tests for helper methods in URLValidator."""

    def test_is_private_ip_loopback(self) -> None:
        """Loopback IPs should be detected as private."""
        validator = URLValidator()
        assert validator.is_private_ip("127.0.0.1") is True
        assert validator.is_private_ip("127.0.0.255") is True

    def test_is_private_ip_class_a(self) -> None:
        """10.x.x.x range should be detected as private."""
        validator = URLValidator()
        assert validator.is_private_ip("10.0.0.1") is True
        assert validator.is_private_ip("10.255.255.255") is True

    def test_is_private_ip_class_b(self) -> None:
        """172.16.x.x - 172.31.x.x range should be detected as private."""
        validator = URLValidator()
        assert validator.is_private_ip("172.16.0.1") is True
        assert validator.is_private_ip("172.31.255.255") is True

    def test_is_private_ip_class_c(self) -> None:
        """192.168.x.x range should be detected as private."""
        validator = URLValidator()
        assert validator.is_private_ip("192.168.0.1") is True
        assert validator.is_private_ip("192.168.255.255") is True

    def test_is_private_ip_link_local(self) -> None:
        """169.254.x.x link-local range should be detected as private."""
        validator = URLValidator()
        assert validator.is_private_ip("169.254.169.254") is True
        assert validator.is_private_ip("169.254.0.1") is True

    def test_is_private_ip_public(self) -> None:
        """Public IPs should NOT be detected as private."""
        validator = URLValidator()
        assert validator.is_private_ip("8.8.8.8") is False
        assert validator.is_private_ip("1.1.1.1") is False

    def test_is_private_ip_ipv6_loopback(self) -> None:
        """IPv6 loopback should be detected as private."""
        validator = URLValidator()
        assert validator.is_private_ip("::1") is True

    def test_is_private_ip_hostname_not_ip(self) -> None:
        """Hostnames (not IPs) should return False (DNS resolution deferred)."""
        validator = URLValidator()
        # localhost hostname is handled separately, not by is_private_ip
        assert validator.is_private_ip("example.com") is False
