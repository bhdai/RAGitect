"""Tests for Docker Compose setup and service configuration."""

import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def docker_compose_config(project_root: Path) -> dict:
    """Load and parse docker-compose.yml."""
    compose_file = project_root / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml must exist"
    with open(compose_file) as f:
        return yaml.safe_load(f)


class TestDockerComposeConfiguration:
    """Test suite for Docker Compose configuration."""

    def test_docker_compose_file_exists(self, project_root: Path):
        """Verify docker-compose.yml exists at project root."""
        compose_file = project_root / "docker-compose.yml"
        assert compose_file.exists()

    def test_all_required_services_defined(self, docker_compose_config: dict):
        """Verify all three required services are defined."""
        services = docker_compose_config.get("services", {})
        required_services = {"db", "backend", "frontend"}
        assert required_services.issubset(services.keys()), (
            f"Missing services: {required_services - services.keys()}"
        )

    def test_database_service_uses_pgvector(self, docker_compose_config: dict):
        """Verify database service uses pgvector image with PostgreSQL 13+."""
        db_config = docker_compose_config["services"]["db"]
        image = db_config.get("image", "")
        assert "pgvector" in image, "Database must use pgvector image"
        # Extract version number - pg18 means PostgreSQL 18
        assert any(f"pg{v}" in image for v in range(13, 30)), (
            "PostgreSQL version must be 13+"
        )

    def test_database_has_healthcheck(self, docker_compose_config: dict):
        """Verify database service has healthcheck configured."""
        db_config = docker_compose_config["services"]["db"]
        assert "healthcheck" in db_config, "Database must have healthcheck"

    def test_backend_service_configuration(self, docker_compose_config: dict):
        """Verify backend service is properly configured."""
        backend_config = docker_compose_config["services"]["backend"]

        # Check build context
        assert "build" in backend_config
        assert backend_config["build"]["dockerfile"] == "Dockerfile"

        # Check port mapping
        ports = backend_config.get("ports", [])
        assert any("8000" in str(p) for p in ports), "Backend must expose port 8000"

        # Check dependency on db
        depends = backend_config.get("depends_on", {})
        assert "db" in depends or "db" in str(depends), "Backend must depend on db"

    def test_frontend_service_configuration(self, docker_compose_config: dict):
        """Verify frontend service is properly configured."""
        frontend_config = docker_compose_config["services"]["frontend"]

        # Check build context
        assert "build" in frontend_config
        assert frontend_config["build"]["context"] == "./frontend"

        # Check port mapping
        ports = frontend_config.get("ports", [])
        assert any("3000" in str(p) for p in ports), "Frontend must expose port 3000"

        # Check dependency on backend
        depends = frontend_config.get("depends_on", [])
        assert "backend" in depends or "backend" in str(depends), (
            "Frontend must depend on backend"
        )

    def test_database_environment_variables(self, docker_compose_config: dict):
        """Verify database has required environment variables."""
        db_config = docker_compose_config["services"]["db"]
        env = db_config.get("environment", {})
        required_vars = ["POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
        for var in required_vars:
            assert var in env, f"Database must have {var} environment variable"

    def test_backend_has_database_url(self, docker_compose_config: dict):
        """Verify backend has DATABASE_URL configured."""
        backend_config = docker_compose_config["services"]["backend"]
        env = backend_config.get("environment", {})
        assert "DATABASE_URL" in env, "Backend must have DATABASE_URL"
        assert "db:5432" in env["DATABASE_URL"], (
            "DATABASE_URL must reference db service"
        )

    def test_volumes_defined(self, docker_compose_config: dict):
        """Verify persistent volumes are defined."""
        volumes = docker_compose_config.get("volumes", {})
        assert "postgres_data" in volumes, "postgres_data volume must be defined"


class TestBackendDockerfile:
    """Test suite for backend Dockerfile."""

    def test_backend_dockerfile_exists(self, project_root: Path):
        """Verify backend Dockerfile exists."""
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists(), "Backend Dockerfile must exist at project root"

    def test_backend_dockerfile_uses_python(self, project_root: Path):
        """Verify Dockerfile uses Python 3.13."""
        dockerfile = project_root / "Dockerfile"
        content = dockerfile.read_text()
        assert "python:3.13" in content, "Dockerfile must use Python 3.13"

    def test_backend_dockerfile_uses_uv(self, project_root: Path):
        """Verify Dockerfile uses uv for package management."""
        dockerfile = project_root / "Dockerfile"
        content = dockerfile.read_text()
        assert "uv" in content, "Dockerfile must use uv package manager"

    def test_backend_dockerfile_exposes_port(self, project_root: Path):
        """Verify Dockerfile exposes port 8000."""
        dockerfile = project_root / "Dockerfile"
        content = dockerfile.read_text()
        assert "EXPOSE 8000" in content, "Dockerfile must expose port 8000"


class TestFrontendDockerfile:
    """Test suite for frontend Dockerfile."""

    def test_frontend_dockerfile_exists(self, project_root: Path):
        """Verify frontend Dockerfile exists."""
        dockerfile = project_root / "frontend" / "Dockerfile"
        assert dockerfile.exists(), "Frontend Dockerfile must exist"

    def test_frontend_dockerfile_uses_node(self, project_root: Path):
        """Verify Dockerfile uses Node.js."""
        dockerfile = project_root / "frontend" / "Dockerfile"
        content = dockerfile.read_text()
        assert "node:" in content.lower(), "Dockerfile must use Node.js"

    def test_frontend_dockerfile_is_multistage(self, project_root: Path):
        """Verify Dockerfile uses multi-stage build."""
        dockerfile = project_root / "frontend" / "Dockerfile"
        content = dockerfile.read_text()
        # Multi-stage builds have multiple FROM statements
        from_count = content.lower().count("from ")
        assert from_count >= 2, "Dockerfile should use multi-stage build"


class TestFrontendStructure:
    """Test suite for frontend project structure."""

    def test_frontend_directory_exists(self, project_root: Path):
        """Verify frontend directory exists."""
        frontend_dir = project_root / "frontend"
        assert frontend_dir.exists() and frontend_dir.is_dir()

    def test_frontend_has_package_json(self, project_root: Path):
        """Verify frontend has package.json."""
        package_json = project_root / "frontend" / "package.json"
        assert package_json.exists()

    def test_frontend_has_typescript(self, project_root: Path):
        """Verify frontend has TypeScript configured."""
        tsconfig = project_root / "frontend" / "tsconfig.json"
        assert tsconfig.exists(), "TypeScript config must exist"

    def test_frontend_has_tailwind(self, project_root: Path):
        """Verify frontend has Tailwind CSS configured."""
        # Check for Tailwind in package.json dependencies
        package_json = project_root / "frontend" / "package.json"
        with open(package_json) as f:
            pkg = json.load(f)

        all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        assert "tailwindcss" in all_deps, "Tailwind CSS must be in dependencies"

    def test_frontend_has_eslint(self, project_root: Path):
        """Verify frontend has ESLint configured."""
        # Check for eslint config file
        eslint_config = project_root / "frontend" / "eslint.config.mjs"
        assert eslint_config.exists(), "ESLint config must exist"

    def test_frontend_has_src_directory(self, project_root: Path):
        """Verify frontend uses src directory structure."""
        src_dir = project_root / "frontend" / "src"
        assert src_dir.exists() and src_dir.is_dir(), "Frontend must have src directory"

    def test_frontend_has_app_directory(self, project_root: Path):
        """Verify frontend uses App Router (app directory)."""
        app_dir = project_root / "frontend" / "src" / "app"
        assert app_dir.exists() and app_dir.is_dir(), (
            "Frontend must have src/app directory for App Router"
        )


class TestBackendStructure:
    """Test suite for backend project structure."""

    def test_ragitect_directory_exists(self, project_root: Path):
        """Verify ragitect backend directory still exists."""
        ragitect_dir = project_root / "ragitect"
        assert ragitect_dir.exists() and ragitect_dir.is_dir(), (
            "ragitect directory must remain intact"
        )

    def test_main_py_exists(self, project_root: Path):
        """Verify main.py entry point exists."""
        main_py = project_root / "main.py"
        assert main_py.exists(), "main.py must exist"

    def test_main_py_has_fastapi_app(self, project_root: Path):
        """Verify main.py defines FastAPI app."""
        main_py = project_root / "main.py"
        content = main_py.read_text()
        assert "FastAPI" in content, "main.py must import FastAPI"
        assert "app = FastAPI" in content or "app=FastAPI" in content.replace(
            " ", ""
        ), "main.py must define FastAPI app"


class TestFastAPIApplication:
    """Test suite for FastAPI application endpoints."""

    def test_health_endpoint_exists(self, project_root: Path):
        """Verify health check endpoint is defined."""
        main_py = project_root / "main.py"
        content = main_py.read_text()
        assert "/health" in content, "Health check endpoint must be defined"

    def test_cors_configured(self, project_root: Path):
        """Verify CORS is configured for frontend."""
        main_py = project_root / "main.py"
        content = main_py.read_text()
        assert "CORSMiddleware" in content, "CORS middleware must be configured"
        assert "localhost:3000" in content, "CORS must allow frontend origin"
