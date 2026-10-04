"""
tests/test_dev_docker.py

Unit tests for omnibioai-dev-docker.

Validates:
  1. Dockerfile structure and required instructions
  2. Required tools and packages are installed
  3. Exposed ports are correct
  4. requirements.txt is valid and complete
  5. run_ai_dev.sh script correctness
  6. Environment variables and labels
  7. Security and best practices

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import re
import runpy
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
REQUIREMENTS_DGX = REPO_ROOT / "requirements.dgx.txt"
RUN_SCRIPT = REPO_ROOT / "run_ai_dev.sh"

# ── Expected configuration ────────────────────────────────────────────────────
EXPECTED_BASE_IMAGE = "nvcr.io/nvidia/pytorch"
EXPECTED_PORTS = {8888, 11434}          # MySQL removed from image
EXPECTED_WORKDIR = "/workspace"
EXPECTED_IMAGE_NAME = "ghcr.io/omnibioai/omnibioai-dev-env"

# System packages that must be installed
REQUIRED_APT_PACKAGES = {
    "git", "curl", "wget", "build-essential",
    "libssl-dev", "libffi-dev", "pkg-config",
    "r-base", "r-base-dev",
    "fastqc", "samtools", "bcftools",
    "bedtools", "tabix",
    "openjdk-17-jre-headless",          # upgraded from 11 → 17 for GATK 4.5
}

# Python packages that must be in requirements.txt
REQUIRED_PIP_PACKAGES = {
    "jupyterlab", "notebook",
    "pandas", "numpy", "scipy", "scikit-learn",
    "matplotlib", "seaborn",
    "sqlalchemy", "pymysql",
    "xgboost", "lightgbm", "plotly", "bokeh", "polars",
    "transformers", "datasets", "accelerate",
    "huggingface-hub", "safetensors",
    "tensorboard",
    "nvidia-ml-py",
}

# Bioinformatics tools installed via curl/wget
REQUIRED_TOOLS = {"nextflow", "gatk", "snpeff"}

# R packages that must be installed
REQUIRED_R_PACKAGES = {
    "tidyverse", "data.table", "BiocManager",
    "ComplexHeatmap", "limma", "edgeR", "DESeq2",
    "SingleCellExperiment", "scran", "scater",  # single-cell additions
}

# Docker labels that must be present
REQUIRED_LABELS = {
    "org.opencontainers.image.authors",
    "org.opencontainers.image.description",
    "org.opencontainers.image.version",
    "org.opencontainers.image.source",
    "org.opencontainers.image.licenses",
    "omnibioai.type",
    "omnibioai.requires-gpu",
}

# run_ai_dev.sh must contain these docker run flags
REQUIRED_DOCKER_FLAGS = {
    "--gpus all",
    "--ipc=host",
    "--ulimit memlock=-1",
    "--ulimit stack=67108864",
}

# Environment variables that must be set in Dockerfile
REQUIRED_ENV_VARS = {
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "HF_HOME",
    "OLLAMA_HOST",
}

# Ports that must be published in run script
REQUIRED_PUBLISHED_PORTS = {8888, 11434}


# ── Helpers ───────────────────────────────────────────────────────────────────

def dockerfile_content() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def requirements_content() -> str:
    return REQUIREMENTS.read_text(encoding="utf-8")


def requirements_dgx_content() -> str:
    return REQUIREMENTS_DGX.read_text(encoding="utf-8")


def run_script_content() -> str:
    return RUN_SCRIPT.read_text(encoding="utf-8")


def dockerfile_lines() -> list[str]:
    return dockerfile_content().splitlines()


def requirements_packages() -> set[str]:
    """Parse requirements.txt and return lowercase package names."""
    pkgs = set()
    for line in requirements_content().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[>=<!@\s]", line)[0].strip().lower().replace("-", "_")
        if name:
            pkgs.add(name)
    return pkgs


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFilePresence(unittest.TestCase):
    """All required files must exist."""

    def test_dockerfile_exists(self) -> None:
        """The Dockerfile exists at the repository root."""
        self.assertTrue(DOCKERFILE.exists(), "Dockerfile not found")

    def test_requirements_exists(self) -> None:
        """requirements.txt exists at the repository root."""
        self.assertTrue(REQUIREMENTS.exists(), "requirements.txt not found")

    def test_requirements_dgx_exists(self) -> None:
        """requirements.dgx.txt exists at the repository root."""
        self.assertTrue(REQUIREMENTS_DGX.exists(), "requirements.dgx.txt not found")

    def test_run_script_exists(self) -> None:
        """run_ai_dev.sh exists at the repository root."""
        self.assertTrue(RUN_SCRIPT.exists(), "run_ai_dev.sh not found")

    def test_readme_exists(self) -> None:
        """README.md exists at the repository root."""
        self.assertTrue((REPO_ROOT / "README.md").exists(), "README.md not found")

    def test_dockerignore_exists(self) -> None:
        """.dockerignore exists at the repository root."""
        self.assertTrue((REPO_ROOT / ".dockerignore").exists(), ".dockerignore not found")

    def test_gitignore_exists(self) -> None:
        """.gitignore exists at the repository root."""
        self.assertTrue((REPO_ROOT / ".gitignore").exists(), ".gitignore not found")

    def test_pyproject_exists(self) -> None:
        """pyproject.toml exists at the repository root."""
        self.assertTrue((REPO_ROOT / "pyproject.toml").exists(), "pyproject.toml not found")

    def test_dockerfile_not_empty(self) -> None:
        """The Dockerfile has a non-zero size."""
        self.assertGreater(DOCKERFILE.stat().st_size, 0, "Dockerfile is empty")

    def test_requirements_not_empty(self) -> None:
        """requirements.txt has a non-zero size."""
        self.assertGreater(REQUIREMENTS.stat().st_size, 0, "requirements.txt is empty")

    def test_run_script_not_empty(self) -> None:
        """run_ai_dev.sh has a non-zero size."""
        self.assertGreater(RUN_SCRIPT.stat().st_size, 0, "run_ai_dev.sh is empty")


class TestDockerfileBaseImage(unittest.TestCase):
    """Dockerfile must use the correct NVIDIA base image."""

    def test_from_instruction_present(self) -> None:
        """The Dockerfile text contains a FROM instruction."""
        self.assertIn("FROM ", dockerfile_content(), "No FROM instruction found")

    def test_base_image_is_nvidia_pytorch(self) -> None:
        """The first FROM line uses the nvcr.io/nvidia/pytorch base image."""
        from_lines = [l for l in dockerfile_lines() if l.startswith("FROM ")]
        self.assertTrue(len(from_lines) > 0, "No FROM instruction")
        self.assertIn(EXPECTED_BASE_IMAGE, from_lines[0],
                      f"Base image should be {EXPECTED_BASE_IMAGE}")

    def test_base_image_has_cuda_tag(self) -> None:
        """The first FROM line's image tag includes the py3 marker; a CUDA marker is not
        asserted."""
        from_lines = [l for l in dockerfile_lines() if l.startswith("FROM ")]
        self.assertIn("py3", from_lines[0], "Base image tag should include 'py3'")

    def test_single_from_instruction(self) -> None:
        """The Dockerfile has exactly one FROM instruction, so it is not a multi-stage
        build."""
        from_lines = [l for l in dockerfile_lines() if l.startswith("FROM ")]
        self.assertEqual(len(from_lines), 1,
                         "Should have exactly one FROM — not a multi-stage build")


class TestDockerfileLabels(unittest.TestCase):
    """Required LABEL instructions must be present."""

    def test_required_labels_present(self) -> None:
        """Every label named in REQUIRED_LABELS (the OCI image labels plus
        omnibioai.type and omnibioai.requires-gpu) appears in the Dockerfile text."""
        content = dockerfile_content()
        errors = [f"Missing LABEL: {l}" for l in REQUIRED_LABELS if l not in content]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_maintainer_label_not_empty(self) -> None:
        """The org.opencontainers.image.authors label is present with a non-blank value."""
        content = dockerfile_content()
        match = re.search(r'LABEL org\.opencontainers\.image\.authors="([^"]+)"', content)
        self.assertIsNotNone(match, "authors label not found or empty")
        self.assertTrue(match.group(1).strip(), "authors label is empty")

    def test_version_label_follows_semver(self) -> None:
        """The org.opencontainers.image.version label matches MAJOR.MINOR.PATCH."""
        content = dockerfile_content()
        match = re.search(r'LABEL org\.opencontainers\.image\.version="([^"]+)"', content)
        self.assertIsNotNone(match, "version label not found")
        self.assertRegex(match.group(1), r"^\d+\.\d+\.\d+$",
                         f"version '{match.group(1)}' does not follow semver")

    def test_source_label_points_to_github(self) -> None:
        """The Dockerfile text references the authoritative OmniBioAI GitHub location; the check covers the whole
        file, not only the source label."""
        content = dockerfile_content()
        self.assertIn("github.com/OmniBioAI", content, "source label should point to GitHub")

    def test_license_label_is_apache(self) -> None:
        """The Dockerfile text contains Apache-2.0."""
        content = dockerfile_content()
        self.assertIn("Apache-2.0", content, "license label should be Apache-2.0")

    def test_omnibioai_type_label_is_dev_environment(self) -> None:
        """The Dockerfile sets the omnibioai.type label to dev-environment."""
        content = dockerfile_content()
        self.assertIn('omnibioai.type="dev-environment"', content,
                      "omnibioai.type should be dev-environment")

    def test_requires_gpu_label_is_true(self) -> None:
        """The Dockerfile sets the omnibioai.requires-gpu label to true."""
        content = dockerfile_content()
        self.assertIn('omnibioai.requires-gpu="true"', content,
                      "omnibioai.requires-gpu should be true")


class TestDockerfileWorkdir(unittest.TestCase):
    """WORKDIR must be set correctly."""

    def test_workdir_is_set(self) -> None:
        """The Dockerfile has a WORKDIR instruction."""
        self.assertIn("WORKDIR", dockerfile_content(), "No WORKDIR instruction")

    def test_workdir_is_workspace(self) -> None:
        """The Dockerfile sets WORKDIR to /workspace."""
        self.assertIn(f"WORKDIR {EXPECTED_WORKDIR}", dockerfile_content(),
                      f"WORKDIR should be {EXPECTED_WORKDIR}")


class TestDockerfilePorts(unittest.TestCase):
    """Required ports must be exposed — MySQL removed."""

    def _exposed_ports(self) -> set[int]:
        """Collects the integer ports from the Dockerfile's EXPOSE lines, silently
        skipping tokens that are not integers."""
        ports = set()
        for line in dockerfile_lines():
            if line.startswith("EXPOSE "):
                for port in line.replace("EXPOSE", "").split():
                    try:
                        ports.add(int(port.strip()))
                    except ValueError:
                        pass
        return ports

    def test_all_required_ports_exposed(self) -> None:
        """Both required ports, 8888 (Jupyter) and 11434 (Ollama), are exposed."""
        exposed = self._exposed_ports()
        for port in EXPECTED_PORTS:
            self.assertIn(port, exposed, f"Port {port} not exposed")

    def test_mysql_port_not_exposed(self) -> None:
        """Port 3306 is not exposed, since MySQL was removed from the image."""
        self.assertNotIn(3306, self._exposed_ports(),
                         "Port 3306 should not be exposed — MySQL removed from image")

    def test_jupyter_port_exposed(self) -> None:
        """Port 8888 (Jupyter) is exposed."""
        self.assertIn(8888, self._exposed_ports())

    def test_ollama_port_exposed(self) -> None:
        """Port 11434 (Ollama) is exposed."""
        self.assertIn(11434, self._exposed_ports())


class TestDockerfileAptPackages(unittest.TestCase):
    """Required apt packages must be in RUN apt-get install."""

    def test_required_apt_packages_installed(self) -> None:
        """Every package in REQUIRED_APT_PACKAGES appears in the Dockerfile text."""
        content = dockerfile_content()
        errors = [f"apt package not found: {p}"
                  for p in REQUIRED_APT_PACKAGES if p not in content]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_apt_cache_cleaned(self) -> None:
        """The Dockerfile text contains an rm -rf /var/lib/apt/lists cleanup."""
        self.assertIn("rm -rf /var/lib/apt/lists", dockerfile_content(),
                      "apt cache not cleaned")

    def test_apt_install_uses_no_install_recommends(self) -> None:
        """The Dockerfile text contains --no-install-recommends."""
        self.assertIn("--no-install-recommends", dockerfile_content(),
                      "apt-get install should use --no-install-recommends")

    def test_java_17_not_java_11(self) -> None:
        """The Dockerfile references openjdk-17 (needed for GATK 4.5) and does not
        reference openjdk-11."""
        content = dockerfile_content()
        self.assertIn("openjdk-17", content, "Should use Java 17 for GATK 4.5")
        self.assertNotIn("openjdk-11", content, "Java 11 should be replaced with Java 17")

    def test_mysql_server_not_installed(self) -> None:
        """The Dockerfile does not mention mysql-server, so MySQL stays in an external
        container."""
        self.assertNotIn("mysql-server", dockerfile_content(),
                         "mysql-server should not be in image — use external container")

    def test_bedtools_installed(self) -> None:
        """The Dockerfile text mentions bedtools."""
        self.assertIn("bedtools", dockerfile_content(), "bedtools not installed")

    def test_tabix_installed(self) -> None:
        """The Dockerfile text mentions tabix."""
        self.assertIn("tabix", dockerfile_content(), "tabix not installed")

    def test_r_base_dev_installed(self) -> None:
        """The Dockerfile text mentions r-base-dev, which R package compilation needs."""
        self.assertIn("r-base-dev", dockerfile_content(),
                      "r-base-dev needed for R package compilation")


class TestDockerfilePipPackages(unittest.TestCase):
    """Required Python packages must be in requirements.txt."""

    def test_dockerfile_references_requirements(self) -> None:
        """The Dockerfile text references requirements.txt."""
        self.assertIn("requirements.txt", dockerfile_content(),
                      "Dockerfile should install from requirements.txt")

    def test_required_pip_packages_installed(self) -> None:
        """Every package in REQUIRED_PIP_PACKAGES appears in requirements.txt,
        tolerating dash and underscore spelling variants."""
        req_content = requirements_content().lower()
        errors = []
        for pkg in REQUIRED_PIP_PACKAGES:
            pkg_norm = pkg.lower().replace("-", "_").replace("_", "-")
            pkg_alt = pkg.lower().replace("-", "_")
            if (pkg_norm not in req_content
                    and pkg_alt not in req_content
                    and pkg.lower() not in req_content):
                errors.append(f"pip package not found in requirements.txt: {pkg}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_pip_upgraded_before_install(self) -> None:
        """The Dockerfile text contains --upgrade pip; only its presence is asserted,
        not its order relative to installs."""
        content = dockerfile_content()
        self.assertTrue(
            "--upgrade pip" in content,
            "pip should be upgraded before installing packages",
        )

    def test_no_dgx_file_installs_in_requirements(self) -> None:
        """requirements.txt has no lines installing from file:///opt/, file:///tmp/ or
        file:///workspace/; DGX-specific installs belong in requirements.dgx.txt."""
        content = requirements_content()
        dgx_installs = [l for l in content.splitlines()
                        if "file:///opt/" in l or "file:///tmp/" in l
                        or "file:///workspace/" in l]
        self.assertEqual(dgx_installs, [],
                         f"DGX-specific file:// installs in requirements.txt: {dgx_installs}\n"
                         "Move these to requirements.dgx.txt")

    def test_requirements_dgx_documents_local_installs(self) -> None:
        """requirements.dgx.txt mentions torch (case-insensitive) and contains the
        upper-case string DGX."""
        content = requirements_dgx_content()
        self.assertIn("torch", content.lower(),
                      "requirements.dgx.txt should document torch DGX install")
        self.assertIn("DGX", content or "dgx" in content.lower(),
                      "requirements.dgx.txt should mention DGX")


class TestDockerfileBioinformaticsTools(unittest.TestCase):
    """Bioinformatics tools must be installed."""

    def test_nextflow_installed(self) -> None:
        """The Dockerfile text mentions nextflow (case-insensitive)."""
        self.assertIn("nextflow", dockerfile_content().lower(), "Nextflow not installed")

    def test_nextflow_version_verified(self) -> None:
        """The Dockerfile text contains nextflow -version, a post-install version check."""
        self.assertIn("nextflow -version", dockerfile_content(),
                      "Nextflow version should be verified after install")

    def test_gatk_installed(self) -> None:
        """The Dockerfile text mentions gatk (case-insensitive)."""
        self.assertIn("gatk", dockerfile_content().lower(), "GATK not installed")

    def test_gatk_version_verified(self) -> None:
        """The Dockerfile text contains gatk --version, a post-install version check."""
        self.assertIn("gatk --version", dockerfile_content(),
                      "GATK version should be verified after install")

    def test_fastqc_installed(self) -> None:
        """The Dockerfile text mentions fastqc (case-insensitive)."""
        self.assertIn("fastqc", dockerfile_content().lower(), "FastQC not installed")

    def test_samtools_installed(self) -> None:
        """The Dockerfile text mentions samtools."""
        self.assertIn("samtools", dockerfile_content(), "SAMtools not installed")

    def test_bcftools_installed(self) -> None:
        """The Dockerfile text mentions bcftools."""
        self.assertIn("bcftools", dockerfile_content(), "BCFtools not installed")

    def test_snpeff_installed(self) -> None:
        """The Dockerfile text mentions snpEff (case-sensitive)."""
        self.assertIn("snpEff", dockerfile_content(), "SnpEff not installed")

    def test_ollama_installed(self) -> None:
        """The Dockerfile text mentions ollama (case-insensitive)."""
        self.assertIn("ollama", dockerfile_content().lower(), "Ollama not installed")

    def test_java_installed(self) -> None:
        """The Dockerfile text mentions openjdk."""
        self.assertIn("openjdk", dockerfile_content(), "Java (OpenJDK) not installed")

    def test_validation_step_present(self) -> None:
        """The Dockerfile text contains the 'All tools validated' build-time validation
        message."""
        content = dockerfile_content()
        self.assertIn("All tools validated", content,
                      "Build-time validation step missing")


class TestDockerfileRPackages(unittest.TestCase):
    """R and Bioconductor packages must be installed."""

    def test_r_base_installed(self) -> None:
        """The Dockerfile text mentions r-base."""
        self.assertIn("r-base", dockerfile_content(), "r-base not installed")

    def test_required_r_packages_present(self) -> None:
        """Every package in REQUIRED_R_PACKAGES appears in the Dockerfile text."""
        content = dockerfile_content()
        errors = [f"R package not found: {p}"
                  for p in REQUIRED_R_PACKAGES if p not in content]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_biocmanager_used(self) -> None:
        """The Dockerfile text mentions BiocManager, used for Bioconductor packages."""
        self.assertIn("BiocManager", dockerfile_content(),
                      "BiocManager not used for Bioconductor packages")

    def test_single_cell_r_packages_installed(self) -> None:
        """SingleCellExperiment, scran and scater each appear in the Dockerfile text."""
        content = dockerfile_content()
        for pkg in {"SingleCellExperiment", "scran", "scater"}:
            self.assertIn(pkg, content, f"Single-cell R package not installed: {pkg}")


class TestDockerfileJupyter(unittest.TestCase):
    """Jupyter must be configured for remote access."""

    def test_jupyter_config_created(self) -> None:
        """The Dockerfile text references jupyter_notebook_config.py."""
        self.assertIn("jupyter_notebook_config.py", dockerfile_content())

    def test_jupyter_configured_for_all_interfaces(self) -> None:
        """The Dockerfile text contains 0.0.0.0, i.e. Jupyter is set to listen on all
        interfaces; the check covers the whole file."""
        self.assertIn("0.0.0.0", dockerfile_content(),
                      "Jupyter not configured to listen on all interfaces")

    def test_jupyter_browser_disabled(self) -> None:
        """The Dockerfile text contains open_browser = False."""
        self.assertIn("open_browser = False", dockerfile_content(),
                      "Jupyter browser auto-launch not disabled")

    def test_jupyter_allow_root(self) -> None:
        """The Dockerfile text contains allow_root = True."""
        self.assertIn("allow_root = True", dockerfile_content(),
                      "Jupyter root access not enabled")

    def test_jupyter_server_app_configured(self) -> None:
        """The Dockerfile text contains ServerApp, the JupyterLab 4.x configuration
        namespace."""
        self.assertIn("ServerApp", dockerfile_content(),
                      "JupyterLab 4.x ServerApp config missing")


class TestDockerfileEnvVars(unittest.TestCase):
    """Required environment variables must be set in Dockerfile."""

    def test_required_env_vars_present(self) -> None:
        """Every name in REQUIRED_ENV_VARS (PYTHONUNBUFFERED, PYTHONDONTWRITEBYTECODE,
        HF_HOME, OLLAMA_HOST) appears in the Dockerfile text."""
        content = dockerfile_content()
        errors = [f"ENV var not set: {v}"
                  for v in REQUIRED_ENV_VARS if v not in content]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_path_env_set(self) -> None:
        """The Dockerfile text contains an ENV PATH= instruction."""
        self.assertIn("ENV PATH=", dockerfile_content(),
                      "PATH environment variable not set")

    def test_gatk_in_path(self) -> None:
        """The Dockerfile text mentions gatk (case-insensitive); PATH membership itself
        is not asserted."""
        self.assertIn("gatk", dockerfile_content().lower())

    def test_snpeff_in_path(self) -> None:
        """The Dockerfile text mentions snpEff; PATH membership itself is not asserted."""
        self.assertIn("snpEff", dockerfile_content())

    def test_pythonunbuffered_set(self) -> None:
        """The Dockerfile text contains PYTHONUNBUFFERED=1."""
        self.assertIn("PYTHONUNBUFFERED=1", dockerfile_content())

    def test_hf_home_set_in_dockerfile(self) -> None:
        """The Dockerfile text contains an HF_HOME= assignment."""
        self.assertIn("HF_HOME=", dockerfile_content(),
                      "HF_HOME should be set as ENV in Dockerfile")

    def test_ollama_host_set_in_dockerfile(self) -> None:
        """The Dockerfile text contains an OLLAMA_HOST= assignment."""
        self.assertIn("OLLAMA_HOST=", dockerfile_content(),
                      "OLLAMA_HOST should be set as ENV in Dockerfile")


class TestDockerfileCmd(unittest.TestCase):
    """CMD instruction must be present."""

    def test_cmd_instruction_present(self) -> None:
        """The Dockerfile text contains a CMD instruction."""
        self.assertIn("CMD ", dockerfile_content(), "No CMD instruction in Dockerfile")

    def test_cmd_is_bash(self) -> None:
        """The Dockerfile's default command is CMD [bash] in exec form."""
        self.assertIn('CMD ["bash"]', dockerfile_content(),
                      "CMD should default to bash")


class TestRequirementsTxt(unittest.TestCase):
    """requirements.txt must be valid and contain key packages."""

    REQUIRED_IN_REQUIREMENTS = {
        "transformers", "datasets", "accelerate",  # torch is DGX-only → requirements.dgx.txt
        "huggingface_hub", "safetensors",
        "jupyterlab", "notebook",
        "pandas", "numpy", "scipy", "scikit_learn",
        "matplotlib", "seaborn",
        "xgboost", "lightgbm", "plotly", "bokeh", "polars",
        "sqlalchemy", "pymysql",
        "tensorboard",
        "pytest",
    }

    def test_requirements_parseable(self) -> None:
        """Parsing requirements.txt yields at least one package name."""
        self.assertGreater(len(requirements_packages()), 0,
                           "requirements.txt has no packages")

    def test_required_packages_in_requirements(self) -> None:
        """Every name in REQUIRED_IN_REQUIREMENTS is among the parsed requirements.txt
        packages, compared lower-case with underscores."""
        pkgs = requirements_packages()
        errors = [f"Missing from requirements.txt: {p}"
                  for p in self.REQUIRED_IN_REQUIREMENTS if p.lower() not in pkgs]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_no_duplicate_packages(self) -> None:
        """requirements.txt lists no package name more than once, ignoring comment and
        blank lines."""
        names = []
        for line in REQUIREMENTS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[>=<!@\s]", line)[0].strip().lower()
            if name:
                names.append(name)
        duplicates = [n for n in set(names) if names.count(n) > 1]
        self.assertEqual(duplicates, [], f"Duplicate packages: {duplicates}")

    def test_pytorch_in_requirements(self) -> None:
        """requirements.dgx.txt mentions torch, so the DGX-only PyTorch install is
        documented there rather than in requirements.txt."""
        # torch is DGX-specific — installed from file:// in nvcr.io base
        # it must be documented in requirements.dgx.txt, not requirements.txt
        self.assertIn("torch", requirements_dgx_content().lower(),
                      "PyTorch should be documented in requirements.dgx.txt")

    def test_jupyter_in_requirements(self) -> None:
        """requirements.txt mentions jupyterlab."""
        self.assertIn("jupyterlab", requirements_content().lower())

    def test_no_local_file_installs(self) -> None:
        """requirements.txt has no uncommented file:/// install lines."""
        bad = [l for l in requirements_content().splitlines()
               if "file:///" in l and not l.strip().startswith("#")]
        self.assertEqual(bad, [],
                         f"Local file:// installs found — move to requirements.dgx.txt:\n"
                         + "\n".join(bad))


class TestRunScript(unittest.TestCase):
    """run_ai_dev.sh must be correct and safe."""

    def test_shebang_present(self) -> None:
        """run_ai_dev.sh starts with a bash shebang, either #!/bin/bash or
        #!/usr/bin/env bash."""
        content = run_script_content()
        self.assertTrue(
            content.startswith("#!/bin/bash")
            or content.startswith("#!/usr/bin/env bash"),
            "run_ai_dev.sh missing bash shebang",
        )

    def test_image_name_is_ghcr(self) -> None:
        """The script text references the authoritative GHCR image name."""
        content = run_script_content()
        self.assertIn(EXPECTED_IMAGE_NAME, content,
                      f"IMAGE_NAME should be {EXPECTED_IMAGE_NAME}")

    def test_gpu_flag_present(self) -> None:
        """The script text contains --gpus all."""
        self.assertIn("--gpus all", run_script_content(),
                      "GPU flag --gpus all not present")

    def test_ipc_host_flag_present(self) -> None:
        """The script text contains --ipc=host, which PyTorch DataLoader workers need."""
        self.assertIn("--ipc=host", run_script_content(),
                      "--ipc=host required for PyTorch DataLoader")

    def test_memlock_ulimit_present(self) -> None:
        """The script text contains --ulimit memlock=-1."""
        self.assertIn("--ulimit memlock=-1", run_script_content(),
                      "memlock ulimit not set")

    def test_stack_ulimit_present(self) -> None:
        """The script text contains --ulimit stack=67108864."""
        self.assertIn("--ulimit stack=67108864", run_script_content(),
                      "stack ulimit not set")

    def test_jupyter_port_published(self) -> None:
        """The script text configures the Jupyter port: a -p 8888:8888 mapping, a
        JUPYTER_PORT mapping, or any mention of JUPYTER_PORT."""
        content = run_script_content()
        self.assertTrue(
            "-p 8888:8888" in content
            or "-p ${JUPYTER_PORT}:${JUPYTER_PORT}" in content
            or "JUPYTER_PORT" in content,
            "Jupyter port 8888 not configured",
        )

    def test_ollama_port_published(self) -> None:
        """The script text configures the Ollama port: a -p 11434:11434 mapping, an
        OLLAMA_PORT mapping, or any mention of OLLAMA_PORT."""
        content = run_script_content()
        self.assertTrue(
            "-p 11434:11434" in content
            or "-p ${OLLAMA_PORT}:${OLLAMA_PORT}" in content
            or "OLLAMA_PORT" in content,
            "Ollama port 11434 not configured",
        )

    def test_mysql_port_not_published(self) -> None:
        """The script text does not contain -p 3306, so MySQL is not published."""
        self.assertNotIn("-p 3306", run_script_content(),
                         "MySQL port should not be published — removed from image")

    def test_huggingface_cache_mounted(self) -> None:
        """The script text references .cache/huggingface, i.e. the HuggingFace cache is
        mounted."""
        self.assertIn(".cache/huggingface", run_script_content(),
                      "HuggingFace cache not mounted")

    def test_ollama_model_dir_mounted(self) -> None:
        """The script text references .ollama, i.e. the Ollama model directory is
        mounted."""
        self.assertIn(".ollama", run_script_content(),
                      "Ollama model directory not mounted")

    def test_workspace_mounted(self) -> None:
        """The script text references /workspace."""
        self.assertIn("/workspace", run_script_content(),
                      "Workspace not mounted")

    def test_gitconfig_mounted(self) -> None:
        """The script text references .gitconfig so git works inside the container."""
        self.assertIn(".gitconfig", run_script_content(),
                      ".gitconfig should be mounted so git works inside container")

    def test_hf_home_env_set(self) -> None:
        """The script text contains an HF_HOME= assignment."""
        self.assertIn("HF_HOME=", run_script_content(),
                      "HF_HOME env var not set")

    def test_ollama_host_env_set(self) -> None:
        """The script text contains an OLLAMA_HOST= assignment."""
        self.assertIn("OLLAMA_HOST=", run_script_content(),
                      "OLLAMA_HOST env var not set")

    def test_nvidia_smi_check_present(self) -> None:
        """The script text references nvidia-smi, its GPU availability check."""
        self.assertIn("nvidia-smi", run_script_content(),
                      "nvidia-smi check not present")

    def test_docker_daemon_check_present(self) -> None:
        """The script text contains docker info, its Docker daemon check."""
        self.assertIn("docker info", run_script_content(),
                      "Docker daemon check not present")

    def test_container_cleanup_before_run(self) -> None:
        """The script text contains docker rm -f, used to clean up an old container
        before running."""
        self.assertIn("docker rm -f", run_script_content(),
                      "Old container not cleaned up before run")

    def test_interactive_flag_present(self) -> None:
        """The script text contains -it or -i, so the container can start interactively."""
        content = run_script_content()
        self.assertTrue("-it " in content or "-i " in content,
                        "Container not started in interactive mode")

    def test_jupyter_flag_supported(self) -> None:
        """The script text supports a --jupyter option."""
        self.assertIn("--jupyter", run_script_content(),
                      "--jupyter flag not supported")

    def test_ollama_flag_supported(self) -> None:
        """The script text supports an --ollama option."""
        self.assertIn("--ollama", run_script_content(),
                      "--ollama flag not supported")

    def test_build_flag_supported(self) -> None:
        """The script text supports a --build option."""
        self.assertIn("--build", run_script_content(),
                      "--build flag not supported")

    def test_help_flag_supported(self) -> None:
        """The script text supports a --help option."""
        self.assertIn("--help", run_script_content(),
                      "--help flag not supported")

    def test_pull_before_build(self) -> None:
        """The script text contains docker pull, so it can pull from GHCR before
        building; the order is not asserted."""
        self.assertIn("docker pull", run_script_content(),
                      "Script should try pulling from GHCR before building")


class TestDockerfileSecurity(unittest.TestCase):
    """Basic security and best practice checks."""

    def test_no_hardcoded_passwords(self) -> None:
        """The Dockerfile has no quoted PASSWORD or SECRET assignment (case-insensitive
        match)."""
        content = dockerfile_content()
        for pattern in [r'PASSWORD\s*=\s*["\'][^"\']+["\']',
                        r'SECRET\s*=\s*["\'][^"\']+["\']']:
            match = re.search(pattern, content, re.IGNORECASE)
            self.assertIsNone(match, f"Possible hardcoded secret: {match}")

    def test_no_hardcoded_api_keys(self) -> None:
        """The Dockerfile contains no hf_-prefixed run of 20 or more letters, i.e. no
        hardcoded HuggingFace API key pattern."""
        self.assertNotRegex(dockerfile_content(), r'hf_[A-Za-z]{20,}',
                            "HuggingFace API key appears hardcoded")

    def test_apt_update_before_install(self) -> None:
        """Every RUN block that uses apt-get install also contains apt-get update within
        the same block."""
        content = dockerfile_content()
        run_blocks = re.findall(
            r'RUN.*?(?=\nRUN|\nFROM|\nCMD|\nENTRYPOINT|\Z)',
            content, re.DOTALL,
        )
        errors = []
        for block in run_blocks:
            if "apt-get install" in block and "apt-get update" not in block:
                errors.append(f"apt-get install without update: {block.split(chr(10))[0][:80]}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_no_sudo_in_dockerfile(self) -> None:
        """The Dockerfile does not use sudo."""
        self.assertNotIn("sudo ", dockerfile_content(),
                         "sudo should not be used in Dockerfile")

    def test_no_ssh_keys_in_dockerfile(self) -> None:
        """The Dockerfile contains no ssh-rsa public key pattern."""
        content = dockerfile_content()
        self.assertNotRegex(content, r'ssh-rsa\s+[A-Za-z0-9+/]',
                            "SSH key appears hardcoded in Dockerfile")


class TestBranchCoverage(unittest.TestCase):
    """Exercises every error-path branch and helper edge case for 100% coverage."""

    @staticmethod
    def _mod():
        """Returns this test module so patch.object can replace the module-level helpers
        (dockerfile_content, requirements_content and so on) with stubs."""
        import sys
        return sys.modules[__name__]

    def test_requirements_packages_skips_empty_and_comment_lines(self) -> None:
        """requirements_packages ignores comment and blank lines in stubbed requirements
        content and still returns pandas."""
        with patch.object(self._mod(), "requirements_content",
                          return_value="# a comment\n\npandas==2.0\n"):
            pkgs = requirements_packages()
        self.assertIn("pandas", pkgs)

    def test_exposed_ports_ignores_non_integer_token(self) -> None:
        """_exposed_ports skips a non-numeric EXPOSE token and still returns 8888, with
        dockerfile_lines stubbed."""
        with patch.object(self._mod(), "dockerfile_lines",
                          return_value=["EXPOSE 8888", "EXPOSE bad_port"]):
            ports = TestDockerfilePorts()._exposed_ports()
        self.assertIn(8888, ports)

    def test_labels_failure_branch(self) -> None:
        """test_required_labels_present raises AssertionError when the stubbed
        Dockerfile is only FROM scratch with no labels."""
        with patch.object(self._mod(), "dockerfile_content",
                          return_value="FROM scratch"):
            with self.assertRaises(AssertionError):
                TestDockerfileLabels().test_required_labels_present()

    def test_apt_packages_failure_branch(self) -> None:
        """test_required_apt_packages_installed raises AssertionError when the stubbed
        Dockerfile installs only git."""
        stub = ("FROM scratch\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends git\n"
                " && rm -rf /var/lib/apt/lists/*")
        with patch.object(self._mod(), "dockerfile_content", return_value=stub):
            with self.assertRaises(AssertionError):
                TestDockerfileAptPackages().test_required_apt_packages_installed()

    def test_pip_packages_failure_branch(self) -> None:
        """test_required_pip_packages_installed raises AssertionError when the stubbed
        requirements.txt lists only an unrelated package."""
        df_stub = ("COPY requirements.txt /tmp/requirements.txt\n"
                   "RUN pip install -r /tmp/requirements.txt")
        req_stub = "some-other-package==1.0\n"
        with patch.object(self._mod(), "dockerfile_content", return_value=df_stub), \
             patch.object(self._mod(), "requirements_content", return_value=req_stub):
            with self.assertRaises(AssertionError):
                TestDockerfilePipPackages().test_required_pip_packages_installed()

    def test_r_packages_failure_branch(self) -> None:
        """test_required_r_packages_present raises AssertionError when the stubbed
        Dockerfile lacks the required R packages."""
        with patch.object(self._mod(), "dockerfile_content",
                          return_value="RUN apt-get install -y r-base\nBiocManager"):
            with self.assertRaises(AssertionError):
                TestDockerfileRPackages().test_required_r_packages_present()

    def test_requirements_packages_failure_branch(self) -> None:
        """test_required_packages_in_requirements raises AssertionError when the stubbed
        package set lacks the required packages."""
        with patch.object(self._mod(), "requirements_packages",
                          return_value={"some_other_pkg"}):
            with self.assertRaises(AssertionError):
                TestRequirementsTxt().test_required_packages_in_requirements()

    def test_no_duplicate_packages_skips_empty_lines(self) -> None:
        """test_no_duplicate_packages passes on mocked requirements text made of
        comments, blank lines and two distinct packages."""
        mock_req = MagicMock()
        mock_req.read_text.return_value = "# comment\n\npandas==2.0\nnumpy==1.0\n"
        with patch.object(self._mod(), "REQUIREMENTS", mock_req):
            TestRequirementsTxt().test_no_duplicate_packages()

    def test_apt_update_without_install_failure_branch(self) -> None:
        """test_apt_update_before_install raises AssertionError for a stubbed Dockerfile
        whose RUN block runs apt-get install without apt-get update."""
        bad_dockerfile = "RUN apt-get install -y curl\nRUN echo done"
        with patch.object(self._mod(), "dockerfile_content", return_value=bad_dockerfile):
            with self.assertRaises(AssertionError):
                TestDockerfileSecurity().test_apt_update_before_install()

    def test_mysql_port_not_exposed_failure_branch(self) -> None:
        """test_mysql_port_not_exposed raises AssertionError when the stubbed Dockerfile
        exposes port 3306."""
        with patch.object(self._mod(), "dockerfile_lines",
                          return_value=["EXPOSE 3306", "EXPOSE 8888", "EXPOSE 11434"]):
            with self.assertRaises(AssertionError):
                TestDockerfilePorts().test_mysql_port_not_exposed()

    def test_no_local_file_installs_failure_branch(self) -> None:
        """test_no_local_file_installs raises AssertionError for a stubbed requirements
        line that installs torch from file:///opt/."""
        with patch.object(self._mod(), "requirements_content",
                          return_value="torch @ file:///opt/pytorch/torch.whl\n"):
            with self.assertRaises(AssertionError):
                TestRequirementsTxt().test_no_local_file_installs()

    def test_main_guard_invokes_unittest_main(self) -> None:
        """Re-running this module as __main__ through runpy, with unittest.main patched,
        calls unittest.main exactly once."""
        with patch("unittest.main") as mock_main:
            runpy.run_path(__file__, run_name="__main__")
        mock_main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
