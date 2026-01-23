from pathlib import Path

from setuptools import setup


root = Path(__file__).parent
readme = (root / "README.md").read_text(encoding="utf-8")
requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
requirements = [line.strip() for line in requirements if line.strip() and not line.startswith("#")]


setup(
    name="fastslam64",
    version="0.0.0",
    description="FastSLAM 1.0 GPU implementation",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_dir={"fastslam64": "."},
    packages=[
        "fastslam64",
        "fastslam64.cuda",
        "fastslam64.lib",
        "fastslam64.simulation",
    ],
    package_data={
        "fastslam64": [
            "cuda/*.cu",
            "fsonline/*.json",
            "fsonline/*.npy",
            "simulation/*.npy",
            "*.json",
            "*.md",
            "*.npy",
            "*.txt",
        ]
    },
    install_requires=requirements,
    python_requires=">=3.8",
)

