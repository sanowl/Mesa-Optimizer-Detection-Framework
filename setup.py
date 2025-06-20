from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mesa-optimizer-detection",
    version="0.1.0",
    author="Mesa Detection Team",
    author_email="contact@mesa-detection.org",
    description="A comprehensive framework for detecting mesa-optimization in large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Mesa-Optimizer-Detection-Framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "triton>=2.0.0",
        ],
        "distributed": [
            "ray>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mesa-detect=mesa_optimizer_detection.cli:main",
        ],
    },
) 