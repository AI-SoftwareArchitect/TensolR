from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt if it exists
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "numpy>=1.21.0",
        "numba>=0.56.0",
    ]

setup(
    name="tensolr",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A TensorFlow-like tensor framework with automatic differentiation and MLIR integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tensolr",
    packages=find_packages(where="src", include=["tensolr", "tensolr.*"]),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
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
            "pytest>=6.0",
            "pytest-cov",
        ],
        "gpu": [
            "cupy>=10.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # If you have any CLI tools, add them here
            # "tensolr=tensolr.cli:main",
        ],
    },
)