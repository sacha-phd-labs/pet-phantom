from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

# Read long description from README.md if present
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setup(
    name="pet-phantom",
    version="0.1.0",
    description="Pet Phantom — a small utility/project",
    long_description=README,
    long_description_content_type="text/markdown",
    author="sacha.bouchez-delotte",
    author_email="",
    url="",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=[
        "albumentations==2.0.6",
        "numpy",
        "toolbox @ git+https://github.com/sacha-phd-labs/toolbox.git"
    ],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)