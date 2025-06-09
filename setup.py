from setuptools import setup, find_packages
from pathlib import Path

# read the long description from README.md
here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="marlax",  # lowercase is recommended on PyPI
    version="0.1.0",
    author="Rudramani Singha",
    author_email="rgs2151@columbia.edu",
    description="Minimal multi-agent reinforcement learning library powered by JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["analysis", "analysis.*", "tests", "docs"]),
    python_requires=">=3.12",
    install_requires=[
        "tqdm>=4.65.0",   # progress bars used by Engine
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx-book-theme",
            "sphinx-autodoc-typehints",
            "myst-nb",
            "sphinx-math-dollar",
        ],
        "dev": [
            "pytest",
            "flake8",
            "black",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    entry_points={
        # you can add console_scripts here if you expose any CLI commands,
        # e.g. "marlax-train = marlax.runner:main"
    },
)
