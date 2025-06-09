from setuptools import setup, find_packages

setup(
    name="Marlax",
    version="0.1.0",
    author="Rudramani Singha",
    author_email="rgs2151@columbia.edu",
    description="Code for Naturalistic Movie Analysis",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["analysis", "analysis.*"]),
    python_requires=">=3.12",
    install_requires=[],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={},
)