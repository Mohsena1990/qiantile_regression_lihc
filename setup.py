from setuptools import setup, find_packages

setup(
    name="LIHC_Project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "scikit-learn",
        "catboost",
        # add any other dependencies here
    ],
    python_requires=">=3.10",
)
