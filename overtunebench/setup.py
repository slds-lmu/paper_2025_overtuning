from setuptools import find_packages, setup

setup(
    name="overtunebench",
    version="0.0.9",
    author="Lennart Schneider",
    author_email="lennart.sch@web.de",
    packages=find_packages(),
    url="https://github.com/slds-lmu/paper_2025_overtuning",
    license="MIT License",
    description="Python package to benchmark and analyze overtuning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.25.2",
        "pandas==2.1.0",
        "scipy==1.11.2",
        "optuna==3.3.0",
        "scikit-learn==1.3.0",
        "torch==2.1.1",
        "tqdm==4.66.1",
        "setuptools==65.5.1",
        "pyarrow==13.0.0",
        "openml==0.14.1",
        "tabpfn==0.1.9",
        "xgboost==2.0.2",
        "catboost==1.2.1",
        "numdifftools==0.9.41",
        "ConfigSpace==1.2.0",
    ],
    extras_require={
        "HEBO": ["HEBO==0.3.5"],
        "smac": ["smac==2.2.0"],
    },
    python_requires=">=3.10",
    classifiers=[
        # Trove classifiers
        # Full list at https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
