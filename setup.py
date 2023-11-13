import pathlib

from setuptools import setup, find_packages

setup(
    name='yarmmaxds',
    version='0.2.4',
    description='my test DS package',
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    # url=,
    author='yarmmax',
    # author_email=,
    license="The Unlicense",
    # project_urls={},
    # classifiers=[]
    python_requires='>=3.8, <4',
    install_requires=[
                    'numpy'
                    'pandas'
                    'setuptools'
                    'pathlib'
                    'seaborn'
                    'matplotlib'
                    'scipy'
                    'scikit-learn'
                    'datetime'
                    'scikit-learn'
                    'permetrics'
                    'lightgbm'
                    'xgboost'
                    'boruta'
                    'catboost'
                    'optuna'
                    'shap'
                    'hyperopt'
    ],
    # extras_require={ "excel":[openpyxl]},
    packages=find_packages(),
    include_package_data=True,
    # entry_points={"console_scripts": ["yarmmax_DS = yarmmaxDS.entry_point:main"]},
)