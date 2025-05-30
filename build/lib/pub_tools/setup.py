from setuptools import setup, find_packages

setup(
    name='pub_tools',
    version='0.1.0',
    packages=find_packages(),
    description='public tool module',
    author='Ziji Zhang',
    install_requires=[
        'requests>=2.25.1',
        'SQLAlchemy>=1.4.0',
        'pandas>=1.2.0',
        'optuna>=2.0.0',
        'matplotlib>=3.3.0',
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'pymysql>=0.9.3'
    ]
)