from setuptools import setup, find_packages

setup(
    name="seo_ai_models",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'redis>=4.5.0',
        'tqdm>=4.65.0',
        'pydantic>=2.0.0',
        'fastapi>=0.100.0',
    ],
)
