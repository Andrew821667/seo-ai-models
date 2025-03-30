from setuptools import setup, find_packages

setup(
    name="seo_ai_models",
    version="0.2.0",
    description="SEO AI models for content analysis and optimization",
    author="AI Assistant",
    packages=find_packages(),
    package_data={
        'seo_ai_models': ['data/models/eeat/*.joblib'],
    },
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "razdel>=0.5.0",
        "pymystem3>=0.2.0",
        "joblib>=1.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.15.0",
        "tqdm>=4.65.0",
    ],
    python_requires='>=3.8',
)
