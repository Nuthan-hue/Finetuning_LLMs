from setuptools import setup, find_packages

setup(
    name="kaggle-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "kaggle>=1.5.12",
        "scikit-learn>=0.24.2",
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "lightgbm>=3.3.2",
        "xgboost>=1.4.2",
        "beautifulsoup4>=4.9.3",
        "requests>=2.26.0",
        "pydantic>=1.8.2",
        "python-dotenv>=0.19.0",
        "aiohttp>=3.8.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "jupyter>=1.0.0"
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-agent system for automating Kaggle competition workflows",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kaggle-agent",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
