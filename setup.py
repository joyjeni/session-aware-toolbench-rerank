from setuptools import setup, find_packages

setup(
    name="sessionrerank",
    version="0.1.0",
    description="Session-aware re-ranking for ToolBench API retrieval",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy<2.1",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "pandas>=1.5",
        "pyarrow>=12",
        "sentence-transformers>=2.7",
        "rank-bm25",
        "transformers>=4.45",
        "huggingface_hub>=0.23",
        "accelerate>=0.30",
        "bitsandbytes>=0.43",
        "torch>=2.0",
    ],
)
