from setuptools import setup, find_packages

setup(
    name="research_template_legacy",
    version="0.1.0",
    package_dir={"": "src"},
    description="A common utility library for reproducible research projects.",
    packages=find_packages(),
)
