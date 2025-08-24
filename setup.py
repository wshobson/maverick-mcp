#!/usr/bin/env python

from setuptools import find_packages, setup

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the project metadata from pyproject.toml (for dependencies)
# This is a simple implementation; in a production setup you might want to use tomli
dependencies = []
with open("pyproject.toml") as f:
    content = f.read()
    # Find the dependencies section
    if "dependencies = [" in content:
        dependencies_section = content.split("dependencies = [")[1].split("]")[0]
        # Extract each dependency
        for line in dependencies_section.strip().split("\n"):
            dep = line.strip().strip(",").strip('"').strip("'")
            if dep and not dep.startswith("#"):
                dependencies.append(dep)

setup(
    name="maverick_mcp",
    version="0.1.0",
    description="Maverick-MCP is a Python MCP server for financial market analysis and trading strategies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MaverickMCP Contributors",
    author_email="",
    url="https://github.com/wshobson/maverick-mcp",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=dependencies,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    # No console scripts needed as we're running the API server directly
)
