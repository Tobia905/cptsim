from setuptools import find_packages, setup
from codecs import open
from typing import List
import os
import re

curfile = os.path.abspath(os.path.dirname(__file__))  

PNAME, PTITLE = "cptsim", "cptsim"

def get_requirements(path: str) -> List[str]:
    requirements = []
    with open(path) as req:
        for line in req:
            line = re.sub(r"ÿþ|\x00", "", line).replace("\n", "")
            line = os.path.expandvars(line)
            requirements.append(line)

    return list(filter(len, requirements))


if __name__ == "__main__":
    setup(
        name=PNAME,
        description="A simulation study on the introduction of a progressive consumption tax.",
        version="0.0.1",
        author="Tobia Tommasini, Lorenzo Mondaini",
        url="https://github.com/Tobia905/cptsim",
        long_description_content_type="text/markdown",
        author_email="tobiatommasini@gmail.it",
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python 3.11"
        ],
        packages=[PNAME] + [f"{PNAME}.{p}" for p in find_packages(PNAME)],
        package_dir={PTITLE: PNAME},
        py_modules=["settings"],
        include_package_data=True,
        package_data={},
        install_requires=get_requirements("requirements.txt"),
        python_requires=">=3"
    )
