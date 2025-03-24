from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import sys

install_requires = []

test_requires = [
    "pytest",
    "pytest-cov",
]


class CustomCommand:
    """Common functionality for install and develop commands"""

    def run_setup_ops(self):
        try:
            subprocess.check_call([sys.executable, "setup_ops.py", "install"])
        except subprocess.CalledProcessError as e:
            print(f"Error running setup_ops.py: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


class CustomInstallCommand(install, CustomCommand):
    def run(self):
        self.run_setup_ops()
        install.run(self)


class CustomDevelopCommand(develop, CustomCommand):
    def run(self):
        self.run_setup_ops()
        develop.run(self)


setup(
    name="paddle_sparse",
    version="1.0",
    description="PaddlePaddle Sparse Tensor",
    author="AndPuQing",
    python_requires=">=3.8",
    setup_requires=install_requires,
    extras_require={
        "test": test_requires,
    },
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
    },
    packages=find_packages(),
)
