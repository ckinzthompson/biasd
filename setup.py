from setuptools import setup
import re
import io
import os
import subprocess
from setuptools.command.build_py import build_py

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

META_PATH = os.path.join("biasd", "__init__.py")
META_FILE = read(META_PATH)

def find_meta(meta):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.MULTILINE
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError(f"Unable to find __{meta}__ string.")

class CustomBuildPyCommand(build_py):
    def run(self):
        print("Running CustomBuildPyCommand")
        # Change the working directory and run the make command
        make_dir = os.path.join(os.getcwd(), 'biasd/likelihood')
        print(f"Changing directory to: {make_dir}")
        subprocess.check_call(['make'], cwd=make_dir)
        # Proceed with the normal build_py
        super().run()

if __name__ == "__main__":
    setup(
        version=find_meta("version"),
        description=find_meta("description"),
        license=find_meta("license"),
        url=find_meta("url"),
        author=find_meta("author"),
        cmdclass={
            'build_py': CustomBuildPyCommand,
        },
    )