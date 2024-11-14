import io
import os
import re
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

def read(*parts):
	"""
	Build an absolute path from *parts* and and return the contents of the
	resulting file.  Assume UTF-8 encoding.
	"""
	HERE = os.path.abspath(os.path.dirname(__file__))
	with io.open(os.path.join(HERE, *parts), encoding="utf-8") as f:
		return f.read()

def find_meta(meta):
	"""
	Extract __*meta*__ from META_FILE.
	"""
	META_FILE = read(os.path.join("biasd","__init__.py"))
	meta_match = re.search(
		r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
		META_FILE, re.M
	)
	if meta_match:
		return meta_match.group(1)
	raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))

class CustomBuildPyCommand(build_py):
	def run(self):
		if sys.platform != 'win32':
			print("Running CustomBuildPyCommand")
			# Change the working directory and run the make command
			make_dir = os.path.join(os.getcwd(), 'biasd/likelihood')
			print(f"Changing directory to: {make_dir}")
			subprocess.check_call(['make'], cwd=make_dir)
			# Proceed with the normal build_py
			build_py.run(self)
		else:
			print('Skipping make b/c on windows')
			build_py.run(self)


setup(
	name='biasd',
	description=find_meta("description"),
	license=find_meta("license"),
	url=find_meta("url"),
	version=find_meta("version"),
	author=find_meta("author"),
	packages=find_packages(),
	include_package_data=True,
	package_data={
		# Include all .so files in the biasd/likelihood directory
		'biasd.likelihood': ['*.so'],
	},
	install_requires=[
		'numpy>=1.26,<2.0.0',
		'numba>=0.59',
		'matplotlib>=3.9',
		'h5py>=3.9',
		'scipy>=1.12',
		'emcee>=3.1',
		'corner>=2.2',
		'tqdm>=4.66'
	],
	extras_require={
		'test': [
			'sphinx',
			'sphinx_rtd_theme',
			'pytest',
			'pytest-cov',
			'simulate_singlemolecules @ git+https://github.com/ckinzthompson/simulate_singlemolecules@main'
		],
		'gui': [
		],
	},
	cmdclass={
		'build_py': CustomBuildPyCommand,
	},
)