import os
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py

class CustomBuildPyCommand(build_py):
	def run(self):
		print("Running CustomBuildPyCommand")
		# Change the working directory and run the make command
		make_dir = os.path.join(os.getcwd(), 'biasd/likelihood')
		print(f"Changing directory to: {make_dir}")
		subprocess.check_call(['make'], cwd=make_dir)
		# Proceed with the normal build_py
		build_py.run(self)

setup(
	cmdclass={
		'build_py': CustomBuildPyCommand,
	}
)