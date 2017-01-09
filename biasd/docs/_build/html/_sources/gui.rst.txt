.. _gui:

BIASD GUI
=========

Also included is a graphical user interface to facilitate analysis with BIASD. Code for it is found in the `./biasd/gui` folder. It is written with `PyQt5 <https://riverbankcomputing.com/software/pyqt/intro>`_, which is a Python binding for `Qt5 <https://www.qt.io/>`_. Thus, you'll need PyQt5 installed to run it, which you can get from a terminal with

.. code-block:: bash

	conda install PyQt5
	
Once you have this installed, you can invoke the GUI from Python with

.. code-block:: python

	import biasd as b
	b.gui.launch()
	
There is a script to do this in the main directory called `./launch_GUI.py`. You can launch it from that directory with

..code-block:: bash

	python launch_GUI.py

or

..code-block:: bash

	./launch_GUI.py 

Functionality
-------------
Functionality in the GUI is not as rich as in the Python module, however, you can easily explore HDF5 SMD files, which is not trivial from a Python script.