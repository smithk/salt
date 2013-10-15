Suggest-A-Learner Toolbox [SALT]
================================

About
-----

Toolbox to suggest a machine learning technique and options best suited to a particular data set.

Installation and Prerequisites
------------------------------

This software runs on python 2.7+ and Python 3.3+ (with an embedded copy of `six <http://pythonhosted.org/six>`_ for compatibility
between Python versions).
It also relies on the following freely-available packages:

- Numpy >= 1.7.1
- Scipy >= 0.12.0
- Matplotlib >= 1.3.0
- Scikit >= 0.14.1

The prerequisites are listed in the file ``requirements_1.txt`` and ``requirements_2.txt`` [*]_. The easiest way to ensure that you have the
right versions and dependencies is to install them with pip, if available, as follows::

    pip install -r requirements_1.txt
    pip install -r requirements_2.txt

You will also need `sphinx <http://sphinx-doc.org>`_ if you wish to build the documentation.

Installing a virtual environment (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to isolate the runtime environment used by this package, so as not to break dependencies or overwrite packages
used by other programs installed on your computer. Use tools such as virtualenv to achieve this::

    pip install virtualenv
    mkdir ~/.virtualenvs
    virtualenv ~/.virtualenvs/salt --no-site-packages

That will create a clean python environment that you can access anytime by typing::

    . ~/.virtualenvs/salt/bin/activate

(Note that the name of your virtual environment will be added to your command line prompt as ``(salt) $`` to let you know that it is up and running.)

Testing
-------

After installation, please run ``python -m test`` in the base ``salt`` directory, to run a series of tests to ensure that your copy of SALT
and its prerequisites are installed and running correctly.

Common issues
-------------

Linux
~~~~~

For SALT to work on graphical mode on Python 3, make sure you have installed the ``python3-tk`` package.

**Footnotes**

.. [*] Two requirement files are needed, because packages that depend on others (as is the case for [SALT]), will not know their requirements have been installed
       until the whole file has been processed.
