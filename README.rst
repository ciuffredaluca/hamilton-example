================
hamilton-example
================


.. image:: https://img.shields.io/pypi/v/hamilton_example.svg
        :target: https://pypi.python.org/pypi/hamilton_example

.. image:: https://img.shields.io/travis/ciuffredaluca/hamilton_example.svg
        :target: https://travis-ci.com/ciuffredaluca/hamilton_example

.. image:: https://readthedocs.org/projects/hamilton-example/badge/?version=latest
        :target: https://hamilton-example.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




This repo contains an implementation of a simple ML pipeline using `Hamilton DAG library <https://github.com/DAGWorks-Inc/hamilton/tree/main>`_ and is a minor review on the `Hamilton sklearn example code. <https://github.com/DAGWorks-Inc/hamilton/tree/main/examples/model_examples/scikit-learn>`_

From root run: 

``
python -m hamilton_example.cli --model_type <MODEL_TYPE>
``

where MODEL_TYPE is:

- logistic
- svm

Features
--------

* Free software: GPL-3.0-only
* Documentation: https://hamilton-example.readthedocs.io.

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `briggySmalls/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`briggySmalls/cookiecutter-pypackage`: https://github.com/briggySmalls/cookiecutter-pypackage
