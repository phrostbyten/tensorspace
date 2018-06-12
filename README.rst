TensorSpace
===========

TensorSpace is a reference implementation of an artificial intelligence lab.

Why?
~~~~

I was tired of setting up ad hoc environments for various research experiments. I wanted a solution that can turn any "computer" into a research environment that I can use right away.

Features
~~~~~~~~

* Downloads and normalizes datasets. Currently only COCO, but more coming. Saves everything into a nice organized directory structure.
* Single annotation schema for all datasets. You don't need to research with just one dataset at a time anymore. You do queries like "give me all images with bounding boxes from all datasets". 
* Automatically preprocess vectors or other intermidate datasets.
* **Coming soon**: Demos and models that use the data.
* **Coming soon**: GraphQL API for running models
* **Coming soon**: Multiple deployment targets. This will include Kubernetes.

Installation
~~~~~~~~~~~~

::

    pip install tensorspace

Usage
~~~~~

::

    tensorspace up
