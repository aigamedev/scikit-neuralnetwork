# -*- coding: utf-8 -*-
#
# scikit-neuralnetwork documentation build configuration file, created by
# sphinx-quickstart on Tue Mar 31 20:28:10 2015.

import sys
import os

project = u'scikit-neuralnetwork'
copyright = u'2015, scikit-neuralnetwork developers (BSD License)' 


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import sknn
version = sknn.__version__
release = sknn.__version__

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage',]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']

pygments_style = 'sphinx'
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

html_short_title = 'scikit-neuralnetwork'
# html_logo = 'img/logo.png'
# html_favicon = 'img/favicon.ico'

html_static_path = ['_static']
htmlhelp_basename = 'sknndoc'
