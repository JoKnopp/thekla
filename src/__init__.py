# -*- coding: UTF-8 -*-

#This file is part of Thekla.

__version__ = '0.1'
__author__ = 'Johannes Knopp <johannes@informatik.uni-mannheim.de>'
__copyright__ = '© Copyright 2013 Johannes Knopp'

import logging

class NullHandler(logging.Handler):
    def emit(self, record):
        pass

_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)
_log.addHandler(NullHandler())
