from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import argparse

def main():
    import logging
    parser  = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    # TODO: add argument options
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)
    # TODO: implement here.
    logging.debug('run an impala example with gym env')
    

if __name__ == '__main__':
    main()