"""
Compilation of C++ code for further use with ctypes
"""

# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/simulator_awgn_python/.
# Copyright (c) 2023 Kirill Andreev.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os


def compile_shared(src_list, lib_path):
    """
    Compile the C++ implementation for further use with ctypes
    """
    wdir = os.path.dirname(__file__)
    # if os.path.isfile(LIB_PATH):
    #     return
    src_abs = [os.path.join(wdir, s) for s in src_list]

    if not os.popen('which g++').read():
        raise RuntimeError('g++ not found.')

    for src_file in src_abs:
        os.system(
            f'g++ -Wall -Werror -O3 -fPIC -c -o {src_file}.o {src_file}.cpp')
    os.system(
        'g++ -shared -o ' +
        os.path.join(wdir, lib_path) + ' ' +
        ''.join([s + '.o ' for s in src_abs])
    )
    obj_files = os.path.join(wdir, '*.o')

    os.system(f'rm {obj_files}')
