#!/usr/bin/env python
# Copyright (C) 2018  Mario Juez-Gil <mariojg@ubu.es>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Engine Dataset loading.

Opens h5 file dataset and loads CVV and V datasets prepared for
its usage with keras DNN library.
Loading a subset containing the first N seconds of the dataset
is also possible.

"""

import h5py
import numpy as np
import random

__author__ = "Mario Juez-Gil"
__copyright__ = "Copyright 2018, Mario Juez-Gil"
__credits__ = ["Mario Juez-Gil", "Cesar Garcia-Osorio",
               "Álvar Arnaiz-González", "Carlos López"]
__license__ = "GPLv3"
__version__ = "0.5"
__maintainer__ = "Mario Juez-Gil"
__email__ = "mariojg@ubu.es"
__status__ = "Development"

ALL = "all"  # filter, workload, and frequency shared discriminator

CVV = "cvv"
V = "v"

MIXED_WORKLOAD = "mixed"
MEDIUM_WORKLOAD = "medium"
NO_WORKLOAD = "no"

FREQ_3HZ = "three"
FREQ_12HZ = "twelve"
FREQ_30HZ = "thirty"
FREQ_60HZ = "sixty"
FREQ_LINE = "line"

T_STEPS = 1  # shape index of timesteps value

VA = (CVV, 0)
VB = (CVV, 1)
VC = (CVV, 2)
GND = (CVV, 3)
CA = (CVV, 4)
CB = (CVV, 5)
CC = (CVV, 6)
CN = (CVV, 7)
EC = (CVV, 8)
AREF = (V, 0)
AX = (V, 1)
AY = (V, 2)
AZ = (V, 3)

MASK_ALL = (VA, VB, VC, GND, CA, CB, CC, CN, EC, AREF, AX, AY, AZ)
MASK_NO_GND = (VA, VB, VC, CA, CB, CC, CN, EC, AREF, AX, AY, AZ)
MASK_CVV = (VA, VB, VC, GND, CA, CB, CC, CN, EC)
MASK_CVV_NO_GND = (VA, VB, VC, CA, CB, CC, CN, EC)
MASK_V = (AREF, AX, AY, AZ)

# With this implementation the file is going to be opened each time we request
# a window, which could penalize the performance.
def data_window(begin=0, size=5, workload=ALL, frequency=ALL, mask=MASK_ALL,
                root_path="/home/mariojg/research/datasets/motor_faults"):

	def read_indexes():
		with h5py.File(indexes_file, "r") as si:
			indexes = list(si[workload][frequency][ALL])
			random.shuffle(indexes)
			# when using with statements, it closes the file after returning.
			return indexes

	def mask_to_dict():
		mask_dict = {
			CVV: [],
			V: []
		}

		for data_filter, col_index in mask:
			mask_dict[data_filter].append(col_index)

		return mask_dict

	def remove_bearing_defect(inputs, outputs):
		def indexes_without_bd(outputs):
			bd_indexes = []
			bd = (0, 0, 0, 1)
			idx = 0
			for exp in outputs:
				if(np.array_equal(exp, bd)):
					bd_indexes.append(idx)
				idx += 1

			nobd_indexes = np.arange(len(outputs)).tolist()
			for index in sorted(bd_indexes, reverse=True):
				del nobd_indexes[index]

			return nobd_indexes

		no_bd_indexes = indexes_without_bd(outputs)
		filtered_inputs = None
		filtered_outputs = outputs[no_bd_indexes,:3]
		if(len(inputs) == 2):
			filtered_inputs = [inputs[0][no_bd_indexes],inputs[1][no_bd_indexes]]
		else:
			filtered_inputs = inputs[no_bd_indexes]
		
		return filtered_inputs, filtered_outputs

	dataset_file = f"{root_path}/full_dataset_normalized.h5"
	indexes_file = f"{root_path}/subdatasets_indexes.h5"
	num_seconds = 10
	indexes = read_indexes()
	mask = mask_to_dict()

	with h5py.File(dataset_file, "r") as ds:
		num_timesteps = {
			CVV: int((ds[CVV].shape[T_STEPS] / 10) * num_seconds),
			V: int((ds[V].shape[T_STEPS] / 10) * num_seconds)
		}

		labels = [feats[2:6] for feats in ds["exp"]]

		inputs = []
		outputs = np.array([labels[idx] for idx in indexes])

		for data_filter, cols in mask.items():
			if len(cols) > 0:
				tps = int(num_timesteps[data_filter] / num_seconds)
				first_step = int(begin * tps)
				last_step = int((begin + size) * tps)
				X = []
				for idx in indexes:
					ins = ds[data_filter][idx, first_step:last_step, cols]
					X.append(np.array(ins))
				inputs.append(np.array(X))

		if len(inputs) == 1:
			inputs = inputs[0]
		return remove_bearing_defect(inputs, outputs)
