# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Readout error mitigation from backend properties
"""
from functools import reduce

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.providers.basebackend import BaseBackend
from qiskit.result import Result


def backend_readout_error_mitigation(result: Result, backend: BaseBackend):

    if result.backend_name != backend.name():
        raise QiskitError(
            "Given names are different. "
            f"result.backend_name is {result.backend_name} and backend.name() is {backend.name()}"
        )

    n_qubits = backend.configuration().n_qubits
    readout_errors = [backend.properties().readout_error(i) for i in range(n_qubits)]
    error_01 = error_10 = readout_errors
    # TODO: Use error rates of 0 → 1 and 1 → 0.

    A_inv = [
        np.linalg.pinv(
            np.matrix([[1 - error_01[i], error_10[i]], [error_01[i], 1 - error_10[i]]])
        )
        for i in range(n_qubits)
    ]

    mitigation_matrix = reduce(np.kron, reversed(A_inv))

    raw_data = result.get_counts()
    raw_data2 = np.zeros(2 ** n_qubits, dtype=float)
    for stateidx in range(2 ** n_qubits):
        raw_data2[stateidx] = raw_data.get(format(stateidx, f"0{n_qubits}b"), 0)
    mitigated_data = np.dot(reduce(np.kron, reversed(A_inv)), raw_data2)
    return {
        format(stateidx, f"0{n_qubits}b"): mitigated_data.item(stateidx)
        for stateidx in range(2 ** n_qubits)
        if raw_data2[stateidx] != 0
    }
