# SPDX-License-Identifier: Apache-2.0


import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class BlackmanWindow(Base):

    @staticmethod
    def export() -> None:
        # Test periodic window
        node = onnx.helper.make_node(
            'BlackmanWindow',
            inputs=['x'],
            outputs=['y'],
        )
        size = np.int32(10)
        a0 = .42
        a1 = -.5
        a2 = .08
        y = a0
        y += a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
        y += a2 * np.cos(4 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(node, inputs=[size], outputs=[y],
               name='test_blackmanwindow')

        # Test symmetric window
        node = onnx.helper.make_node(
            'BlackmanWindow',
            inputs=['x'],
            outputs=['y'],
            periodic=0
        )
        size = np.int32(10)
        a0 = .42
        a1 = -.5
        a2 = .08
        y = a0
        y += a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
        y += a2 * np.cos(4 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
        expect(node, inputs=[size], outputs=[y],
               name='test_blackmanwindow_symmetric')