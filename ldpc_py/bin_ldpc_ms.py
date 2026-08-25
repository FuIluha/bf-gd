import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcMsDecoder(BinLdpcDecoderBase):
    """Implementation of min-sum decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)

        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)

        self.edges_count = len(self.edge_cn)

        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks
        )

        self.check_offsets = np.concatenate((
            np.array([0]),
            np.cumsum(check_degrees),
        ))

    def decode(self, llr_in, llr_out):
        def sign(a):
            return -1.0 if a < 0 else 1.0

        def horizontal_step(edge_cn_to_vn):
            for i in range(len(self.check_offsets) - 1):
                l = self.check_offsets[i]
                r = self.check_offsets[i + 1]
                first_min = float("inf")
                second_min = float("inf")
                sign_prod = 1
                for j in range(l, r):
                    if first_min > np.abs(edge_vn_to_cn[j]):
                        second_min = first_min
                        first_min = np.abs(edge_vn_to_cn[j])
                    elif second_min > np.abs(edge_vn_to_cn[j]):
                        second_min = np.abs(edge_vn_to_cn[j])
                    sign_prod *= sign(edge_vn_to_cn[j])
                for j in range(l, r):
                    edge_cn_to_vn[j] = sign_prod * sign(edge_vn_to_cn[j])
                    if np.abs(edge_vn_to_cn[j]) < second_min:
                        edge_cn_to_vn[j] *= second_min
                    else:
                        edge_cn_to_vn[j] *= first_min

        edge_vn_to_cn = np.zeros(self.edges_count)
        for i in range(self.edges_count):
            edge_vn_to_cn[i] = llr_in[self.edge_vn[i]]

        # initial step
        edge_cn_to_vn = np.zeros(self.edges_count) + float("inf")
        horizontal_step(edge_cn_to_vn)

        for iteration in range(self.n_iterations):
            # получение vn to cn
            llr_out[:] = llr_in.copy()
            for i in range(self.edges_count):
                llr_out[self.edge_vn[i]] += edge_cn_to_vn[i]
            for i in range(self.edges_count):
                edge_vn_to_cn[i] = (llr_out[self.edge_vn[i]] - edge_cn_to_vn[i])

            # проверка
            decoded_bits = np.array(llr_out < 0, dtype=np.uint8)
            if (not np.any(self.syndrome(decoded_bits)) or 
                iteration == self.n_iterations - 1):
                return iteration + 1

            # получение cn to vn
            horizontal_step(edge_cn_to_vn)

        return self.n_iterations