#!/usr/bin/env python3
import os
import sys
import flatbuffers
import tflite
import numpy as np
from path_utils import get_repo_root

# Setup
REPO_ROOT = get_repo_root()

# Helpers
def bytes_per_element_from_tensor(tensor):
    TT = tflite.TensorType
    tt = tensor.Type()
    if tt == TT.FLOAT32:
        return 4
    if tt == TT.FLOAT16:
        return 2
    if tt == TT.INT32:
        return 4
    if tt == TT.INT64:
        return 8
    if tt in (TT.UINT16, TT.INT16):
        return 2
    if tt in (TT.UINT8, TT.INT8, TT.BOOL):
        return 1
    return 1


def count_tflite_params(model_path):
    with open(model_path, "rb") as f:
        buf = f.read()
    model = tflite.Model.GetRootAsModel(buf, 0)

    if model.SubgraphsLength() == 0:
        raise RuntimeError("No subgraphs found in model")
    subgraph = model.Subgraphs(0)
    buf_to_tensors = {}

    for ti in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(ti)
        bidx = tensor.Buffer()
        if bidx and bidx > 0:
            buf_to_tensors.setdefault(bidx, []).append(ti)
    total_params = 0

    for bidx, tensor_idxs in buf_to_tensors.items():
        bufobj = model.Buffers(bidx)
        data_len = bufobj.DataLength()

        if data_len == 0:
            continue

        rep_tensor = subgraph.Tensors(tensor_idxs[0])
        bpe = bytes_per_element_from_tensor(rep_tensor)
        elems = data_len // bpe if bpe > 0 else 0
        total_params += elems

    return total_params

# Exports
class ParamCounts:
    def __init__(self, dir, verbose=False):
        self.dir = os.path.expanduser(dir)
        self.verbose = verbose
        print(self.dir)
    def scan_models(self):
        """
        Traverse self.dir, find subdirectories containing 'CPU_ref', and count parameters
        for each .tflite file inside
        returns list of parameter counts - order should be the same as in the excel tables
        """

        results = []
        #print(f"Scanning: {self.dir}")
        for root, dirs, files in os.walk(self.dir):
            dirs.sort()   # ensures alphabetical walk of subdirs
            if os.path.basename(root) == "CPU_ref":
                for f in sorted(files):
                    if f.endswith(".tflite"):
                        model_path = os.path.join(root, f)
                        try:
                            count = count_tflite_params(model_path)
                            results.append(count)
                            if self.verbose:
                                print(f"{f}: {count:,}")
                        except Exception as e:
                            results.append(None)  # placeholder if error
                            if self.verbose:
                                print(f"{f}: Error - {e}")
        return results


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python scan_models.py path/to/base_dir [--verbose]")
    #     sys.exit(1)
    #
    # base_dir = sys.argv[1]
    # verbose = "--verbose" in sys.argv[2:]
    # pc = Param_Counts(base_dir, verbose=verbose)
    # results = pc.scan_models()
    #
    # print("\nParameter counts:")
    # print(results)

    base_dir = "~/Coral-TPU-Characterization/data/models/Image_Classification/"
    pc = ParamCounts(base_dir)
    results = pc.scan_models()
    print(results)