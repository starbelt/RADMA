import sys
import numpy as np
import flatbuffers
import tflite

def count_tflite_params(model_path):
    # Load TFLite flatbuffer
    with open(model_path, "rb") as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    param_count = 0
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        # Filter only constant tensors (weights/biases usually in buffers > 0)
        if tensor.Buffer() > 0:
            shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
            if shape:  # ignore scalars/placeholders
                param_count += np.prod(shape)

    return param_count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_params.py <model.tflite>")
        sys.exit(1)

    model_path = sys.argv[1]
    params = count_tflite_params(model_path)
    print(f"Model: {model_path}")
    print(f"Total parameters: {params:,}")
