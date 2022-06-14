import python.taso as ts
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to input ONNX file", required=True)

args = parser.parse_args()

# graph = taso.load_onnx("/home/ubuntu/taso/onnx/squeezenet1.1.onnx")
# graph = taso.load_onnx("/home/ubuntu/taso/onnx/bertsquad10.onnx")
graph = ts.load_onnx(args.file)
# graph = xflow.load("/home/ubuntu/resnext-101.onnx")
# graph = xflow.load("/home/ubuntu/ONNXModel/inception_v2/model.onnx")
print(" original_cost = {}".format(graph.cost()))
new_graph = ts.optimize(graph, alpha=1.0, budget=100, print_subst=True)
print("optimized_cost = {}".format(new_graph.cost()))
onnx_model = ts.export_onnx(new_graph)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "{}.taso.onnx".format(args.file))
