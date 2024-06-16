import torch
import argparse
import sys
sys.path.append("..")
from baselines import resnet
parser = argparse.ArgumentParser()
parser.add_argument("--load_path", help = "Location from which to load pre-trained model",
                    type=str)
parser.add_argument("--save_path", help = "Location from which to load pre-trained model",
                    type=str)
parser.add_argument("--val", help = "Specific ResNet Architecture (e.g resnet20)",
                    type=int, default = 20)
args = parser.parse_args()

if args.val == 20:
    model = resnet.resnet20().cuda()
elif args.val == 32:
    model = resnet.resnet32().cuda()
elif args.val == 44:
    model = resnet.resnet44().cuda()
elif args.val == 56:
    model = resnet.resnet56().cuda()
elif args.val == 110:
    model = resnet.resnet110().cuda()
else:
    raise Exception("Please choose a supported model type")
model.load_state_dict(torch.load(args.load_path)['state_dict'])
dummy_in = torch.randn(3,3,32,32).cuda()
torch.onnx.export(model,                                       # model
                  dummy_in,                                 # model input
                  args.save_path,                                  # path
                  export_params=True,                               # store the trained parameter weights inside the model file
                  opset_version=14,                                 # the ONNX version to export the model to
                  do_constant_folding=False,                        # constant folding for optimization
                  input_names = ['input'],                          # input names
                  output_names = ['output'],                        # output names
                  dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                'output' : {0 : 'batch_size'}})