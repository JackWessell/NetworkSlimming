'''
here we will implement training of a ResNet model using knowledge distillation. In particular, we will use regressor distillation.
This requires modifying our base resnet model to include a regressor layer that maps the intermediate output to a size that matches up with 
a larger model. This helps our smaller model learn an intermediate representation from a larger, more successful model.
A cool fact about ResNets is that no matter the size, they have the same shaped intermediate outputs at each layer. This makes knowledge distillation
incredibly natural and effective for them. I use resnet110 as the teacher by default, but you do not have to stick with this. 
'''
import torch 
import torch.nn as nn
import modified_resnet

import sys
sys.path.append("..")
from baselines import resnet
import time
from utils import benchmarking, get_dataloaders
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", help = "Learning Rate",
                    type = float, default=1e-3)
parser.add_argument("--epochs", help = "Number of epochs to train",
                    type = int, default = 100)
parser.add_argument("--t_weight", help = "Weight to place on teacher loss",
                    type = float, default=.5)
parser.add_argument("--ce_weight", help = "Weight to place on CE loss",
                    type = float, default = .5)
parser.add_argument("--student", help = "Location from which to load student model",
                    type=str, default = "../baselines/resnet20.th")
parser.add_argument("--val", help = "Specific ResNet Architecture (e.g resnet20)",
                    type=int, default = 20)
parser.add_argument("--save_to", help="Location to save the distilled model",
                    type = str, default="models/")
parser.add_argument("--name", help = "Experiment name",
                    type=str)
args = parser.parse_args()


def train_knowledge_distillation(optimizer, student, teacher, loader, soft_weight, ce_weight, epoch):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    teacher.eval()
    student.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        #we don't care about teacher predictions, only intermediate outputs.
        with torch.no_grad():
            _, teach1, teach2, teach3 = teacher(inputs)
        student_pred, stu1, stu2, stu3 = student(inputs)
        L1 = mse_loss(teach1, stu1)
        L2 = mse_loss(teach2, stu2)
        L3 = mse_loss(teach3, stu3)
        label_loss = ce_loss(student_pred, targets)

        loss = ce_weight*label_loss + soft_weight*(L1+L2+L3)
        loss.backward()
        optimizer.step()
        acc = torch.sum(torch.argmax(student_pred, dim = 1) == targets)/len(inputs)
        running_loss += loss.item()
        if i % 50 == 0:
            print(f'Epoch: {epoch} Iteration {i} Loss: {loss.item():.3f} Acc.: {acc:.3f}')
            
def validate(model, criterion, validation):
    batch_time = benchmarking.AverageMeter()
    losses = benchmarking.AverageMeter()
    top1 = benchmarking.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(validation):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output, _, _, _ = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = torch.sum(torch.argmax(output, dim = 1) == target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(validation), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg
if args.val == 20:
    student = modified_resnet.modified_resnet20().cuda()
elif args.val == 32:
    student = modified_resnet.modified_resnet32().cuda()
elif args.val == 44:
    student = modified_resnet.modified_resnet44().cuda()
elif args.val == 56:
    student = modified_resnet.modified_resnet56().cuda()
else:
    raise Exception("Please choose a supported model type")
student_path = args.save_to + str(args.val) + "/" + args.name + ".pt"

teacher = modified_resnet.modified_resnet110().cuda()
#I hard-coded this load as I always used Resnet110 as my teacher. Feel free to change to match your filesystem
state_dict = torch.load("../baselines/fine-tuned/resnet110-B.th")['state_dict']
teacher.load_state_dict(state_dict)
train, validation = get_dataloaders.get_dataloaders()

optimizer = torch.optim.Adam(student.parameters(), lr = args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
val_criterion = nn.CrossEntropyLoss()
for i in range(args.epochs):
    train_knowledge_distillation(optimizer, student, teacher, train, args.t_weight, args.ce_weight, i)
    acc = validate(student, val_criterion, validation)
    print(f'Accuracy @epoch {i}: {acc}')
    scheduler.step()
print("Training Complete")
torch.save(student.state_dict(), student_path)

