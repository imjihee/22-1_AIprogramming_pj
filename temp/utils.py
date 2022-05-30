import torch
import numpy as np
from tqdm import tqdm
"""
AverageMeter, simple_accuracy, valid
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def valid(model, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True, )
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            y = y.view(-1)

            with torch.no_grad():
                logits = model(x)
                eval_loss = loss_fct(logits, y)
                eval_losses.update(eval_loss.item())

                preds = torch.argmax(logits, dim=-1)
                # print(preds.shape)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )
        # epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]

    accuracy = simple_accuracy(all_preds, all_label)

    print("\n")
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % (accuracy * 100))
    return accuracy
