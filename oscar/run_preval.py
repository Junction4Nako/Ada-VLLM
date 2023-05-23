import torch
from sklearn.metrics import precision_recall_curve
import argparse

def main():
    parser = argparse.ArgumentParser(description='precision recall evaluator')
    parser.add_argument('--pred_file', type=str, default=None, help='prediction file to evaluate')
    args = parser.parse_args()
    res = torch.load(args.pred_file)
    labels = [v[1] for k,v in res.items()]
    logits = [v[0] for k,v in res.items()]
    lr_precision, lr_recall, _ = precision_recall_curve(labels, logits)
    for i in range(len(lr_precision)):
        if lr_precision[i] >= 0.9:
            print('Recall (Precision>=0.9):', lr_recall[i])
            break

if __name__=='__main__':
    main()
