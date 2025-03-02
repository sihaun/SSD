import argparse

def config():
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments with default values from JSON
    parser.add_argument('-a', '--arch', metavar='ARCH', default='tf_efficientnet_b7',
                        choices=['tf_efficientnet_b7', 'mobilenetv3_ssd'],
                        help='model architecture')
    parser.add_argument('--load-weights', default='default', type=str, metavar='FILE.pth')
    parser.add_argument('--datapath', default='image_data', type=str, metavar='PATH',
                        help='where you want to load/save your dataset?')  
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=5, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save', default='weight.pt', type=str, metavar='FILE.pt',
                        help='name of checkpoint for saving model\'s WEIGHT')

    # Parse command-line arguments
    cfg = parser.parse_args()

    return cfg