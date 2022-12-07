import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    # Data loader.
    Data_Loader = None
    Data_Loader = get_loader(config.image_dir,config.image_size,config.batch_size,config.dataset,config.mode,config.num_workers,config.wmat_dir)
    Data_Loader_Test = get_loader(config.image_val_dir,config.image_size,config.batch_size,config.dataset,config.mode,config.num_workers,config.wmat_dir)
	
    # Solver for training and testing PIMoG 
    solver = Solver(Data_Loader, Data_Loader_Test, config)
    if config.mode == 'train_mask':
    	solver.train_mask()
    elif config.mode == 'test_accuracy':
    	solver.test_accuracy()
    elif config.mode == 'test_embedding':
    	solver.test_embedding()

	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
	# Model configuration.
    parser.add_argument('--image_size', type=int, default=128, help='host image size')
    parser.add_argument('--num_channels',type=int, default=64, help='channels for discriminator')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='train_mask',choices=['train_mask','test_accuracy','test_embedding'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--lambda1',type=float, default=3, help='the weights of message loss')
    parser.add_argument('--lambda2',type=float, default=1, help='the weights of image loss')
    parser.add_argument('--lambda3',type=float, default=0.001, help='the weights of GAN loss')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of total epochs')
    parser.add_argument('--embedding_epoch', type=int, default=0, help='The adversarial training epoch')
    parser.add_argument('--distortion',type=str, default='ScreenShooting', choices=['Identity','ScreenShooting'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=99, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train_mask', choices=['train_mask','test_accuracy','test_embedding'])

    # Directories.
    parser.add_argument('--image_dir', type=str, default='Dataset/COCOMask/train/train_class/') 
    parser.add_argument('--image_val_dir', type=str, default='Dataset/COCOMask/val/val_class/')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--model_name',type=str,default='Encoder_Decoder_Model')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--wmat_dir', type=str, default='results/WatermarkMatrix/w.mat')

    # Step size.
    parser.add_argument('--log_step', type=int, default=40)
    parser.add_argument('--model_save_step', type=int, default=1)

    config = parser.parse_args()
    print(config)
    main(config)