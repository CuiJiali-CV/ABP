import os
import argparse

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-isTraining', type=bool, default=True, help='add noise during the langevin or not')
        self.parser.add_argument('-continue_train', type=bool, default=False, help='add noise during the langevin or not')



        self.parser.add_argument('-prior', type=int, default=1, help='training epochs')
        self.parser.add_argument('-num', type=int, default=2000, help='training epochs')
        self.parser.add_argument('-vis_step', type=int, default=100, help='training batch size')
        self.parser.add_argument('-Train_Epochs', type=int, default=1000, help='how many rows of images in the output')
        self.parser.add_argument('-batch_size', type=int, default=128, help='how many columns of images in the output')
        self.parser.add_argument('-z_size', type=int, default=200, help='output image size')
        self.parser.add_argument('-langevin_num', type=int, default=30, help='How many images to generate during testing')
        self.parser.add_argument('-lr', type=float, default=0.0001, help='sigma of reference distribution')
        self.parser.add_argument('-theta', type=float, default=0.1, help='sigma of reference distribution')
        self.parser.add_argument('-delta', type=float, default=0.1, help='sigma of reference distribution')
        self.parser.add_argument('-category', default='Mnist', help='training category')
        self.parser.add_argument('-checkpoint_dir', default='./output/checkpoint/', help='training category')
        self.parser.add_argument('-recon_dir', default='./output/recon/', help='training category')
        self.parser.add_argument('-logs_dir', default='./output/logs/', help='training category')
        self.parser.add_argument('-gen_dir', default='./output/gens/', help='training category')
        self.parser.add_argument('-with_noise', type=bool, default=True, help='add noise during the langevin or not')


    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.checkpoint_dir):
            os.makedirs(self.opt.checkpoint_dir)
        file_name = os.path.join(self.opt.checkpoint_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
        return self.opt