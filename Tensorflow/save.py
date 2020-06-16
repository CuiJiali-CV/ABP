import shutil
from loadData import DataSet


class saveModule(object):
    def __init__(self, category='Mnist', prior=1, z_size=100,
                 langevin_num=20, lr=0.001, theta=1, delta=0.001):
        self.prior = prior
        self.category = category
        self.z_size = z_size
        self.langevin_num = langevin_num

        self.lr = lr
        self.theta = theta
        self.delta = delta

        self.src_dir = './output/'
        self.des_dir = './output/history/WarmStart/2000{}/fc_dc_noised/langevin{}/z{}_lr{}_delta{}_theta{}_prior{}'.format(
            self.category,
            str(self.langevin_num),
            str(self.z_size),
            str(self.lr),
            str(self.delta),
            str(self.theta),
            str(self.prior))

    def process(self):
        lst = [
            'checkpoint',
            'recon',
            'gens',
            'logs',
        ]
        for dir in lst:
            src = self.src_dir + dir
            des = self.des_dir + '/' + dir
            shutil.copytree(src, des)
            shutil.rmtree(src)

