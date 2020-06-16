from model import Abp
from config import opts

def main():
    config = opts().parse()
    m = Abp(config)
    m.train()


if __name__=='__main__':
    main()
