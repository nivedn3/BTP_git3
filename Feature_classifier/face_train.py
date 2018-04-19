




parser = argparse.ArgumentParser(description="FAC")
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--n_iter", type=int, default=1000)
parser.add_argument("--image_size", type=int, default=148)
parser.add_argument("--batch_size", type=int, default=64)

args = parser.parse_args()


def train():

	for i in xrange(1,args.n_iter):

		