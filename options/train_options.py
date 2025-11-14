from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # -------------------- Logging & Display --------------------
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')

        # -------------------- Checkpoints --------------------
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--evaluation_freq', type=int, default=5000,
                            help='frequency of evaluation')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='starting epoch count')
        parser.add_argument('--pretrained_name', type=str, default=None,
                            help='resume training from another checkpoint')
        parser.add_argument('--phase', type=str, default='train')

        # -------------------- Training Hyperparameters --------------------
        parser.add_argument('--n_epochs', type=int, default=200,
                            help='epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=200,
                            help='epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='Adam momentum term beta1')
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='Adam momentum term beta2')
        parser.add_argument('--lr', type=float, default=0.0002,
                            help='initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy: linear | step | plateau | cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
