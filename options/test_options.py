from .base_options import BaseOptions


class TestOptions(BaseOptions):


    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # -------------------- Test Settings --------------------
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='directory for saving test results')

        parser.add_argument('--eval', action='store_true',
                            help='use model.eval() during test')

        # explicitly mark phase
        parser.add_argument('--phase', type=str, default='test')

        self.isTrain = False
        return parser
