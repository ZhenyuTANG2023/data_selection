import argparse
import datetime
import yaml

class LoadFromFile(argparse.Action):
    def __cal__(self, parser:argparse.ArgumentParser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


class Argument:
    def __init__(self, args):
        self.info:str = args.info

        self.gpu:bool = args.gpu
        self.run_with_slurm:bool = args.run_with_slurm
        self.data_folder:str = args.data_folder
        self.vis:bool = args.vis
        self.observe:bool = args.observe
        self.fullset:bool = args.fullset

        self.dataset:str = args.dataset
        self.valid_ratio:float = args.valid_ratio
        self.model:str = args.model
        self.pretrained:bool = args.pretrained
        self.criterion:str = args.criterion
        self.optimizer:str = args.optimizer
        self.lr_sched:str = args.lr_sched

        self.rnd_seed:int = args.rnd_seed
        self.num_epochs:int = args.num_epochs
        self.batch_size:int = args.batch_size
        self.lrate:float = args.lrate
        self.patience:int = args.patience
        self.weight_decay:float = args.weight_decay
        self.momentum:float = args.momentum
        self.use_grad_clip:bool = args.grad_clip
        self.grad_clip:float = 0.1

        self.multi_label:bool = args.multi_label
        
        self.use_coreset:bool = args.use_coreset
        
        if "selection_method" in dir(args):
            self.selection_method:str = args.selection_method
            self._parse_selection_method(self.selection_method)
            self.data_val_batch_size:int = args.data_val_batch_size
            self.sample_size:float = args.sample_size # float: percentage
            self.lr_adjust:bool = args.lr_adjust

        for attr in dir(args):
            if attr[0] != "_" and attr not in dir(self):
                setattr(self, attr, getattr(args, attr))

    def _parse_selection_method(self, selection_method:str):
        method_confs = [s.strip() for s in selection_method.split("&")]
        self.data_val_method:str = method_confs[0]
        self.data_val_dims:str = method_confs[1]
        self.selection_strategy:str = method_confs[2]

    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        arg_str = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + "\n"
        for attr in dir(self):
            if attr[0] != "_":
                arg_str += f"{attr}: {getattr(self, attr)}\n"
        return arg_str


def parse_args(default_config_file:str="./config/fullset.yaml"):
    parser = argparse.ArgumentParser(description="Full dataset image classification baselines")

    parser.add_argument("--info", "-i", type=str, default="No info", 
        help="Additional information")
    parser.add_argument("--config", "-c", type=str, default=default_config_file, metavar="FILE",
        help="Argument configuration YAML file")
    
    parser.add_argument("--gpu", action="store_true",
        help="Whether to use GPU for training")
    parser.add_argument("--run_with_slurm", action="store_true", 
        help="marking if the program is run with slurm so as to disable tqdm")
    parser.add_argument("--data_folder", type=str, default="/path/to/data",
        help="Data folder")
    parser.add_argument("--vis", action="store_true",
        help="Whether to visualize the training curve")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--fullset", action="store_true", help="toggle using fullset")
    
    parser.add_argument("--dataset", "-d", type=str, default="svhn", 
        help="Dataset to train on, current supported options: svhn | cifar10 | cifar100 | mnist")
    parser.add_argument("--valid_ratio", "-v", type=float, default=0.3,
        help="The valid split ratio")
    parser.add_argument("--model", "-m", type=str, default="resnet18", 
        help="Model to train, current supported options: resnet18")
    parser.add_argument("--pretrained", "-p", action="store_true",
        help="Whether to use the pretrained model parameters")
    parser.add_argument("--criterion", type=str, default="cross_entropy",
        help="The criterion used for back-propagation")
    parser.add_argument("--optimizer", "-o", type=str, default="adam",
        help="The optimizer used in the training process")
    parser.add_argument("--lr_sched", type=str, default="none",
        help="The learning rate scheduler for training")
    parser.add_argument("--selection_method", type=str, default="gradnorm&1-dim&thresh")
    
    parser.add_argument("--rnd_seed", type=int, default=1,
        help="The random seed used to initialize the training")
    parser.add_argument("--num_epochs", type=int, default=100,
        help="Number of epochs to run in the training")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
        help="Training batch size")
    parser.add_argument("--lrate", "-l", type=float, default=0.001,
        help="Training learning rate")
    parser.add_argument("--patience", type=int, default=7,
        help="The patience of iterations for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0,
        help="The weight decay factor used in training")
    parser.add_argument("--momentum", type=float, default=0.0,
        help="The momentum used in SGD-based training")
    parser.add_argument("--grad_clip", action="store_true",
        help="Toggle using gradient clip")
    
    parser.add_argument("--multi_label", action="store_true")

    given_args, remaining = parser.parse_known_args()
    if given_args.config:
        with open(given_args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    ARGS = parser.parse_args(remaining)

    return Argument(ARGS)