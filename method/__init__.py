# from importlib import import_module; train = lambda method, args: import_module(f"{__name__}.{"llada" if args.model_type in ["llada_instruct", "llada_1.5"] else "dream"}").train(args)
from importlib import import_module

def train(args):
    module_name = "llada" if args.model_type in ["llada_instruct", "llada_1.5"] else "dream"
    import_module(f"{__name__}.{module_name}").train(args)