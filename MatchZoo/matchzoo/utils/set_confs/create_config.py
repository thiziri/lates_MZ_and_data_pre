import sys
import json
from itertools import product
from os.path import join
from tqdm import tqdm
import numpy as np

# generate a list of float intervals
def frange(start, stop, step=0.1):
    i = float(start)
    while i < float(stop):
        yield i
        i += float(step)


if __name__ == '__main__':
    config_file = sys.argv[1]  # basic configuration
    variables_file = sys.argv[2]  # parameters to be varied
    out = sys.argv[3]  # output folder
    basic_configuration = json.load(open(config_file))
    to_variate = json.load(open(variables_file))

    combinations = []
    variable_values = {}
    print("Combine parameters ...")
    for parameter in tqdm(to_variate):
        if isinstance(to_variate[parameter][0], list):
            values = [round(n, 1) for n in list(np.arange(to_variate[parameter][0][0],
                                                          to_variate[parameter][0][1]+to_variate[parameter][1],
                                                          to_variate[parameter][1]))]  # exp: 0.1
        else:
            values = to_variate[parameter]
        variable_values[parameter] = values

    combinations = list(product(*[variable_values[variable] for variable in variable_values]))

    print("List of configurations:")
    configurations = []
    for combination in tqdm(combinations):
        config = {}
        for i, parameter in enumerate(list(variable_values.keys())):
            config[parameter] = combination[i]
        configurations.append(config)

    print(configurations)

    print("Create valid configurations...")
    valid_config = basic_configuration.copy()
    model_py = basic_configuration["model"]["model_py"].split(".")[1]
    model_weights = valid_config["global"]["weights_file"]
    save_path = valid_config["outputs"]["predict"]["save_path"]
    unique_conf = []
    for configuration in tqdm(configurations):
        # print(valid_config)
        for parameter in configuration:
            # print(parameter)
            if len(parameter.split(',')) == 1:
                if parameter in valid_config["model"]["setting"]:
                    valid_config["model"]["setting"][parameter] = configuration[parameter]
                if parameter in valid_config["inputs"]["share"]:
                    valid_config["inputs"]["share"][parameter] = configuration[parameter]
            else:
                # print(valid_config["model"]["setting"]['number_q_lstm_units'])
                # print(parameter, parameter.split(','))
                for sub_param in parameter.split(','):
                    valid_config["model"]["setting"][sub_param] = configuration[parameter]
                if parameter in valid_config["inputs"]["share"]:
                    valid_config["inputs"]["share"][sub_param] = configuration[parameter]
        # print(json.dumps(valid_config, indent=2))
        # print(parameter)

        valid_file_name = "_".join([model_py,
                              "Qlstm", str(valid_config["model"]["setting"]["number_q_lstm_units"]),
                              "Dlstm", str(valid_config["model"]["setting"]["number_d_lstm_units"]),
                              "mask0", str(valid_config["model"]["setting"]["mask_zero"]),
                              "train_embed", str(valid_config["inputs"]["share"]["train_embed"])
                              ])
        valid_config["global"]["weights_file"] = model_weights + valid_file_name + ".weights"
        valid_config["outputs"]["predict"]["save_path"] = save_path + valid_file_name + ".txt"
        # print(json.dumps(valid_config, indent=2))
        if not configuration in unique_conf:
            unique_conf.append(configuration)
        conf_file = open(join(out, valid_file_name+".config"), 'w')
        conf_file.write(json.dumps(valid_config, indent=2))

    print("{%d} configurations" % len(unique_conf))



