#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
from ruamel.yaml import YAML
import io
import tempfile
import tensorflow as tf
from tensorflow.python.framework import graph_util
from stable_baselines import PPO2
import tf2onnx


#
import os
import math
import argparse
import numpy as np
import tensorflow as tf

#
from stable_baselines import logger

#
from rpg_baselines.common.policies import MlpPolicy
# from stable_baselines3 import PPO
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
from tensorflow.python.framework import graph_util

# import stable_baselines3.common.utils as U
# import ruamel as YAML
#
from flightgym import QuadrotorEnv_v1


def export_onnx(model_path, name):
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    import tf2onnx

    model = PPO2.load(model_path)
    sess = model.sess

    input_names  = ["input/Ob:0"]
    output_names = ["model/pi/add:0"]

    print("Freezing the graph...")

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ["model/pi/add"]
    )

    print("Converting to ONNX...")

    model_proto, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        input_names=input_names,
        output_names=output_names,
        opset=11
    )

    with open(name, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"✅ ONNX export successful: {name}")




def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-19-11-44-04_Iteration_530.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-19-13-28-47_Iteration_787.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-17-08-17-12_Iteration_1796.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-17-14-14-30_Iteration_4392.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-18-09-12-38_Iteration_1398.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-18-13-51-47_Iteration_5322.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-24-15-48-44_Iteration_18975.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-25-11-24-35_Iteration_3962.zip',
    parser.add_argument('-w', '--weight', type=str, default='./saved_new/2026-02-02-18-56-00_Iteration_992659.zip',
                        help='trained weight path')
    return parser


def main():
    # args = parser().parse_args()
    # cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
    #                        "/flightlib/configs/vec_env.yaml", 'r'))
    # if not args.train:
    #     cfg["env"]["num_envs"] = 1
    #     cfg["env"]["num_threads"] = 1

    # if args.render:
    #     cfg["env"]["render"] = "yes"
    # else:
    #     cfg["env"]["render"] = "no"

    # yaml = YAML() 
    # string_stream = io.StringIO() 
    # yaml.dump(cfg, string_stream) 
    # yaml_string = string_stream.getvalue()
    # print(yaml_string)
    # env = wrapper.FlightEnvVec(QuadrotorEnv_v1(yaml_string))

    args = parser().parse_args()
    yaml = YAML()

    # Load the base config
    cfg_path = os.path.join(os.environ["FLIGHTMARE_PATH"], "flightlib/configs/vec_env.yaml")
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f)

    # Modify config according to args
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    cfg["env"]["render"] = "yes" if args.render else "no"

    # Save modified config to a temporary YAML file
    tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
    yaml.dump(cfg, tmp_file)
    tmp_file_path = tmp_file.name
    tmp_file.close()

    
    print(f"Using temporary config file: {tmp_file_path}")

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(tmp_file_path))

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # env.connectUnity()
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved_new'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        model = PPO2(
            tensorboard_log=saver.data_dir,
            policy=MlpPolicy,  # check activation function  
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
            env=env,
            lam=0.95,
            gamma=0.99,  # lower 0.9 ~ 0.99
            # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
            n_steps=250,
            ent_coef=0.00,
            learning_rate=3e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            nminibatches=1,
            noptepochs=10,
            cliprange=0.2,
            verbose=1,
        )

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        # 250000000  is 500 iterations.
        # 50000000   is 100 iterations.
        logger.configure(folder=saver.data_dir)
        summary_writer = tf.summary.FileWriter(saver.data_dir,graph=tf.get_default_graph())
        model.summary_writer = summary_writer

        model.learn(
            total_timesteps=int(250000000000),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)
        summary_writer.flush()
        summary_writer.close()

    # # Testing mode with a trained weight
    else:

        export_onnx(args.weight, "2ilight.onnx")

        # model = PPO2.load(args.weight)

        # params = model.get_parameters()
        # np.save("dr_m0.5_a_0.2_i_0.2_params.npy", params)

        # test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
