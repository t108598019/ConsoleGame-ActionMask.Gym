import os
import sys
sys.path.append('./')

from env.EmptyRooms.action_mask_env import ActionMaskEnv
from stable_baselines import PPO2
from stable_baselines.common.vec_env import  DummyVecEnv
from stable_baselines.common.custom_saver import CustomSaver
from examples.utils.utils import get_policy

tensorboard_folder = './tensorboard/EmptyRooms/action_mask/'
model_folder = './models/EmptyRooms/action_mask/'
if not os.path.isdir(tensorboard_folder):
    os.makedirs(tensorboard_folder)
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

policy = ''
model_tag = ''
if len(sys.argv) > 1:
    policy = sys.argv[1]
    model_tag = '_' + sys.argv[1]

env = DummyVecEnv([lambda: ActionMaskEnv()])

saver = CustomSaver('checkpoints/EmptyRooms/action_mask', 'info//EmptyRooms/action_mask', 'EmptyRooms_PPO2', mode=1)
model = PPO2(get_policy(policy), env, verbose=0, nminibatches=1, tensorboard_log=tensorboard_folder, saver=saver)
model.learn(total_timesteps=500000, tb_log_name='PPO2' + model_tag)

model.save(model_folder + "PPO2" + model_tag + '_' + saver.timestamp)
del model
model = PPO2.load(model_folder + "PPO2" + model_tag + '_' + saver.timestamp)

done = False
states = None
action_masks = []
obs = env.reset()

while not done:
    action, states = model.predict(obs, states, action_mask=action_masks)
    obs, _, done, infos = env.step(action)
    env.render()
    action_masks.clear()
    for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask) 
