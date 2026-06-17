import subprocess
import shlex
import time
import os
import argparse

commands_ctgraph_ppo = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py baseline --max_steps 51200 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py baseline --max_steps 51200 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py baseline --max_steps 51200 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py baseline --max_steps 51200 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py baseline --max_steps 51200 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py baseline --max_steps 51200 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py baseline --max_steps 51200 --seed 92']
]

commands_minigrid_ppo = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py baseline --env_config_path ./env_configs/minigrid_object_remap_seed86.json  --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py baseline --env_config_path ./env_configs/minigrid_object_remap_seed87.json  --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py baseline --env_config_path ./env_configs/minigrid_object_remap_seed88.json  --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py baseline --env_config_path ./env_configs/minigrid_object_remap_seed89.json  --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py baseline --env_config_path ./env_configs/minigrid_object_remap_seed90.json  --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py baseline --env_config_path ./env_configs/minigrid_object_remap_seed91.json  --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py baseline --env_config_path ./env_configs/minigrid_object_remap_seed92.json  --max_steps 512_000 --seed 92 --disable_task_label_input']
]

# FOCCAL (CT-Graph)
commands_ctgraph_lc = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92']
]

commands_ctgraph_sc = [
    # MIG 1 (7-13)
    #['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    #['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    #['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    #['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    #['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    #['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    #['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92'],

    # MIG 2 (7-13)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 86'],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 87'],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 88'],
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 89'],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 90'],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 91'],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_ctgraph.py ll_supermask --new_task_mask linear_comb --max_steps 51200 --seed 92']
]

commands_minigrid = [
    # MIG 1 (7-13)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python train_minigrid.py ll_supermask --new_task_mask random --env_config_path ./env_configs/minigrid_object_remap_seed86.json --max_steps 512_000 --seed 86 --disable_task_label_input'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python train_minigrid.py ll_supermask --new_task_mask random --env_config_path ./env_configs/minigrid_object_remap_seed87.json --max_steps 512_000 --seed 87 --disable_task_label_input'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python train_minigrid.py ll_supermask --new_task_mask random --env_config_path ./env_configs/minigrid_object_remap_seed88.json --max_steps 512_000 --seed 88 --disable_task_label_input'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python train_minigrid.py ll_supermask --new_task_mask random --env_config_path ./env_configs/minigrid_object_remap_seed89.json --max_steps 512_000 --seed 89 --disable_task_label_input'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python train_minigrid.py ll_supermask --new_task_mask random --env_config_path ./env_configs/minigrid_object_remap_seed90.json --max_steps 512_000 --seed 90 --disable_task_label_input'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python train_minigrid.py ll_supermask --new_task_mask random --env_config_path ./env_configs/minigrid_object_remap_seed91.json --max_steps 512_000 --seed 91 --disable_task_label_input'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python train_minigrid.py ll_supermask --new_task_mask random --env_config_path ./env_configs/minigrid_object_remap_seed92.json --max_steps 512_000 --seed 92 --disable_task_label_input'],

    # MIG 2 (7-13)
    #['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed86.json --max_steps 512_000 --seed 86 --disable_task_label_input'],
    #['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed87.json --max_steps 512_000 --seed 87 --disable_task_label_input'],
    #['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed88.json --max_steps 512_000 --seed 88 --disable_task_label_input'],
    #['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed89.json --max_steps 512_000 --seed 89 --disable_task_label_input'],
    #['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed90.json --max_steps 512_000 --seed 90 --disable_task_label_input'],
    #['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed91.json --max_steps 512_000 --seed 91 --disable_task_label_input'],
    #['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python train_minigrid.py ll_supermask --new_task_mask linear_comb --env_config_path ./env_configs/minigrid_object_remap_seed92.json --max_steps 512_000 --seed 92 --disable_task_label_input']
]

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='indicate which experiment is being run for command selection', type=str, default='ctgraph')
parser.add_argument('--exp', help='', type=str, default='')
args = parser.parse_args()
commands = None
if args.env == 'ctgraph_lc':
    commands = commands_ctgraph_lc
elif args.env == 'minigrid':
    commands = commands_minigrid
elif args.env == 'ctgraph_ppo':
    commands = commands_ctgraph_ppo
elif args.env == 'ctgraph_sc':
    commands = commands_ctgraph_sc
elif args.env == 'minigrid_ppo':
    commands = commands_minigrid_ppo
else:
    raise ValueError(f'no commands have been setup for --exp {args.env}')


env = dict(os.environ)

path_header = args.env
if len(args.exp) > 0:
    path_header = args.exp

# Run the commands in seperate terminals
processes = []
for command in commands:
    print(f"{command[0]}, {command[1]} -p {path_header}")
    env['CUDA_VISIBLE_DEVICES'] = command[0]
    process = subprocess.Popen(shlex.split(command[1] + f' -p {path_header}'), env=env)
    processes.append(process)
    #time.sleep(5)

for process in processes:
    stdout, stderr = process.communicate()