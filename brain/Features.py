import numpy as np

from brain.Actions import to_local_action, get_reversed_action_map


def get_screen_unit_type(obs):
    return obs[3]['screen'][6]


def get_screen_unit_selected(obs):
    return obs[3]['screen'][7].flatten()

def get_player_data(obs):
    player_ = [
        obs[3]['player'][1]/4000.0,
        np.array(obs[3]['player'][3], dtype=np.int32),
        np.array(obs[3]['player'][4], dtype=np.float32),
        obs[3]['player'][5],
        obs[3]['player'][7],
        obs[3]['player'][8]
    ]
    return player_

def get_available_actions(obs):
    action_map_rev = get_reversed_action_map()
    available_action = np.empty(len(action_map_rev))
    available_action.fill(0)
    for e in obs.observation["available_actions"]:
        if e in action_map_rev:
            available_action[to_local_action(e)] = 1

    return available_action
