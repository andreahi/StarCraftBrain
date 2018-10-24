action_map = {0:0, 1:6, 2:2, 3:483, 4:467, 5:504, 6:84, 7:9, 8:486, 9:85, 10:264, 11:52, 12:13, 13:1, 14:59}
#action_map = {0:0, 1:idle_worker, 2:select, 3:overlord, 4:drone, 5:504, 6:84, 7:larva, 8:queen, 9:85, 10:264, 11:52, 12:13, 13:move_camera, 14:hatchery}

action_map_reversed = {v: k for k, v in action_map.items()}

screen_actions = [2, 6, 9, 10, 11, 12, 13, 14]

def to_sc2_action(action):
    return action_map[action]

def to_local_action(action):
    return action_map_reversed[action]

def get_action_map():
    return action_map

def get_reversed_action_map():
    return action_map_reversed

def get_screen_acions():
    return screen_actions


'''
        Function.ability(84, "Build_SpawningPool_screen", cmd_screen, 1155),
    Function.ability(85, "Build_SpineCrawler_screen", cmd_screen, 1166),


'''