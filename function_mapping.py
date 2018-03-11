#common
Function.ui_func(6, "select_idle_worker", select_idle_worker, lambda obs: obs.player_common.idle_worker_count > 0),
Function.ui_func(2, "select_point", select_point),

#protos
Function.ability(70, "Build_Pylon_screen", cmd_screen, 881),
Function.ability(485, "Train_Probe_quick", cmd_quick, 1006),
Function.ability(69, "Build_PhotonCannon_screen", cmd_screen, 887),
Function.ability(55, "Build_Forge_screen", cmd_screen, 884),

#zerg
Function.ability(483, "Train_Overlord_quick", cmd_quick, 1344),
Function.ability(467, "Train_Drone_quick", cmd_quick, 1342),
Function.ability(504, "Train_Zergling_quick", cmd_quick, 1343),
Function.ability(84, "Build_SpawningPool_screen", cmd_screen, 1155),
Function.ui_func(9, "select_larva", select_larva,lambda obs: obs.player_common.larva_count > 0),
Function.ability(486, "Train_Queen_quick", cmd_quick, 1632),
Function.ability(85, "Build_SpineCrawler_screen", cmd_screen, 1166),


#common+
Function.ui_func(1, "move_camera", move_camera),
Function.ability(19, "Scan_Move_screen", cmd_screen, 19, 3674),
Function.ability(331, "Move_screen", cmd_screen, 16),

#zerg+
Function.ability(59, "Build_Hatchery_screen", cmd_screen, 1152),
Function.ui_func(7, "select_army", select_army,
                     lambda obs: obs.player_common.army_count > 0),

out["player"] = np.array([
    obs.player_common.player_id,
    obs.player_common.minerals,
    obs.player_common.vespene,
    obs.player_common.food_used,
    obs.player_common.food_cap,
    obs.player_common.food_army,
    obs.player_common.food_workers,
    obs.player_common.idle_worker_count,
    obs.player_common.army_count,
    obs.player_common.warp_gate_count,
    obs.player_common.larva_count,
], dtype=np.int32)
