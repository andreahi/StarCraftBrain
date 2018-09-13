import redis

from Actions import get_screen_acions, to_sc2_action

from redis_int.RedisUtil import recv_zipped_pickle, send_zipped_pickle


class SC2Game:
    TIMEOUT = 30
    def __init__(self):
        self.r = redis.StrictRedis(host='in.space', port=6379, db=0)

    def new_episode(self):
        data = recv_zipped_pickle(self.r, key="episode")
        self.id = data[0]
        #self.action_spec_functions = data[1] # not implemented because of serialization error


    def get_state(self):
        data = recv_zipped_pickle(self.r, key="from_agent" + self.id, timeout=self.TIMEOUT)
        if data[0] == 'finished':
            return True, [], data[2], data[3]

        return False, data[1], data[2], data[3]

    def make_action(self, action, xy, max_axis = 42):
        x = xy/max_axis
        y = xy%max_axis

        x = int(x) * 2
        y = y * 2
        args = []

        if action in get_screen_acions():
            if action == 13:
                args = [[x, y]]
            else:
                args = [[0],
                    [x, y]]

        action = to_sc2_action(action)
        send_zipped_pickle(self.r, [action, args], key="from_brain" + self.id)
        return recv_zipped_pickle(self.r, key="reward" + self.id, timeout=self.TIMEOUT)

