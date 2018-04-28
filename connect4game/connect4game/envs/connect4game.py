import math
import gym

from gym import spaces, logger
from gym.utils import seeding
import numpy as np

# action space 0-6
# obersationspace board, gamestate - win/lose/draw/pending
# reward : win 2 lose -1 draw 0 


class Connect4GameEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):

        # define constants and initialize values
        self.turnCounter_reset = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            2 * 2,
            np.finfo(np.float32).max,
            2 * 2,
            np.finfo(np.float32).max])

        # action_space: 0, 1, 2, 3, 4, 5, 6 (columns on board)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action_space: 0, 1, 2, 3, 4, 5, 6
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        print(action)

        # load state
        state = self.state
        turnCounter, selectedAction = state

        # update state values
        turnCounter = turnCounter + 1 # increment counter
        selectedAction = action

        # save state
        self.state = (turnCounter,selectedAction)

        # check if done
        done = turnCounter >= 49.0
        done = bool(done)

        if not done:
            if action == 0:
                reward = 1.0
            else:
                reward = -1.0
        elif self.steps_beyond_done is None:
            # Game ended
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        print(np.array(self.state), reward, done, {})
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.state[1] = None # Reset action
        self.state[0] = 0 # Reset counter
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        rows,cols = 7,7
        boardwidth = 500.0
        boardheight = 350.0

        slotdiameter = 40.0
        slotxmargin = (boardwidth-rows*slotdiameter) / rows
        slotymargin = (boardheight-cols*slotdiameter) / cols


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l = (screen_width - boardwidth)
            r = boardwidth
            t = boardheight + (screen_height - boardheight)/2
            b = (screen_height - boardheight)/2
            board = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            board.set_color(0,0,.8)
            self.viewer.add_geom(board)

            boardstate = np.zeros((rows, cols)) 
            boardstate[0][0] = 1
            boardstate[1][0] = 1
            boardstate[2][1] = 2
            boardstate[3][1] = 2
            boardstate[4][2] = 1

            for i in range(0, rows):
                for j in range(0, cols):
                    # print (boardstate[i,j])
                    slot = rendering.make_circle(slotdiameter/2)

                    if boardstate[i][j] == 0:
                        slot.set_color(1,1,1)
                    elif boardstate[i][j] == 1:
                        slot.set_color(0,0,0)
                    elif boardstate[i][j] == 2:
                        slot.set_color(.8,0,0)
                    elif boardstate[i][j] == 3:
                        slot.set_color(0,.8,0)

                    slottrans = rendering.Transform(translation=(l+slotdiameter/2 + i*slotdiameter + slotxmargin/2 + i*slotxmargin/2, 
                        t-slotdiameter/2 - j*slotdiameter - j*slotymargin/2 - slotymargin*2))
                    slot.add_attr(slottrans)
                    self.viewer.add_geom(slot)

        if self.state is None: return None

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()