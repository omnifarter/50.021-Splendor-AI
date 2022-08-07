class AbstractPlayer:
    """Abstract player class for simulation of game"""

    def make_move(self, game_state, possible_actions):
        """Returns an action taken by the player."""
        raise NotImplementedError("abstract")


class CommandLinePlayer(AbstractPlayer):

    def make_move(self, game_state, possible_actions):
        # TODO display game state
        # TODO have user select action
        raise NotImplementedError()
