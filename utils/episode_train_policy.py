class TrainingEvaluator:
    def __init__(self, training_episodes: int, test_episodes: int):
        """
        Initializes the TrainingEvaluator with specified episode counts for training and testing.

        Args:
            training_episodes (int): Number of consecutive episodes to run in training mode.
            test_episodes (int): Number of consecutive episodes to run in evaluation mode.

        Attributes:
            _counter (int): Internal counter tracking the number of episodes in the current mode.
            _in_train_mode (bool): Boolean flag indicating whether the current mode is training.
        """
        
        self.training_episodes = training_episodes
        self.test_episodes = test_episodes
        self._counter = 0
        self._in_train_mode = True

    def get_policy(self) -> bool:
        """Returns True if in training mode, False if in evaluation mode."""
        return self._in_train_mode

    def update(self):
        """
        Updates the internal state after completing an episode.

        Increments the episode counter and switches between training/evaluation modes
        based on the predefined thresholds (training_episodes/test_episodes).
        Resets the counter when a mode switch occurs.
        """
        
        self._counter += 1
        change_policy = False
        if self._in_train_mode and self._counter >= self.training_episodes:
            change_policy = True
        elif not self._in_train_mode and self._counter >= self.test_episodes:
            change_policy = True

        if change_policy:
            self._in_train_mode = not self._in_train_mode
            self._counter = 0

    def reset_counters(self):
        """Resets the internal episode counter to zero, allowing manual control of mode transitions."""
        self._counter = 0
