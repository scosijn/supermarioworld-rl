import retro


def playback(path):
    recording = retro.Movie(path)
    recording.step()
    env = retro.make(
        game=recording.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=recording.players
    )
    env.initial_state = recording.get_state()
    env.reset()
    while recording.step():
        actions = []
        for p in range(recording.players):
            for i in range(env.num_buttons):
                actions.append(recording.get_key(i, p))
        env.step(actions)
        env.render()
