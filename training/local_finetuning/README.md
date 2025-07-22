## Fine tuning guide for local LLAMA model, which will be the ultimate DebateLord model.
For now, fine tuning will be very simple. Simply prepare a bunch of blocks of data that train the model on how to respond to certain talking points and teach it to dunk on arguments.

Example:

[
    {
        "loser": "[Something the loser of this debate would bring up]",
        "winner": "[Something the winner of this debate would have responded with and dunked on the other to say]"
    },
    {
        "loser": "[Something the loser of this debate would bring up]",
        "winner": "[Something the winner of this debate would have responded with and dunked on the other to say]"
    },
    ...
]