transition_dictionary = {
    '124': 'lie to sit transition',
    '421': 'sit to lie transition',
    '456': 'sit to stand transition',
    '654': 'stand to sit transition',
    '454': 'sit to stand effort',
    '121': 'lie to sit effort',
}


def transition_valid(input_states):
    try:
        transition = transition_dictionary[input_states]
        return transition
    except:
        return -1
    




    