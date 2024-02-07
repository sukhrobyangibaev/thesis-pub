def get_win(radiant, radiant_win):
    if radiant and radiant_win:
        return 1
    elif radiant and not radiant_win:
        return 0
    elif not radiant and radiant_win:
        return 0
    else:
        return 1
