def get_side(radiant):
    return 'radiant' if radiant else 'dire'


def get_win(radiant, radiant_win):
    if radiant and radiant_win:
        return 1
    elif radiant and not radiant_win:
        return 0
    elif not radiant and radiant_win:
        return 0
    else:
        return 1


def get_kills(my_team, radiant_score, dire_score):
    if my_team == 'radiant':
        return (radiant_score, dire_score)
    else:
        return (dire_score, radiant_score)


def picks(team, picks_bans):
    radiant_picks = []
    dire_picks = []
    if picks_bans:
        for pick_ban in picks_bans:
            if pick_ban['is_pick']:
                if pick_ban['team'] == 0:
                    radiant_picks.append(pick_ban['hero_id'])
                else:
                    dire_picks.append(pick_ban['hero_id'])
        if team == 'radiant':
            return tuple(radiant_picks)
        else:
            return tuple(dire_picks)
    else:
        return 0, 0, 0, 0, 0


def get_my_heroes(my_team, picks_bans):
    return picks(my_team, picks_bans)


def get_oppose_team_heroes(my_team, picks_bans):
    if my_team == 'radiant':
        return picks('dire', picks_bans)
    else:
        return picks('radiant', picks_bans)


def get_gold_adv(my_team, radiant_gold_adv):
    try:
        if my_team == 'radiant':
            return radiant_gold_adv[-1]
        else:
            return - radiant_gold_adv[-1]
    except:
        return 0


def get_xp_adv(my_team, radiant_xp_adv):
    try:
        if my_team == 'radiant':
            return radiant_xp_adv[-1]
        else:
            return - radiant_xp_adv[-1]
    except:
        return 0


def get_tower_status(my_team, tower_status_radiant, tower_status_dire):
    if my_team == 'radiant':
        return tower_status_radiant, tower_status_dire
    else:
        return tower_status_dire, tower_status_radiant
