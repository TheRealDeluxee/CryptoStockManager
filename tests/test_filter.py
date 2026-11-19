"""Duplicate filter: the same alarm is not sent twice in a row."""
import copy
import pytest

import func  # filter() and the threshold 'previous_alarm_change'
from classes import GroupedState  # holds 'alarm_prev'

@pytest.fixture(autouse=True)
def set_threshold():
    # every test gets a defined threshold
    old = getattr(func, "previous_alarm_change", None)
    func.previous_alarm_change = 1
    yield
    # restore the old value for the following tests
    if old is None:
        delattr(func, "previous_alarm_change")
    else:
        func.previous_alarm_change = old

@pytest.fixture
def state():
    return GroupedState()  # empty alarm_prev

def test_first_time_adds_all_keys(state):  # first alarms are always reported
    alarms = {
        "101": {"value": 1.0, "msg": "a", "priority": 0},
        "201": {"value": -2.0, "msg": "b", "priority": 1},
    }
    result = func.filter(alarms, state)

    assert result == alarms
    assert state.alarm_prev == alarms

def test_above_threshold_updates_and_reports_change(state):  # change above the threshold
    state.alarm_prev["101"] = {"value": 10.0, "msg": "old", "priority": 0}
    new = {"101": {"value": 11.1, "msg": "new", "priority": 1}}
    result = func.filter(new, state)

    assert result == new
    assert state.alarm_prev["101"] == new["101"]

def test_mixed_keys(state):  # below the threshold and a new alarm
    state.alarm_prev["101"] = {"value": 5.0, "msg": "oldA", "priority": 0}
    alarms = {
        "101": {"value": 5.2, "msg": "newA", "priority": 0},   # below the threshold (0.2)
        "102": {"value": 1.0, "msg": "newB", "priority": 0},   # new -> always reported
    }
    result = func.filter(alarms, state)

    assert result == {"102": alarms["102"]}
    assert state.alarm_prev["101"]["value"] == 5.0
    assert state.alarm_prev["102"] == alarms["102"]
