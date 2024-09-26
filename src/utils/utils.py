"""
Author: <Not Me>
Original Src of the content https://github.com/clembench/clembench/blob/main/backends/utils.py

Just copied the file to reuse utils and stay in sync with the actual game repo
"""

import copy
from functools import wraps
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def ensure_alternating_roles(messages: List[Dict], cull_system_message: bool = True) -> List[Dict]:
    """
    The messages format assumes alternating roles of user and assistant. This method checks, if this constraint
    is satisfied. If this is not the case and there are consecutive user or assistant messages,
    then these are merged into a single one.

    :param cull_system_message:
    :param messages: to be checked
    :return: a new messages object with the alternating roles ensured
    """
    _messages = copy.deepcopy(messages)

    if cull_system_message:
        if _messages[0]['role'] == "system" and not _messages[0]["content"]:
            del _messages[0]

    def is_same_role(msg1, msg2):
        return msg1["role"] == msg2["role"]

    delimiter = "\n\n"

    def join_content(msg1, msg2):
        return f"{msg1['content']}{delimiter}{msg2['content']}"

    if len(_messages) <= 1:
        return _messages

    def is_valid(idx):
        return idx < len(_messages)

    msg_idx = 1
    while is_valid(msg_idx):
        prev_message = _messages[msg_idx - 1]
        message = _messages[msg_idx]
        if is_same_role(prev_message, message):
            warn_msg = (f"Found consecutive role assignments. These will be merged into one:\n"
                        f"{prev_message}\n"
                        f"{message}")
            logger.warning(warn_msg)
            prev_message['content'] = join_content(prev_message, message)
            del _messages[msg_idx]
        else:
            msg_idx += 1

    return _messages


def ensure_messages_format(generate_response_fn):
    @wraps(generate_response_fn)
    def wrapped_fn(self, messages):
        _messages = ensure_alternating_roles(messages)
        return generate_response_fn(self, _messages)

    return wrapped_fn
