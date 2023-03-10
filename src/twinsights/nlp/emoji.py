import json
import os
import re
from collections import namedtuple
from typing import Dict

EmojiEntry = namedtuple('EmojiEntry', ['name', 'group', 'sub_group'])
__EmojiDict = None


def get_emoji_dict() -> Dict[str, EmojiEntry]:
    global __EmojiDict
    if __EmojiDict is None:
        __EmojiDict = dict()
        with open(os.path.dirname(os.path.abspath(__file__)) + '/emoji.json') as fp:
            __EmojiDict = dict()
            for k, v in json.load(fp).items():
                emoji = EmojiEntry(name=v['name'], group=v['group'], sub_group=v['sub_group'])
                __EmojiDict[k] = emoji
    return __EmojiDict


def main():
    emoji_entries = []
    emoji_dict = dict()
    E_regex = re.compile(r' ?E\d+\.\d+ ')
    with open('/emoji-test.txt') as fp:
        for line in fp.read().splitlines()[32:]:
            if line == '# Status Counts':  # the last line in the document
                break

            if 'subtotal:' in line or \
                    len(line.strip()) == 0:  # these are lines showing statistics about each group, not needed
                continue

            if line.startswith('#'):  # these lines contain group and/or sub-group names
                if '# group:' in line:
                    group = line.split(':')[-1].strip()
                if '# subgroup:' in line:
                    subgroup = line.split(':')[-1].strip()

            if group == 'Component':  # skin tones, and hair types, skip, as mentioned above
                continue

            if re.search('^[0-9A-F]{3,}', line):  # if the line starts with a hexadecimal number (an emoji code point)
                # here we define all the elements that will go into emoji entries
                codepoint = line.split(';')[0].strip()  # in some cases it is one and in others multiple code points
                status = line.split(';')[-1].split()[
                    0].strip()  # status: fully-qualified, minimally-qualified, unqualified
                if line[-1] == '#':
                    # The special case where the emoji is actually the hash sign "#". In this case manually assign the emoji
                    if 'fully-qualified' in line:
                        emoji = '#️⃣'
                    else:
                        emoji = '#⃣'  # they look the same, but are actually different
                else:  # the default case
                    emoji = line.split('#')[-1].split()[0].strip()  # the emoji character itself
                if line[-1] == '#':  # (the special case)
                    name = '#'
                else:  # extract the emoji name
                    split_hash = line.split('#')[1]
                    rm_capital_E = E_regex.split(split_hash)[1]
                    name = rm_capital_E
                emoji_dict[emoji] = {
                    "name": name,
                    "group": group,
                    "sub_group": subgroup
                }
        with open('emoji.json', 'w') as op:
            json.dump(emoji_dict, op)


if __name__ == '__main__':
    main()
