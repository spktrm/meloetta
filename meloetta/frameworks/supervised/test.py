import json


from meloetta.room import BattleRoom


class LocalWrapper:
    def __init__(self):
        self.started = True
        self.room = BattleRoom()

    def _recieve(self, data: str):
        if data.startswith(">"):
            try:
                nlIndex = data.index("\n")
            except IndexError:
                nlIndex = -1
            self.room._battle_tag = data[1:nlIndex]
            data = data[nlIndex + 1 :]
        if not data:
            return
        if self.started:
            self.room.recieve(data)
        else:
            if data.startswith("|init|"):
                self.started = True
                self.room.recieve(data)

        if any(prefix in data for prefix in {"|turn", "|teampreview"}):
            return True

        forceSwitch = self.room.get_js_attr("request?.forceSwitch")
        if "|request" not in data:
            return forceSwitch


def main():
    fpath = "meloetta/frameworks/supervised/data_small.json"
    with open(fpath, "r") as f:
        data = json.load(f)[3:]

    for match in data:
        wrapper = LocalWrapper()

        for line in match["log"].split("\n"):

            wrapper._recieve(line)


if __name__ == "__main__":
    main()
