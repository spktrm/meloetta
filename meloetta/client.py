import json
import requests
import websockets


class Client:
    websocket = None
    address = None
    login_uri = None
    username = None
    password = None
    last_message = None
    last_challenge_time = 0

    @classmethod
    async def create(cls, username, password, address) -> "Client":
        self = Client()
        self.username = username
        self.password = password
        self.address = "ws://{}/showdown/websocket".format(address)
        self.websocket = await websockets.connect(self.address)
        self.login_uri = "https://play.pokemonshowdown.com/action.php"
        return self

    async def receive_message(self):
        message = await self.websocket.recv()
        return message

    async def send_message(self, room, message_list):
        message = room + "|" + "|".join(message_list)
        await self.websocket.send(message)
        self.last_message = message

    async def update_team(self, battle_format, team):
        if "random" in battle_format:
            message = ["/utm None"]
        else:
            message = ["/utm {}".format(team)]
        await self.send_message("", message)

    async def search_for_match(self, battle_format, team):
        await self.update_team(battle_format, team)
        message = ["/search {}".format(battle_format)]
        await self.send_message("", message)

    async def get_id_and_challstr(self):
        while True:
            message = await self.receive_message()
            split_message = message.split("|")
            if split_message[1] == "challstr":
                return split_message[2], split_message[3]

    async def login(self):
        client_id, challstr = await self.get_id_and_challstr()
        if self.password is not None:
            response = requests.post(
                self.login_uri,
                data={
                    "act": "login",
                    "name": self.username,
                    "pass": self.password,
                    "challstr": "|".join([client_id, challstr]),
                },
            )
            if response.status_code == 200:
                if self.password:
                    response_json = json.loads(response.text[1:])
                    if not response_json["actionsuccess"]:
                        raise ValueError("Could not log-in")

                    assertion = response_json.get("assertion")
                else:
                    assertion = response.text

        else:
            assertion = ""

        message = ["/trn " + self.username + ",0," + assertion]
        await self.send_message("", message)

    async def leave_battle(self, battle_tag):
        message = ["/leave {}".format(battle_tag)]
        await self.send_message("", message)

    async def join_room(self, room_name):
        message = "/join {}".format(room_name)
        await self.send_message("", [message])

    async def challenge_user(
        self, user_to_challenge: str, battle_format: str, team: str = "null"
    ):
        await self.update_team(battle_format, team)
        message = ["/challenge {},{}".format(user_to_challenge, battle_format)]
        await self.send_message("", message)

    async def accept_challenge(
        self, battle_format: str, team: str = "null", room_name: str = None
    ):
        if room_name is not None:
            await self.join_room(room_name)

        await self.update_team(battle_format, team)
        username = None
        while username is None:
            msg = await self.receive_message()
            split_msg = msg.split("|")
            if (
                len(split_msg) == 9
                and split_msg[1] == "pm"
                and split_msg[3].strip().replace("!", "").replace("‽", "")
                == self.username
                and split_msg[4].startswith("/challenge")
                and split_msg[5] == battle_format
            ):
                username = split_msg[2].strip()

        message = ["/accept " + username]
        await self.send_message("", message)
