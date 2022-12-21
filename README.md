# Meloetta

![meloetta](meloetta.png "meloetta")

```bash
pip install meloetta
```

Meloetta is a Pokémon Battle Client for Interacting with Pokémon Showdown written in Python. This project was born out of frustration for currently existing tools and their lack of dependency on Zarel's (PS Creator) existing code for handling client server interation.

The client works by reading messages from an asyncio stream and forwarding these to the javascript client battle object with PyMiniRacer.

As is, the necessary javascript source files come with the pip install. Whenever Pokemon Showdown Client releases an update, the client code can also be automatically updated from the source.

I have taken asyncio code directly from [pmariglia](https://github.com/pmariglia/showdown). All credit to them.

# Quickstart

See the code in `test.py`

# Manual Sync

```bash
git clone https://github.com/smogon/pokemon-showdown-client.git
```

Then run these commands in node (remeber to use the other backslash `\` on windows)

```bash
node pokemon-showdown-client/build-tools/build-indexes
node pokemon-showdown-client/build-tools/build-learnsets
```

Then finally run `extract.py` from the main directory. This will copy over the necessary source files.