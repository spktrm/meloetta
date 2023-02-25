/**
 * Text parser
 *
 * No dependencies
 * Optional dependency: BattleText
 *
 * @author Guangcong Luo <guangcongluo@gmail.com>
 * @license MIT
 */ var BattleTextParser = (function () {
    function BattleTextParser() {
        var perspective =
            arguments.length > 0 && arguments[0] !== undefined
                ? arguments[0]
                : "p1";
        this.p1 = "Player 1";
        this.p2 = "Player 2";
        this.p3 = "Player 3";
        this.p4 = "Player 4";
        this.perspective = void 0;
        this.gen = 9;
        this.turn = 0;
        this.curLineSection = "break";
        this.lowercaseRegExp = undefined;
        this.pokemonName = function (pokemon) {
            if (!pokemon) return "";
            if (!pokemon.startsWith("p"))
                return "???pokemon:" + pokemon + "???";
            if (pokemon.charAt(3) === ":") return pokemon.slice(4).trim();
            else if (pokemon.charAt(2) === ":") return pokemon.slice(3).trim();
            return "???pokemon:" + pokemon + "???";
        };
        this.perspective = perspective;
    }
    BattleTextParser.parseLine = function parseLine(line, noDefault) {
        if (!line.startsWith("|")) {
            return ["", line];
        }
        if (line === "|") {
            return ["done"];
        }
        var index = line.indexOf("|", 1);
        var cmd = line.slice(1, index);
        switch (cmd) {
            case "chatmsg":
            case "chatmsg-raw":
            case "raw":
            case "error":
            case "html":
            case "inactive":
            case "inactiveoff":
            case "warning":
            case "fieldhtml":
            case "controlshtml":
            case "bigerror":
            case "debug":
            case "tier":
            case "challstr":
            case "popup":
            case "":
                return [cmd, line.slice(index + 1)];
            case "c":
            case "chat":
            case "uhtml":
            case "uhtmlchange":
            case "queryresponse":
                var index2a = line.indexOf("|", index + 1);
                return [
                    cmd,
                    line.slice(index + 1, index2a),
                    line.slice(index2a + 1),
                ];
            case "c:":
            case "pm":
                var index2b = line.indexOf("|", index + 1);
                var index3b = line.indexOf("|", index2b + 1);
                return [
                    cmd,
                    line.slice(index + 1, index2b),
                    line.slice(index2b + 1, index3b),
                    line.slice(index3b + 1),
                ];
        }
        if (noDefault) return null;
        return line.slice(1).split("|");
    };
    BattleTextParser.parseBattleLine = function parseBattleLine(line) {
        var args = this.parseLine(line, true);
        if (args) return { args: args, kwArgs: {} };
        args = line.slice(1).split("|");
        var kwArgs = {};
        while (args.length > 1) {
            var lastArg = args[args.length - 1];
            if (lastArg.charAt(0) !== "[") break;
            var bracketPos = lastArg.indexOf("]");
            if (bracketPos <= 0) break;
            kwArgs[lastArg.slice(1, bracketPos)] =
                lastArg.slice(bracketPos + 1).trim() || ".";
            args.pop();
        }
        return BattleTextParser.upgradeArgs({ args: args, kwArgs: kwArgs });
    };
    BattleTextParser.parseNameParts = function parseNameParts(text) {
        var group = "";
        if (!/[A-Za-z0-9]/.test(text.charAt(0))) {
            group = text.charAt(0);
            text = text.slice(1);
        }
        var name = text;
        var atIndex = text.indexOf("@");
        var status = "";
        var away = false;
        if (atIndex > 0) {
            name = text.slice(0, atIndex);
            status = text.slice(atIndex + 1);
            if (status.startsWith("!")) {
                away = true;
                status = status.slice(1);
            }
        }
        return { group: group, name: name, away: away, status: status };
    };
    BattleTextParser.upgradeArgs = function upgradeArgs(_ref) {
        var args = _ref.args,
            kwArgs = _ref.kwArgs;
        switch (args[0]) {
            case "-activate": {
                if (
                    kwArgs.item ||
                    kwArgs.move ||
                    kwArgs.number ||
                    kwArgs.ability
                )
                    return { args: args, kwArgs: kwArgs };
                var _args = args,
                    pokemon = _args[1],
                    effect = _args[2],
                    arg3 = _args[3],
                    arg4 = _args[4];
                var target = kwArgs.of;
                var _id = BattleTextParser.effectId(effect);
                if (kwArgs.block)
                    return { args: ["-fail", pokemon], kwArgs: kwArgs };
                if (_id === "wonderguard")
                    return {
                        args: ["-immune", pokemon],
                        kwArgs: { from: "ability:Wonder Guard" },
                    };
                if (_id === "beatup" && kwArgs.of)
                    return { args: args, kwArgs: { name: kwArgs.of } };
                if (
                    [
                        "ingrain",
                        "quickguard",
                        "wideguard",
                        "craftyshield",
                        "matblock",
                        "protect",
                        "mist",
                        "safeguard",
                        "electricterrain",
                        "mistyterrain",
                        "psychicterrain",
                        "telepathy",
                        "stickyhold",
                        "suctioncups",
                        "aromaveil",
                        "flowerveil",
                        "sweetveil",
                        "disguise",
                        "safetygoggles",
                        "protectivepads",
                    ].includes(_id)
                ) {
                    if (target) {
                        kwArgs.of = pokemon;
                        return {
                            args: ["-block", target, effect, arg3],
                            kwArgs: kwArgs,
                        };
                    }
                    return {
                        args: ["-block", pokemon, effect, arg3],
                        kwArgs: kwArgs,
                    };
                }
                if (_id === "charge") {
                    return {
                        args: ["-singlemove", pokemon, effect],
                        kwArgs: { of: target },
                    };
                }
                if (
                    [
                        "bind",
                        "wrap",
                        "clamp",
                        "whirlpool",
                        "firespin",
                        "magmastorm",
                        "sandtomb",
                        "infestation",
                        "snaptrap",
                        "thundercage",
                        "trapped",
                    ].includes(_id)
                ) {
                    return {
                        args: ["-start", pokemon, effect],
                        kwArgs: { of: target },
                    };
                }
                if (_id === "fairylock") {
                    return { args: ["-fieldactivate", effect], kwArgs: {} };
                }
                if (_id === "symbiosis" || _id === "poltergeist") {
                    kwArgs.item = arg3;
                } else if (_id === "magnitude") {
                    kwArgs.number = arg3;
                } else if (
                    _id === "skillswap" ||
                    _id === "mummy" ||
                    _id === "lingeringaroma" ||
                    _id === "wanderingspirit"
                ) {
                    kwArgs.ability = arg3;
                    kwArgs.ability2 = arg4;
                } else if (
                    [
                        "eeriespell",
                        "gmaxdepletion",
                        "spite",
                        "grudge",
                        "forewarn",
                        "sketch",
                        "leppaberry",
                        "mysteryberry",
                    ].includes(_id)
                ) {
                    kwArgs.move = arg3;
                    kwArgs.number = arg4;
                }
                args = ["-activate", pokemon, effect, target || ""];
                break;
            }
            case "-fail": {
                if (kwArgs.from === "ability: Flower Veil") {
                    return {
                        args: ["-block", kwArgs.of, "ability: Flower Veil"],
                        kwArgs: { of: args[1] },
                    };
                }
                break;
            }
            case "-start": {
                if (kwArgs.from === "Protean" || kwArgs.from === "Color Change")
                    kwArgs.from = "ability:" + kwArgs.from;
                break;
            }
            case "move": {
                if (kwArgs.from === "Magic Bounce")
                    kwArgs.from = "ability:Magic Bounce";
                break;
            }
            case "cant": {
                var _args2 = args,
                    _pokemon2 = _args2[1],
                    _effect2 = _args2[2],
                    move = _args2[3];
                if (
                    [
                        "ability: Damp",
                        "ability: Dazzling",
                        "ability: Queenly Majesty",
                        "ability: Armor Tail",
                    ].includes(_effect2)
                ) {
                    args[0] = "-block";
                    return {
                        args: ["-block", _pokemon2, _effect2, move, kwArgs.of],
                        kwArgs: {},
                    };
                }
                break;
            }
            case "-nothing":
                return {
                    args: ["-activate", "", "move:Splash"],
                    kwArgs: kwArgs,
                };
        }
        return { args: args, kwArgs: kwArgs };
    };
    var _proto = BattleTextParser.prototype;
    _proto.extractMessage = function extractMessage(buf) {
        var out = "";
        for (
            var _i = 0, _buf$split = buf.split("\n");
            _i < _buf$split.length;
            _i++
        ) {
            var _line = _buf$split[_i];
            var _BattleTextParser$par = BattleTextParser.parseBattleLine(_line),
                args = _BattleTextParser$par.args,
                kwArgs = _BattleTextParser$par.kwArgs;
            out += this.parseArgs(args, kwArgs) || "";
        }
        return out;
    };
    _proto.fixLowercase = function fixLowercase(input) {
        if (this.lowercaseRegExp === undefined) {
            var prefixes = [
                "pokemon",
                "opposingPokemon",
                "team",
                "opposingTeam",
                "party",
                "opposingParty",
            ]
                .map(function (templateId) {
                    var template = BattleText["default"][templateId];
                    if (template.charAt(0) === template.charAt(0).toUpperCase())
                        return "";
                    var bracketIndex = template.indexOf("[");
                    if (bracketIndex >= 0)
                        return template.slice(0, bracketIndex);
                    return template;
                })
                .filter(function (prefix) {
                    return prefix;
                });
            if (prefixes.length) {
                var buf =
                    "((?:^|\n)(?:  |  \\(|\\[)?)(" +
                    prefixes.map(BattleTextParser.escapeRegExp).join("|") +
                    ")";
                this.lowercaseRegExp = new RegExp(buf, "g");
            } else {
                this.lowercaseRegExp = null;
            }
        }
        if (!this.lowercaseRegExp) return input;
        return input.replace(this.lowercaseRegExp, function (match, p1, p2) {
            return p1 + p2.charAt(0).toUpperCase() + p2.slice(1);
        });
    };
    BattleTextParser.escapeRegExp = function escapeRegExp(input) {
        return input.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
    };
    _proto.pokemon = function pokemon(_pokemon) {
        if (!_pokemon) return "";
        var side = _pokemon.slice(0, 2);
        if (!["p1", "p2", "p3", "p4"].includes(side))
            return "???pokemon:" + _pokemon + "???";
        var name = this.pokemonName(_pokemon);
        var isNear =
            side === this.perspective || side === BattleTextParser.allyID(side);
        var template =
            BattleText["default"][isNear ? "pokemon" : "opposingPokemon"];
        return template.replace("[NICKNAME]", name);
    };
    _proto.pokemonFull = function pokemonFull(pokemon, details) {
        var nickname = this.pokemonName(pokemon);

        var species = details.split(",")[0];
        if (nickname === species)
            return [pokemon.slice(0, 2), "**" + species + "**"];
        return [pokemon.slice(0, 2), nickname + " (**" + species + "**)"];
    };
    _proto.trainer = function trainer(side) {
        side = side.slice(0, 2);
        if (side === "p1") return this.p1;
        if (side === "p2") return this.p2;
        if (side === "p3") return this.p3;
        if (side === "p4") return this.p4;
        return "???side:" + side + "???";
    };
    BattleTextParser.allyID = function allyID(sideid) {
        if (sideid === "p1") return "p3";
        if (sideid === "p2") return "p4";
        if (sideid === "p3") return "p1";
        if (sideid === "p4") return "p2";
        return "";
    };
    _proto.team = function team(side) {
        var isFar =
            arguments.length > 1 && arguments[1] !== undefined
                ? arguments[1]
                : false;
        side = side.slice(0, 2);
        if (
            side === this.perspective ||
            side === BattleTextParser.allyID(side)
        ) {
            return !isFar
                ? BattleText["default"].team
                : BattleText["default"].opposingTeam;
        }
        return isFar
            ? BattleText["default"].team
            : BattleText["default"].opposingTeam;
    };
    _proto.own = function own(side) {
        side = side.slice(0, 2);
        if (side === this.perspective) {
            return "OWN";
        }
        return "";
    };
    _proto.party = function party(side) {
        side = side.slice(0, 2);
        if (
            side === this.perspective ||
            side === BattleTextParser.allyID(side)
        ) {
            return BattleText["default"].party;
        }
        return BattleText["default"].opposingParty;
    };
    BattleTextParser.effectId = function effectId(effect) {
        if (!effect) return "";
        if (effect.startsWith("item:") || effect.startsWith("move:")) {
            effect = effect.slice(5);
        } else if (effect.startsWith("ability:")) {
            effect = effect.slice(8);
        }
        return toID(effect);
    };
    _proto.effect = function effect(_effect) {
        if (!_effect) return "";
        if (_effect.startsWith("item:") || _effect.startsWith("move:")) {
            _effect = _effect.slice(5);
        } else if (_effect.startsWith("ability:")) {
            _effect = _effect.slice(8);
        }
        return _effect.trim();
    };
    _proto.template = function template(type) {
        for (
            var _len = arguments.length,
                namespaces = new Array(_len > 1 ? _len - 1 : 0),
                _key = 1;
            _key < _len;
            _key++
        ) {
            namespaces[_key - 1] = arguments[_key];
        }
        for (var _i2 = 0; _i2 < namespaces.length; _i2++) {
            var namespace = namespaces[_i2];
            if (!namespace) continue;
            if (namespace === "OWN") {
                return BattleText["default"][type + "Own"] + "\n";
            }
            if (namespace === "NODEFAULT") {
                return "";
            }
            var _id2 = BattleTextParser.effectId(namespace);
            if (BattleText[_id2] && type in BattleText[_id2]) {
                if (BattleText[_id2][type].charAt(1) === ".")
                    type = BattleText[_id2][type].slice(2);
                if (BattleText[_id2][type].charAt(0) === "#")
                    _id2 = BattleText[_id2][type].slice(1);
                if (!BattleText[_id2][type]) return "";
                return BattleText[_id2][type] + "\n";
            }
        }
        if (!BattleText["default"][type]) return "";
        return BattleText["default"][type] + "\n";
    };
    _proto.maybeAbility = function maybeAbility(effect, holder) {
        if (!effect) return "";
        if (!effect.startsWith("ability:")) return "";
        return this.ability(effect.slice(8).trim(), holder);
    };
    _proto.ability = function ability(name, holder) {
        if (!name) return "";
        return (
            BattleText["default"].abilityActivation
                .replace("[POKEMON]", this.pokemon(holder))
                .replace("[ABILITY]", this.effect(name)) + "\n"
        );
    };
    BattleTextParser.stat = function stat(_stat) {
        var entry = BattleText[_stat || "stats"];
        if (!entry || !entry.statName) return "???stat:" + _stat + "???";
        return entry.statName;
    };
    _proto.lineSection = function lineSection(args, kwArgs) {
        var cmd = args[0];
        switch (cmd) {
            case "done":
            case "turn":
                return "break";
            case "move":
            case "cant":
            case "switch":
            case "drag":
            case "upkeep":
            case "start":
            case "-mega":
            case "-candynamax":
            case "-terastallize":
                return "major";
            case "switchout":
            case "faint":
                return "preMajor";
            case "-zpower":
                return "postMajor";
            case "-damage": {
                var _id3 = BattleTextParser.effectId(kwArgs.from);
                if (_id3 === "confusion") return "major";
                return "postMajor";
            }
            case "-curestatus": {
                var _id4 = BattleTextParser.effectId(kwArgs.from);
                if (_id4 === "naturalcure") return "preMajor";
                return "postMajor";
            }
            case "-start": {
                var _id5 = BattleTextParser.effectId(kwArgs.from);
                if (_id5 === "protean") return "preMajor";
                return "postMajor";
            }
            case "-activate": {
                var _id6 = BattleTextParser.effectId(args[2]);
                if (_id6 === "confusion" || _id6 === "attract")
                    return "preMajor";
                return "postMajor";
            }
        }

        return cmd.charAt(0) === "-" ? "postMajor" : "";
    };
    _proto.sectionBreak = function sectionBreak(args, kwArgs) {
        var prevSection = this.curLineSection;
        var curSection = this.lineSection(args, kwArgs);
        if (!curSection) return false;
        this.curLineSection = curSection;
        switch (curSection) {
            case "break":
                if (prevSection !== "break") return true;
                return false;
            case "preMajor":
            case "major":
                if (prevSection === "postMajor" || prevSection === "major")
                    return true;
                return false;
            case "postMajor":
                return false;
        }
    };
    _proto.parseArgs = function parseArgs(args, kwArgs, noSectionBreak) {
        var buf =
            !noSectionBreak && this.sectionBreak(args, kwArgs) ? "\n" : "";
        return buf + this.fixLowercase(this.parseArgsInner(args, kwArgs) || "");
    };
    _proto.parseArgsInner = function parseArgsInner(args, kwArgs) {
        var cmd = args[0];
        switch (cmd) {
            case "player": {
                var side = args[1],
                    name = args[2];
                if (side === "p1" && name) {
                    this.p1 = name;
                } else if (side === "p2" && name) {
                    this.p2 = name;
                } else if (side === "p3" && name) {
                    this.p3 = name;
                } else if (side === "p4" && name) {
                    this.p4 = name;
                }
                return "";
            }

            case "gen": {
                var num = args[1];
                this.gen = parseInt(num, 10);
                return "";
            }

            case "turn": {
                var _num = args[1];
                this.turn = Number.parseInt(_num, 10);
                return this.template("turn").replace("[NUMBER]", _num) + "\n";
            }

            case "start": {
                return this.template("startBattle")
                    .replace("[TRAINER]", this.p1)
                    .replace("[TRAINER]", this.p2);
            }

            case "win":
            case "tie": {
                var _name = args[1];
                if (cmd === "tie" || !_name) {
                    return this.template("tieBattle")
                        .replace("[TRAINER]", this.p1)
                        .replace("[TRAINER]", this.p2);
                }
                return this.template("winBattle").replace("[TRAINER]", _name);
            }

            case "switch": {
                var pokemon = args[1],
                    details = args[2];
                var _this$pokemonFull = this.pokemonFull(pokemon, details),
                    _side = _this$pokemonFull[0],
                    fullname = _this$pokemonFull[1];
                var template = this.template("switchIn", this.own(_side));
                return template
                    .replace("[TRAINER]", this.trainer(_side))
                    .replace("[FULLNAME]", fullname);
            }

            case "drag": {
                var _pokemon3 = args[1],
                    _details = args[2];
                var _this$pokemonFull2 = this.pokemonFull(_pokemon3, _details),
                    _side2 = _this$pokemonFull2[0],
                    _fullname = _this$pokemonFull2[1];
                var _template = this.template("drag");
                return _template
                    .replace("[TRAINER]", this.trainer(_side2))
                    .replace("[FULLNAME]", _fullname);
            }

            case "detailschange":
            case "-transform":
            case "-formechange": {
                var _pokemon4 = args[1],
                    arg2 = args[2],
                    arg3 = args[3];
                var newSpecies = "";
                switch (cmd) {
                    case "detailschange":
                        newSpecies = arg2.split(",")[0].trim();
                        break;
                    case "-transform":
                        newSpecies = arg3;
                        break;
                    case "-formechange":
                        newSpecies = arg2;
                        break;
                }

                var newSpeciesId = toID(newSpecies);
                var _id7 = "";
                var _templateName = "transform";
                if (cmd !== "-transform") {
                    switch (newSpeciesId) {
                        case "greninjaash":
                            _id7 = "battlebond";
                            break;
                        case "mimikyubusted":
                            _id7 = "disguise";
                            break;
                        case "zygardecomplete":
                            _id7 = "powerconstruct";
                            break;
                        case "necrozmaultra":
                            _id7 = "ultranecroziumz";
                            break;
                        case "darmanitanzen":
                            _id7 = "zenmode";
                            break;
                        case "darmanitan":
                            _id7 = "zenmode";
                            _templateName = "transformEnd";
                            break;
                        case "darmanitangalarzen":
                            _id7 = "zenmode";
                            break;
                        case "darmanitangalar":
                            _id7 = "zenmode";
                            _templateName = "transformEnd";
                            break;
                        case "aegislashblade":
                            _id7 = "stancechange";
                            break;
                        case "aegislash":
                            _id7 = "stancechange";
                            _templateName = "transformEnd";
                            break;
                        case "wishiwashischool":
                            _id7 = "schooling";
                            break;
                        case "wishiwashi":
                            _id7 = "schooling";
                            _templateName = "transformEnd";
                            break;
                        case "miniormeteor":
                            _id7 = "shieldsdown";
                            break;
                        case "minior":
                            _id7 = "shieldsdown";
                            _templateName = "transformEnd";
                            break;
                        case "eiscuenoice":
                            _id7 = "iceface";
                            break;
                        case "eiscue":
                            _id7 = "iceface";
                            _templateName = "transformEnd";
                            break;
                    }
                } else if (newSpecies) {
                    _id7 = "transform";
                }
                var _template2 = this.template(
                    _templateName,
                    _id7,
                    kwArgs.msg ? "" : "NODEFAULT"
                );

                var line1 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon4
                );

                return (
                    line1 +
                    _template2
                        .replace("[POKEMON]", this.pokemon(_pokemon4))
                        .replace("[SPECIES]", newSpecies)
                );
            }

            case "switchout": {
                var _pokemon5 = args[1];
                var _side3 = _pokemon5.slice(0, 2);
                var _template3 = this.template(
                    "switchOut",
                    kwArgs.from,
                    this.own(_side3)
                );

                return _template3
                    .replace("[TRAINER]", this.trainer(_side3))
                    .replace("[NICKNAME]", this.pokemonName(_pokemon5))
                    .replace("[POKEMON]", this.pokemon(_pokemon5));
            }

            case "faint": {
                var _pokemon6 = args[1];
                var _template4 = this.template("faint");
                return _template4.replace("[POKEMON]", this.pokemon(_pokemon6));
            }

            case "swap": {
                var _pokemon7 = args[1],
                    target = args[2];
                if (!target || !isNaN(Number(target))) {
                    var _template6 = this.template("swapCenter");
                    return _template6.replace(
                        "[POKEMON]",
                        this.pokemon(_pokemon7)
                    );
                }
                var _template5 = this.template("swap");
                return _template5
                    .replace("[POKEMON]", this.pokemon(_pokemon7))
                    .replace("[TARGET]", this.pokemon(target));
            }

            case "move": {
                var _pokemon8 = args[1],
                    move = args[2];
                var _line2 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon8
                );

                if (kwArgs.zeffect) {
                    _line2 = this.template("zEffect").replace(
                        "[POKEMON]",
                        this.pokemon(_pokemon8)
                    );
                }
                var _template7 = this.template("move", kwArgs.from);
                return (
                    _line2 +
                    _template7
                        .replace("[POKEMON]", this.pokemon(_pokemon8))
                        .replace("[MOVE]", move)
                );
            }

            case "cant": {
                var _pokemon9 = args[1],
                    effect = args[2],
                    _move = args[3];
                var _template8 =
                    this.template("cant", effect, "NODEFAULT") ||
                    this.template(_move ? "cant" : "cantNoMove");
                var _line3 = this.maybeAbility(effect, kwArgs.of || _pokemon9);
                return (
                    _line3 +
                    _template8
                        .replace("[POKEMON]", this.pokemon(_pokemon9))
                        .replace("[MOVE]", _move)
                );
            }

            case "-candynamax": {
                var _side4 = args[1];
                var own = this.own(_side4);
                var _template9 = "";
                if (this.turn === 1) {
                    if (own) _template9 = this.template("canDynamax", own);
                } else {
                    _template9 = this.template("canDynamax", own);
                }
                return _template9.replace("[TRAINER]", this.trainer(_side4));
            }

            case "message": {
                var message = args[1];
                return "" + message + "\n";
            }

            case "-start": {
                var _kwArgs$from;
                var _pokemon10 = args[1],
                    _effect3 = args[2],
                    _arg = args[3];
                var _line4 =
                    this.maybeAbility(_effect3, _pokemon10) ||
                    this.maybeAbility(kwArgs.from, kwArgs.of || _pokemon10);
                var _id8 = BattleTextParser.effectId(_effect3);
                if (_id8 === "typechange") {
                    var _template11 = this.template("typeChange", kwArgs.from);
                    return (
                        _line4 +
                        _template11
                            .replace("[POKEMON]", this.pokemon(_pokemon10))
                            .replace("[TYPE]", _arg)
                            .replace("[SOURCE]", this.pokemon(kwArgs.of))
                    );
                }
                if (_id8 === "typeadd") {
                    var _template12 = this.template("typeAdd", kwArgs.from);
                    return (
                        _line4 +
                        _template12
                            .replace("[POKEMON]", this.pokemon(_pokemon10))
                            .replace("[TYPE]", _arg)
                    );
                }
                if (_id8.startsWith("stockpile")) {
                    var _num2 = _id8.slice(9);
                    var _template13 = this.template("start", "stockpile");
                    return (
                        _line4 +
                        _template13
                            .replace("[POKEMON]", this.pokemon(_pokemon10))
                            .replace("[NUMBER]", _num2)
                    );
                }
                if (_id8.startsWith("perish")) {
                    var _num3 = _id8.slice(6);
                    var _template14 = this.template("activate", "perishsong");
                    return (
                        _line4 +
                        _template14
                            .replace("[POKEMON]", this.pokemon(_pokemon10))
                            .replace("[NUMBER]", _num3)
                    );
                }
                if (
                    _id8.startsWith("protosynthesis") ||
                    _id8.startsWith("quarkdrive")
                ) {
                    var stat = _id8.slice(-3);
                    var _template15 = this.template(
                        "start",
                        _id8.slice(0, _id8.length - 3)
                    );

                    return (
                        _line4 +
                        _template15
                            .replace("[POKEMON]", this.pokemon(_pokemon10))
                            .replace("[STAT]", BattleTextParser.stat(stat))
                    );
                }
                var templateId = "start";
                if (kwArgs.already) templateId = "alreadyStarted";
                if (kwArgs.fatigue) templateId = "startFromFatigue";
                if (kwArgs.zeffect) templateId = "startFromZEffect";
                if (kwArgs.damage) templateId = "activate";
                if (kwArgs.block) templateId = "block";
                if (kwArgs.upkeep) templateId = "upkeep";
                if (_id8 === "mist" && this.gen <= 2)
                    templateId = "startGen" + this.gen;
                if (_id8 === "reflect" || _id8 === "lightscreen")
                    templateId = "startGen1";
                if (
                    templateId === "start" &&
                    (_kwArgs$from = kwArgs.from) != null &&
                    _kwArgs$from.startsWith("item:")
                ) {
                    templateId += "FromItem";
                }
                var _template10 = this.template(
                    templateId,
                    kwArgs.from,
                    _effect3
                );
                return (
                    _line4 +
                    _template10
                        .replace("[POKEMON]", this.pokemon(_pokemon10))
                        .replace("[EFFECT]", this.effect(_effect3))
                        .replace("[MOVE]", _arg)
                        .replace("[SOURCE]", this.pokemon(kwArgs.of))
                        .replace("[ITEM]", this.effect(kwArgs.from))
                );
            }

            case "-end": {
                var _kwArgs$from2;
                var _pokemon11 = args[1],
                    _effect4 = args[2];
                var _line5 =
                    this.maybeAbility(_effect4, _pokemon11) ||
                    this.maybeAbility(kwArgs.from, kwArgs.of || _pokemon11);
                var _id9 = BattleTextParser.effectId(_effect4);
                if (_id9 === "doomdesire" || _id9 === "futuresight") {
                    var _template17 = this.template("activate", _effect4);
                    return (
                        _line5 +
                        _template17.replace(
                            "[TARGET]",
                            this.pokemon(_pokemon11)
                        )
                    );
                }
                var _templateId = "end";
                var _template16 = "";
                if (
                    (_kwArgs$from2 = kwArgs.from) != null &&
                    _kwArgs$from2.startsWith("item:")
                ) {
                    _template16 = this.template("endFromItem", _effect4);
                }
                if (!_template16)
                    _template16 = this.template(_templateId, _effect4);
                return (
                    _line5 +
                    _template16
                        .replace("[POKEMON]", this.pokemon(_pokemon11))
                        .replace("[EFFECT]", this.effect(_effect4))
                        .replace("[SOURCE]", this.pokemon(kwArgs.of))
                        .replace("[ITEM]", this.effect(kwArgs.from))
                );
            }

            case "-ability": {
                var _pokemon12 = args[1],
                    ability = args[2],
                    oldAbility = args[3],
                    arg4 = args[4];
                var _line6 = "";
                if (
                    oldAbility &&
                    (oldAbility.startsWith("p1") ||
                        oldAbility.startsWith("p2") ||
                        oldAbility === "boost")
                ) {
                    arg4 = oldAbility;
                    oldAbility = "";
                }
                if (oldAbility) _line6 += this.ability(oldAbility, _pokemon12);
                _line6 += this.ability(ability, _pokemon12);
                if (kwArgs.fail) {
                    var _template19 = this.template("block", kwArgs.from);
                    return _line6 + _template19;
                }
                if (kwArgs.from) {
                    _line6 =
                        this.maybeAbility(kwArgs.from, _pokemon12) + _line6;
                    var _template20 = this.template(
                        "changeAbility",
                        kwArgs.from
                    );

                    return (
                        _line6 +
                        _template20
                            .replace("[POKEMON]", this.pokemon(_pokemon12))
                            .replace("[ABILITY]", this.effect(ability))
                            .replace("[SOURCE]", this.pokemon(kwArgs.of))
                    );
                }
                var _id10 = BattleTextParser.effectId(ability);
                if (_id10 === "unnerve") {
                    var _template21 = this.template("start", ability);
                    return (
                        _line6 +
                        _template21.replace(
                            "[TEAM]",
                            this.team(_pokemon12.slice(0, 2), true)
                        )
                    );
                }
                var _templateId2 = "start";
                if (_id10 === "anticipation" || _id10 === "sturdy")
                    _templateId2 = "activate";
                var _template18 = this.template(
                    _templateId2,
                    ability,
                    "NODEFAULT"
                );

                return (
                    _line6 +
                    _template18.replace("[POKEMON]", this.pokemon(_pokemon12))
                );
            }

            case "-endability": {
                var _pokemon13 = args[1],
                    _ability = args[2];
                if (_ability) return this.ability(_ability, _pokemon13);
                var _line7 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon13
                );

                var _template22 = this.template("start", "Gastro Acid");
                return (
                    _line7 +
                    _template22.replace("[POKEMON]", this.pokemon(_pokemon13))
                );
            }

            case "-item": {
                var _pokemon14 = args[1],
                    item = args[2];
                var _id11 = BattleTextParser.effectId(kwArgs.from);
                var _target = "";
                if (["magician", "pickpocket"].includes(_id11)) {
                    var _ref2 = [kwArgs.of, ""];
                    _target = _ref2[0];
                    kwArgs.of = _ref2[1];
                }
                var _line8 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon14
                );

                if (
                    [
                        "thief",
                        "covet",
                        "bestow",
                        "magician",
                        "pickpocket",
                    ].includes(_id11)
                ) {
                    var _template24 = this.template("takeItem", kwArgs.from);
                    return (
                        _line8 +
                        _template24
                            .replace("[POKEMON]", this.pokemon(_pokemon14))
                            .replace("[ITEM]", this.effect(item))
                            .replace(
                                "[SOURCE]",
                                this.pokemon(_target || kwArgs.of)
                            )
                    );
                }
                if (_id11 === "frisk") {
                    var hasTarget =
                        kwArgs.of && _pokemon14 && kwArgs.of !== _pokemon14;
                    var _template25 = this.template(
                        hasTarget ? "activate" : "activateNoTarget",
                        "Frisk"
                    );

                    return (
                        _line8 +
                        _template25
                            .replace("[POKEMON]", this.pokemon(kwArgs.of))
                            .replace("[ITEM]", this.effect(item))
                            .replace("[TARGET]", this.pokemon(_pokemon14))
                    );
                }
                if (kwArgs.from) {
                    var _template26 = this.template("addItem", kwArgs.from);
                    return (
                        _line8 +
                        _template26
                            .replace("[POKEMON]", this.pokemon(_pokemon14))
                            .replace("[ITEM]", this.effect(item))
                    );
                }
                var _template23 = this.template("start", item, "NODEFAULT");
                return (
                    _line8 +
                    _template23.replace("[POKEMON]", this.pokemon(_pokemon14))
                );
            }

            case "-enditem": {
                var _pokemon15 = args[1],
                    _item = args[2];
                var _line9 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon15
                );

                if (kwArgs.eat) {
                    var _template28 = this.template("eatItem", kwArgs.from);
                    return (
                        _line9 +
                        _template28
                            .replace("[POKEMON]", this.pokemon(_pokemon15))
                            .replace("[ITEM]", this.effect(_item))
                    );
                }
                var _id12 = BattleTextParser.effectId(kwArgs.from);
                if (_id12 === "gem") {
                    var _template29 = this.template("useGem", _item);
                    return (
                        _line9 +
                        _template29
                            .replace("[POKEMON]", this.pokemon(_pokemon15))
                            .replace("[ITEM]", this.effect(_item))
                            .replace("[MOVE]", kwArgs.move)
                    );
                }
                if (_id12 === "stealeat") {
                    var _template30 = this.template("removeItem", "Bug Bite");
                    return (
                        _line9 +
                        _template30
                            .replace("[SOURCE]", this.pokemon(kwArgs.of))
                            .replace("[ITEM]", this.effect(_item))
                    );
                }
                if (kwArgs.from) {
                    var _template31 = this.template("removeItem", kwArgs.from);
                    return (
                        _line9 +
                        _template31
                            .replace("[POKEMON]", this.pokemon(_pokemon15))
                            .replace("[ITEM]", this.effect(_item))
                            .replace("[SOURCE]", this.pokemon(kwArgs.of))
                    );
                }
                if (kwArgs.weaken) {
                    var _template32 = this.template("activateWeaken");
                    return (
                        _line9 +
                        _template32
                            .replace("[POKEMON]", this.pokemon(_pokemon15))
                            .replace("[ITEM]", this.effect(_item))
                    );
                }
                var _template27 = this.template("end", _item, "NODEFAULT");
                if (!_template27)
                    _template27 = this.template("activateItem").replace(
                        "[ITEM]",
                        this.effect(_item)
                    );

                return (
                    _line9 +
                    _template27
                        .replace("[POKEMON]", this.pokemon(_pokemon15))
                        .replace("[TARGET]", this.pokemon(kwArgs.of))
                );
            }

            case "-status": {
                var _pokemon16 = args[1],
                    status = args[2];
                var _line10 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon16
                );

                if (BattleTextParser.effectId(kwArgs.from) === "rest") {
                    var _template34 = this.template("startFromRest", status);
                    return (
                        _line10 +
                        _template34.replace(
                            "[POKEMON]",
                            this.pokemon(_pokemon16)
                        )
                    );
                }
                var _template33 = this.template("start", status);
                return (
                    _line10 +
                    _template33.replace("[POKEMON]", this.pokemon(_pokemon16))
                );
            }

            case "-curestatus": {
                var _kwArgs$from3;
                var _pokemon17 = args[1],
                    _status = args[2];
                if (BattleTextParser.effectId(kwArgs.from) === "naturalcure") {
                    var _template36 = this.template("activate", kwArgs.from);
                    return _template36.replace(
                        "[POKEMON]",
                        this.pokemon(_pokemon17)
                    );
                }
                var _line11 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon17
                );

                if (
                    (_kwArgs$from3 = kwArgs.from) != null &&
                    _kwArgs$from3.startsWith("item:")
                ) {
                    var _template37 = this.template("endFromItem", _status);
                    return (
                        _line11 +
                        _template37
                            .replace("[POKEMON]", this.pokemon(_pokemon17))
                            .replace("[ITEM]", this.effect(kwArgs.from))
                    );
                }
                if (kwArgs.thaw) {
                    var _template38 = this.template("endFromMove", _status);
                    return (
                        _line11 +
                        _template38
                            .replace("[POKEMON]", this.pokemon(_pokemon17))
                            .replace("[MOVE]", this.effect(kwArgs.from))
                    );
                }
                var _template35 = this.template("end", _status, "NODEFAULT");
                if (!_template35)
                    _template35 = this.template("end").replace(
                        "[EFFECT]",
                        _status
                    );
                return (
                    _line11 +
                    _template35.replace("[POKEMON]", this.pokemon(_pokemon17))
                );
            }

            case "-cureteam": {
                return this.template("activate", kwArgs.from);
            }

            case "-singleturn":
            case "-singlemove": {
                var _pokemon18 = args[1],
                    _effect5 = args[2];
                var _line12 =
                    this.maybeAbility(_effect5, kwArgs.of || _pokemon18) ||
                    this.maybeAbility(kwArgs.from, kwArgs.of || _pokemon18);
                var _id13 = BattleTextParser.effectId(_effect5);
                if (_id13 === "instruct") {
                    var _template40 = this.template("activate", _effect5);
                    return (
                        _line12 +
                        _template40
                            .replace("[POKEMON]", this.pokemon(kwArgs.of))
                            .replace("[TARGET]", this.pokemon(_pokemon18))
                    );
                }
                var _template39 = this.template("start", _effect5, "NODEFAULT");
                if (!_template39)
                    _template39 = this.template("start").replace(
                        "[EFFECT]",
                        this.effect(_effect5)
                    );

                return (
                    _line12 +
                    _template39
                        .replace("[POKEMON]", this.pokemon(_pokemon18))
                        .replace("[SOURCE]", this.pokemon(kwArgs.of))
                        .replace("[TEAM]", this.team(_pokemon18.slice(0, 2)))
                );
            }

            case "-sidestart": {
                var _side5 = args[1],
                    _effect6 = args[2];
                var _template41 = this.template("start", _effect6, "NODEFAULT");
                if (!_template41)
                    _template41 = this.template("startTeamEffect").replace(
                        "[EFFECT]",
                        this.effect(_effect6)
                    );

                return _template41
                    .replace("[TEAM]", this.team(_side5))
                    .replace("[PARTY]", this.party(_side5));
            }

            case "-sideend": {
                var _side6 = args[1],
                    _effect7 = args[2];
                var _template42 = this.template("end", _effect7, "NODEFAULT");
                if (!_template42)
                    _template42 = this.template("endTeamEffect").replace(
                        "[EFFECT]",
                        this.effect(_effect7)
                    );

                return _template42
                    .replace("[TEAM]", this.team(_side6))
                    .replace("[PARTY]", this.party(_side6));
            }

            case "-weather": {
                var weather = args[1];
                if (!weather || weather === "none") {
                    var _template44 = this.template(
                        "end",
                        kwArgs.from,
                        "NODEFAULT"
                    );

                    if (!_template44)
                        return this.template("endFieldEffect").replace(
                            "[EFFECT]",
                            this.effect(weather)
                        );

                    return _template44;
                }
                if (kwArgs.upkeep) {
                    return this.template("upkeep", weather, "NODEFAULT");
                }
                var _line13 = this.maybeAbility(kwArgs.from, kwArgs.of);
                var _template43 = this.template("start", weather, "NODEFAULT");
                if (!_template43)
                    _template43 = this.template("startFieldEffect").replace(
                        "[EFFECT]",
                        this.effect(weather)
                    );

                return _line13 + _template43;
            }

            case "-fieldstart":
            case "-fieldactivate": {
                var _effect8 = args[1];
                var _line14 = this.maybeAbility(kwArgs.from, kwArgs.of);
                if (BattleTextParser.effectId(kwArgs.from) === "hadronengine") {
                    return (
                        _line14 +
                        this.template("start", "hadronengine").replace(
                            "[POKEMON]",
                            this.pokemon(kwArgs.of)
                        )
                    );
                }
                var _templateId3 = cmd.slice(6);
                if (BattleTextParser.effectId(_effect8) === "perishsong")
                    _templateId3 = "start";
                var _template45 = this.template(
                    _templateId3,
                    _effect8,
                    "NODEFAULT"
                );
                if (!_template45)
                    _template45 = this.template("startFieldEffect").replace(
                        "[EFFECT]",
                        this.effect(_effect8)
                    );

                return (
                    _line14 +
                    _template45.replace("[POKEMON]", this.pokemon(kwArgs.of))
                );
            }

            case "-fieldend": {
                var _effect9 = args[1];
                var _template46 = this.template("end", _effect9, "NODEFAULT");
                if (!_template46)
                    _template46 = this.template("endFieldEffect").replace(
                        "[EFFECT]",
                        this.effect(_effect9)
                    );

                return _template46;
            }

            case "-sethp": {
                var _effect10 = kwArgs.from;
                return this.template("activate", _effect10);
            }

            case "-message": {
                var _message = args[1];
                return "  " + _message + "\n";
            }

            case "-hint": {
                var _message2 = args[1];
                return "  (" + _message2 + ")\n";
            }

            case "-activate": {
                var _pokemon19 = args[1],
                    _effect11 = args[2],
                    _target2 = args[3];
                var _id14 = BattleTextParser.effectId(_effect11);
                if (_id14 === "celebrate") {
                    return this.template("activate", "celebrate").replace(
                        "[TRAINER]",
                        this.trainer(_pokemon19.slice(0, 2))
                    );
                }
                if (
                    !_target2 &&
                    [
                        "hyperdrill",
                        "hyperspacefury",
                        "hyperspacehole",
                        "phantomforce",
                        "shadowforce",
                        "feint",
                    ].includes(_id14)
                ) {
                    var _ref3 = [kwArgs.of, _pokemon19];
                    _pokemon19 = _ref3[0];
                    _target2 = _ref3[1];
                    if (!_pokemon19) _pokemon19 = _target2;
                }
                if (!_target2) _target2 = kwArgs.of || _pokemon19;

                var _line15 = this.maybeAbility(_effect11, _pokemon19);

                if (_id14 === "lockon" || _id14 === "mindreader") {
                    var _template48 = this.template("start", _effect11);
                    return (
                        _line15 +
                        _template48
                            .replace("[POKEMON]", this.pokemon(kwArgs.of))
                            .replace("[SOURCE]", this.pokemon(_pokemon19))
                    );
                }

                if (
                    (_id14 === "mummy" || _id14 === "lingeringaroma") &&
                    kwArgs.ability
                ) {
                    _line15 += this.ability(kwArgs.ability, _target2);
                    _line15 += this.ability(
                        _id14 === "mummy" ? "Mummy" : "Lingering Aroma",
                        _target2
                    );

                    var _template49 = this.template("changeAbility", _id14);
                    return (
                        _line15 +
                        _template49.replace("[TARGET]", this.pokemon(_target2))
                    );
                }

                var _templateId4 = "activate";
                if (_id14 === "forewarn" && _pokemon19 === _target2) {
                    _templateId4 = "activateNoTarget";
                }
                if (
                    (_id14 === "protosynthesis" || _id14 === "quarkdrive") &&
                    kwArgs.fromitem
                ) {
                    _templateId4 = "activateFromItem";
                }
                if (_id14 === "orichalcumpulse" && kwArgs.source) {
                    _templateId4 = "start";
                }
                var _template47 = this.template(
                    _templateId4,
                    _effect11,
                    "NODEFAULT"
                );
                if (!_template47) {
                    if (_line15) return _line15;
                    _template47 = this.template("activate");
                    return (
                        _line15 +
                        _template47.replace("[EFFECT]", this.effect(_effect11))
                    );
                }

                if (_id14 === "brickbreak") {
                    _template47 = _template47.replace(
                        "[TEAM]",
                        this.team(_target2.slice(0, 2))
                    );
                }
                if (kwArgs.ability) {
                    _line15 += this.ability(kwArgs.ability, _pokemon19);
                }
                if (kwArgs.ability2) {
                    _line15 += this.ability(kwArgs.ability2, _target2);
                }
                if (
                    kwArgs.move ||
                    kwArgs.number ||
                    kwArgs.item ||
                    kwArgs.name
                ) {
                    _template47 = _template47
                        .replace("[MOVE]", kwArgs.move)
                        .replace("[NUMBER]", kwArgs.number)
                        .replace("[ITEM]", kwArgs.item)
                        .replace("[NAME]", kwArgs.name);
                }
                return (
                    _line15 +
                    _template47
                        .replace("[POKEMON]", this.pokemon(_pokemon19))
                        .replace("[TARGET]", this.pokemon(_target2))
                        .replace("[SOURCE]", this.pokemon(kwArgs.of))
                );
            }

            case "-prepare": {
                var _pokemon20 = args[1],
                    _effect12 = args[2],
                    _target3 = args[3];
                var _template50 = this.template("prepare", _effect12);
                return _template50
                    .replace("[POKEMON]", this.pokemon(_pokemon20))
                    .replace("[TARGET]", this.pokemon(_target3));
            }

            case "-damage": {
                var _pokemon21 = args[1],
                    percentage = args[3];
                var _template51 = this.template(
                    "damage",
                    kwArgs.from,
                    "NODEFAULT"
                );

                var _line16 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon21
                );

                var _id15 = BattleTextParser.effectId(kwArgs.from);
                if (_template51) {
                    return (
                        _line16 +
                        _template51.replace(
                            "[POKEMON]",
                            this.pokemon(_pokemon21)
                        )
                    );
                }

                if (!kwArgs.from) {
                    _template51 = this.template(
                        percentage ? "damagePercentage" : "damage"
                    );

                    return (
                        _line16 +
                        _template51
                            .replace("[POKEMON]", this.pokemon(_pokemon21))
                            .replace("[PERCENTAGE]", percentage)
                    );
                }
                if (kwArgs.from.startsWith("item:")) {
                    _template51 = this.template(
                        kwArgs.of ? "damageFromPokemon" : "damageFromItem"
                    );

                    return (
                        _line16 +
                        _template51
                            .replace("[POKEMON]", this.pokemon(_pokemon21))
                            .replace("[ITEM]", this.effect(kwArgs.from))
                            .replace("[SOURCE]", this.pokemon(kwArgs.of))
                    );
                }
                if (
                    kwArgs.partiallytrapped ||
                    _id15 === "bind" ||
                    _id15 === "wrap"
                ) {
                    _template51 = this.template("damageFromPartialTrapping");
                    return (
                        _line16 +
                        _template51
                            .replace("[POKEMON]", this.pokemon(_pokemon21))
                            .replace("[MOVE]", this.effect(kwArgs.from))
                    );
                }

                _template51 = this.template("damage");
                return (
                    _line16 +
                    _template51.replace("[POKEMON]", this.pokemon(_pokemon21))
                );
            }

            case "-heal": {
                var _pokemon22 = args[1];
                var _template52 = this.template(
                    "heal",
                    kwArgs.from,
                    "NODEFAULT"
                );
                var _line17 = this.maybeAbility(kwArgs.from, _pokemon22);
                if (_template52) {
                    return (
                        _line17 +
                        _template52
                            .replace("[POKEMON]", this.pokemon(_pokemon22))
                            .replace("[SOURCE]", this.pokemon(kwArgs.of))
                            .replace("[NICKNAME]", kwArgs.wisher)
                    );
                }

                if (kwArgs.from && !kwArgs.from.startsWith("ability:")) {
                    _template52 = this.template("healFromEffect");
                    return (
                        _line17 +
                        _template52
                            .replace("[POKEMON]", this.pokemon(_pokemon22))
                            .replace("[EFFECT]", this.effect(kwArgs.from))
                    );
                }

                _template52 = this.template("heal");
                return (
                    _line17 +
                    _template52.replace("[POKEMON]", this.pokemon(_pokemon22))
                );
            }

            case "-boost":
            case "-unboost": {
                var _kwArgs$from4;
                var _pokemon23 = args[1],
                    _stat2 = args[2],
                    _num4 = args[3];
                if (_stat2 === "spa" && this.gen === 1) _stat2 = "spc";
                var amount = parseInt(_num4, 10);
                var _line18 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon23
                );

                var _templateId5 = cmd.slice(1);
                if (amount >= 3) _templateId5 += "3";
                else if (amount >= 2) _templateId5 += "2";
                else if (amount === 0) _templateId5 += "0";
                if (amount && kwArgs.zeffect) {
                    _templateId5 += kwArgs.multiple
                        ? "MultipleFromZEffect"
                        : "FromZEffect";
                } else if (
                    amount &&
                    (_kwArgs$from4 = kwArgs.from) != null &&
                    _kwArgs$from4.startsWith("item:")
                ) {
                    var _template54 = this.template(
                        _templateId5 + "FromItem",
                        kwArgs.from
                    );

                    return (
                        _line18 +
                        _template54
                            .replace("[POKEMON]", this.pokemon(_pokemon23))
                            .replace("[STAT]", BattleTextParser.stat(_stat2))
                            .replace("[ITEM]", this.effect(kwArgs.from))
                    );
                }
                var _template53 = this.template(_templateId5, kwArgs.from);
                return (
                    _line18 +
                    _template53
                        .replace("[POKEMON]", this.pokemon(_pokemon23))
                        .replace("[STAT]", BattleTextParser.stat(_stat2))
                );
            }

            case "-setboost": {
                var _pokemon24 = args[1];
                var _effect13 = kwArgs.from;
                var _line19 = this.maybeAbility(
                    _effect13,
                    kwArgs.of || _pokemon24
                );
                var _template55 = this.template("boost", _effect13);
                return (
                    _line19 +
                    _template55.replace("[POKEMON]", this.pokemon(_pokemon24))
                );
            }

            case "-swapboost": {
                var _pokemon25 = args[1],
                    _target4 = args[2];
                var _line20 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon25
                );

                var _id16 = BattleTextParser.effectId(kwArgs.from);
                var _templateId6 = "swapBoost";
                if (_id16 === "guardswap") _templateId6 = "swapDefensiveBoost";
                if (_id16 === "powerswap") _templateId6 = "swapOffensiveBoost";
                var _template56 = this.template(_templateId6, kwArgs.from);
                return (
                    _line20 +
                    _template56
                        .replace("[POKEMON]", this.pokemon(_pokemon25))
                        .replace("[TARGET]", this.pokemon(_target4))
                );
            }

            case "-copyboost": {
                var _pokemon26 = args[1],
                    _target5 = args[2];
                var _line21 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon26
                );

                var _template57 = this.template("copyBoost", kwArgs.from);
                return (
                    _line21 +
                    _template57
                        .replace("[POKEMON]", this.pokemon(_pokemon26))
                        .replace("[TARGET]", this.pokemon(_target5))
                );
            }

            case "-clearboost":
            case "-clearpositiveboost":
            case "-clearnegativeboost": {
                var _pokemon27 = args[1],
                    source = args[2];
                var _line22 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon27
                );

                var _templateId7 = "clearBoost";
                if (kwArgs.zeffect) _templateId7 = "clearBoostFromZEffect";
                var _template58 = this.template(_templateId7, kwArgs.from);
                return (
                    _line22 +
                    _template58
                        .replace("[POKEMON]", this.pokemon(_pokemon27))
                        .replace("[SOURCE]", this.pokemon(source))
                );
            }

            case "-invertboost": {
                var _pokemon28 = args[1];
                var _line23 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon28
                );

                var _template59 = this.template("invertBoost", kwArgs.from);
                return (
                    _line23 +
                    _template59.replace("[POKEMON]", this.pokemon(_pokemon28))
                );
            }

            case "-clearallboost": {
                return this.template("clearAllBoost", kwArgs.from);
            }

            case "-crit":
            case "-supereffective":
            case "-resisted": {
                var _pokemon29 = args[1];
                var _templateId8 = cmd.slice(1);
                if (_templateId8 === "supereffective")
                    _templateId8 = "superEffective";
                if (kwArgs.spread) _templateId8 += "Spread";
                var _template60 = this.template(_templateId8);
                return _template60.replace(
                    "[POKEMON]",
                    this.pokemon(_pokemon29)
                );
            }

            case "-block": {
                var _pokemon30 = args[1],
                    _effect14 = args[2],
                    _move2 = args[3],
                    attacker = args[4];
                var _line24 = this.maybeAbility(
                    _effect14,
                    kwArgs.of || _pokemon30
                );
                var _id17 = BattleTextParser.effectId(_effect14);
                var _templateId9 = "block";
                if (_id17 === "mist" && this.gen <= 2)
                    _templateId9 = "blockGen" + this.gen;
                var _template61 = this.template(_templateId9, _effect14);
                return (
                    _line24 +
                    _template61
                        .replace("[POKEMON]", this.pokemon(_pokemon30))
                        .replace(
                            "[SOURCE]",
                            this.pokemon(attacker || kwArgs.of)
                        )
                        .replace("[MOVE]", _move2)
                );
            }

            case "-fail": {
                var _pokemon31 = args[1],
                    _effect15 = args[2],
                    _stat3 = args[3];
                var _id18 = BattleTextParser.effectId(_effect15);
                var blocker = BattleTextParser.effectId(kwArgs.from);
                var _line25 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon31
                );

                var _templateId10 = "block";
                if (
                    ["desolateland", "primordialsea"].includes(blocker) &&
                    ![
                        "sunnyday",
                        "raindance",
                        "sandstorm",
                        "hail",
                        "snowscape",
                        "chillyreception",
                    ].includes(_id18)
                ) {
                    _templateId10 = "blockMove";
                } else if (blocker === "uproar" && kwArgs.msg) {
                    _templateId10 = "blockSelf";
                }
                var _template62 = this.template(_templateId10, kwArgs.from);
                if (_template62) {
                    return (
                        _line25 +
                        _template62.replace(
                            "[POKEMON]",
                            this.pokemon(_pokemon31)
                        )
                    );
                }

                if (_id18 === "unboost") {
                    _template62 = this.template(
                        _stat3 ? "failSingular" : "fail",
                        "unboost"
                    );

                    return (
                        _line25 +
                        _template62
                            .replace("[POKEMON]", this.pokemon(_pokemon31))
                            .replace("[STAT]", _stat3)
                    );
                }

                _templateId10 = "fail";
                if (
                    [
                        "brn",
                        "frz",
                        "par",
                        "psn",
                        "slp",
                        "substitute",
                        "shedtail",
                    ].includes(_id18)
                ) {
                    _templateId10 = "alreadyStarted";
                }
                if (kwArgs.heavy) _templateId10 = "failTooHeavy";
                if (kwArgs.weak) _templateId10 = "fail";
                if (kwArgs.forme) _templateId10 = "failWrongForme";
                _template62 = this.template(_templateId10, _id18);
                return (
                    _line25 +
                    _template62.replace("[POKEMON]", this.pokemon(_pokemon31))
                );
            }

            case "-immune": {
                var _pokemon32 = args[1];
                var _line26 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon32
                );

                var _template63 = this.template("block", kwArgs.from);
                if (!_template63) {
                    var _templateId11 = kwArgs.ohko ? "immuneOHKO" : "immune";
                    _template63 = this.template(
                        _pokemon32 ? _templateId11 : "immuneNoPokemon",
                        kwArgs.from
                    );
                }
                return (
                    _line26 +
                    _template63.replace("[POKEMON]", this.pokemon(_pokemon32))
                );
            }

            case "-miss": {
                var _source = args[1],
                    _pokemon33 = args[2];
                var _line27 = this.maybeAbility(
                    kwArgs.from,
                    kwArgs.of || _pokemon33
                );

                if (!_pokemon33) {
                    var _template65 = this.template("missNoPokemon");
                    return (
                        _line27 +
                        _template65.replace("[SOURCE]", this.pokemon(_source))
                    );
                }
                var _template64 = this.template("miss");
                return (
                    _line27 +
                    _template64.replace("[POKEMON]", this.pokemon(_pokemon33))
                );
            }

            case "-center":
            case "-ohko":
            case "-combine": {
                return this.template(cmd.slice(1));
            }

            case "-notarget": {
                return this.template("noTarget");
            }

            case "-mega":
            case "-primal": {
                var _pokemon34 = args[1],
                    species = args[2],
                    _item2 = args[3];
                var _id19 = "";
                var _templateId12 = cmd.slice(1);
                if (species === "Rayquaza") {
                    _id19 = "dragonascent";
                    _templateId12 = "megaNoItem";
                }
                if (!_id19 && cmd === "-mega" && this.gen < 7)
                    _templateId12 = "megaGen6";
                if (!_item2 && cmd === "-mega") _templateId12 = "megaNoItem";
                var _template66 = this.template(_templateId12, _id19);
                var _side7 = _pokemon34.slice(0, 2);
                var pokemonName = this.pokemon(_pokemon34);
                if (cmd === "-mega") {
                    var template2 = this.template("transformMega");
                    _template66 += template2
                        .replace("[POKEMON]", pokemonName)
                        .replace("[SPECIES]", species);
                }
                return _template66
                    .replace("[POKEMON]", pokemonName)
                    .replace("[ITEM]", _item2)
                    .replace("[TRAINER]", this.trainer(_side7));
            }

            case "-terastallize": {
                var _pokemon35 = args[1],
                    type = args[2];
                var _id20 = "";
                var _templateId13 = cmd.slice(1);
                var _template67 = this.template(_templateId13, _id20);
                var _pokemonName = this.pokemon(_pokemon35);
                return _template67
                    .replace("[POKEMON]", _pokemonName)
                    .replace("[TYPE]", type);
            }

            case "-zpower": {
                var _pokemon36 = args[1];
                var _template68 = this.template("zPower");
                return _template68.replace(
                    "[POKEMON]",
                    this.pokemon(_pokemon36)
                );
            }

            case "-burst": {
                var _pokemon37 = args[1];
                var _template69 = this.template("activate", "Ultranecrozium Z");
                return _template69.replace(
                    "[POKEMON]",
                    this.pokemon(_pokemon37)
                );
            }

            case "-zbroken": {
                var _pokemon38 = args[1];
                var _template70 = this.template("zBroken");
                return _template70.replace(
                    "[POKEMON]",
                    this.pokemon(_pokemon38)
                );
            }

            case "-hitcount": {
                var _num5 = args[2];
                if (_num5 === "1") {
                    return this.template("hitCountSingular");
                }
                return this.template("hitCount").replace("[NUMBER]", _num5);
            }

            case "-waiting": {
                var _pokemon39 = args[1],
                    _target6 = args[2];
                var _template71 = this.template("activate", "Water Pledge");
                return _template71
                    .replace("[POKEMON]", this.pokemon(_pokemon39))
                    .replace("[TARGET]", this.pokemon(_target6));
            }

            case "-anim": {
                return "";
            }

            default: {
                return null;
            }
        }
    };
    return BattleTextParser;
})();

if (typeof require === "function") {
    global.BattleTextParser = BattleTextParser;
}
//# sourceMappingURL=battle-text-parser.js.map
