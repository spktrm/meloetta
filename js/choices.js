var choices = (function () {
    this.choice = undefined;
    return {
        updateControls: function () {
            if (this.battle.scene.customControls) return;

            if (this.side) {
                // player
                this.controlsShown = true;
                if (
                    !controlsShown ||
                    this.choice === undefined ||
                    (this.choice && this.choice.waiting)
                ) {
                    // don't update controls (and, therefore, side) if `this.choice === null`: causes damage miscalculations
                    this.updateControlsForPlayer();
                }
            }
        },
        updateControlsForPlayer: function () {
            this.callbackWaiting = true;

            var act = "";
            var switchables = [];
            if (this.request) {
                // TODO: investigate when to do this
                this.updateSide();
                if (this.request.ally) {
                    this.addAlly(this.request.ally);
                }

                act = this.request.requestType;
                if (this.request.side) {
                    switchables = this.battle.myPokemon;
                }
                if (!this.finalDecision)
                    this.finalDecision = !!this.request.noCancel;
            }

            if (this.choice && this.choice.waiting) {
                act = "";
            }

            var type = this.choice ? this.choice.type : "";

            // The choice object:
            // !this.choice = nothing has been chosen
            // this.choice.choices = array of choice strings
            // this.choice.switchFlags = dict of pokemon indexes that have a switch pending
            // this.choice.switchOutFlags = ???
            // this.choice.freedomDegrees = in a switch request: number of empty slots that can't be replaced
            // this.choice.type = determines what the current choice screen to be displayed is
            // this.choice.waiting = true if the choice has been sent and we're just waiting for the next turn

            switch (act) {
                case "move":
                    if (!this.choice) {
                        this.choice = {
                            choices: [],
                            switchFlags: {},
                            switchOutFlags: {},
                        };
                    }
                    this.updateMoveControls(type);
                    break;

                case "switch":
                    if (!this.choice) {
                        this.choice = {
                            choices: [],
                            switchFlags: {},
                            switchOutFlags: {},
                            freedomDegrees: 0,
                            canSwitch: 0,
                        };

                        if (this.request.forceSwitch !== true) {
                            var faintedLength = _.filter(
                                this.request.forceSwitch,
                                function (fainted) {
                                    return fainted;
                                }
                            ).length;
                            var freedomDegrees =
                                faintedLength -
                                _.filter(
                                    switchables.slice(this.battle.pokemonControlled),
                                    function (mon) {
                                        return !mon.fainted;
                                    }
                                ).length;
                            this.choice.freedomDegrees = Math.max(
                                freedomDegrees,
                                0
                            );
                            this.choice.canSwitch =
                                faintedLength - this.choice.freedomDegrees;
                        }
                    }
                    this.updateSwitchControls(type);
                    break;

                case "team":
                    if (
                        this.battle.mySide.pokemon &&
                        !this.battle.mySide.pokemon.length
                    ) {
                        // too early, we can't determine `this.choice.count` yet
                        // TODO: send teamPreviewCount in the request object
                        this.controlsShown = false;
                        return;
                    }
                    if (!this.choice) {
                        this.choice = {
                            choices: null,
                            teamPreview: [
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24,
                            ].slice(0, switchables.length),
                            done: 0,
                            count: 1,
                        };
                        if (this.battle.gameType === "multi") {
                            this.choice.count = 1;
                        }
                        if (this.battle.gameType === "doubles") {
                            this.choice.count = 2;
                        }
                        if (
                            this.battle.gameType === "triples" ||
                            this.battle.gameType === "rotation"
                        ) {
                            this.choice.count = 3;
                        }
                        // Request full team order if one of our Pokémon has Illusion
                        for (var i = 0; i < switchables.length && i < 6; i++) {
                            if (toID(switchables[i].baseAbility) === "illusion") {
                                this.choice.count = this.battle.myPokemon.length;
                            }
                        }
                        if (this.battle.teamPreviewCount) {
                            var requestCount = parseInt(
                                this.battle.teamPreviewCount,
                                10
                            );
                            if (
                                requestCount > 0 &&
                                requestCount <= switchables.length
                            ) {
                                this.choice.count = requestCount;
                            }
                        }
                        this.choice.choices = new Array(this.choice.count);
                    }
                    this.updateTeamControls(type);
                    break;

                default:
                    this.updateWaitControls();
                    break;
            }
        },
        updateMoveControls: function (type) {
            var switchables =
                this.request && this.request.side ? this.battle.myPokemon : [];

            if (type !== "movetarget") {
                while (
                    switchables[this.choice.choices.length] &&
                    (switchables[this.choice.choices.length].fainted ||
                        switchables[this.choice.choices.length].commanding) &&
                    this.choice.choices.length + 1 <
                    this.battle.nearSide.active.length
                ) {
                    this.choice.choices.push("pass");
                }
            }

            var moveTarget = this.choice ? this.choice.moveTarget : "";
            var pos = this.choice.choices.length;
            if (type === "movetarget") pos--;

            var hpRatio = switchables[pos].hp / switchables[pos].maxhp;

            var curActive =
                this.request && this.request.active && this.request.active[pos];
            if (!curActive) return;
            var trapped = curActive.trapped;
            var canMegaEvo =
                curActive.canMegaEvo || switchables[pos].canMegaEvo;
            var canZMove = curActive.canZMove || switchables[pos].canZMove;
            var canUltraBurst =
                curActive.canUltraBurst || switchables[pos].canUltraBurst;
            var canDynamax =
                curActive.canDynamax || switchables[pos].canDynamax;
            var maxMoves = curActive.maxMoves || switchables[pos].maxMoves;
            var gigantamax = curActive.gigantamax;
            var canTerastallize =
                curActive.canTerastallize || switchables[pos].canTerastallize;
            if (canZMove && typeof canZMove[0] === "string") {
                canZMove = _.map(canZMove, function (move) {
                    return { move: move, target: Dex.moves.get(move).target };
                });
            }
            if (gigantamax) gigantamax = Dex.moves.get(gigantamax);

            this.finalDecisionMove = curActive.maybeDisabled || false;
            this.finalDecisionSwitch = curActive.maybeTrapped || false;
            for (var i = pos + 1; i < this.battle.nearSide.active.length; ++i) {
                var p = this.battle.nearSide.active[i];
                if (p && !p.fainted) {
                    this.finalDecisionMove = this.finalDecisionSwitch = false;
                    break;
                }
            }

            var requestTitle = "";
            if (type === "move2" || type === "movetarget") {
                requestTitle += '<button name="clearChoice">Back</button> ';
            }

            // Target selector
            if (type === "movetarget") {
                requestTitle += "At who? ";

                var activePos =
                    this.battle.mySide.n > 1
                        ? pos + this.battle.pokemonControlled
                        : pos;

                var targetMenus = ["", ""];
                var nearActive = this.battle.nearSide.active;
                var farActive = this.battle.farSide.active;
                var farSlot = farActive.length - 1 - activePos;

                if (
                    (moveTarget === "adjacentAlly" ||
                        moveTarget === "adjacentFoe") &&
                    this.battle.gameType === "freeforall"
                ) {
                    moveTarget = "normal";
                }

                for (var i = farActive.length - 1; i >= 0; i--) {
                    var pokemon = farActive[i];
                    var tooltipArgs = "activepokemon|1|" + i;

                    var disabled = false;
                    if (
                        moveTarget === "adjacentAlly" ||
                        moveTarget === "adjacentAllyOrSelf"
                    ) {
                        disabled = true;
                    } else if (
                        moveTarget === "normal" ||
                        moveTarget === "adjacentFoe"
                    ) {
                        if (Math.abs(farSlot - i) > 1) disabled = true;
                    }

                    if (disabled) {
                        targetMenus[0] += '<button disabled="disabled"></button> ';
                    } else if (!pokemon || pokemon.fainted) {
                        targetMenus[0] +=
                            '<button name="chooseMoveTarget" value="' +
                            (i + 1) +
                            '"><span class="picon" style="' +
                            Dex.getPokemonIcon("missingno") +
                            '"></span></button> ';
                    } else {
                        targetMenus[0] +=
                            '<button name="chooseMoveTarget" value="' +
                            (i + 1) +
                            '" class="has-tooltip" data-tooltip="' +
                            BattleLog.escapeHTML(tooltipArgs) +
                            '"><span class="picon" style="' +
                            Dex.getPokemonIcon(pokemon) +
                            '"></span>' +
                            (this.battle.ignoreOpponent || this.battle.ignoreNicks
                                ? pokemon.speciesForme
                                : BattleLog.escapeHTML(pokemon.name)) +
                            '<span class="' +
                            pokemon.getHPColorClass() +
                            '"><span style="width:' +
                            (Math.round((pokemon.hp * 92) / pokemon.maxhp) || 1) +
                            'px"></span></span>' +
                            (pokemon.status
                                ? '<span class="status ' +
                                pokemon.status +
                                '"></span>'
                                : "") +
                            "</button> ";
                    }
                }
                for (var i = 0; i < nearActive.length; i++) {
                    var pokemon = nearActive[i];
                    var tooltipArgs = "activepokemon|0|" + i;

                    var disabled = false;
                    if (moveTarget === "adjacentFoe") {
                        disabled = true;
                    } else if (
                        moveTarget === "normal" ||
                        moveTarget === "adjacentAlly" ||
                        moveTarget === "adjacentAllyOrSelf"
                    ) {
                        if (Math.abs(activePos - i) > 1) disabled = true;
                    }
                    if (moveTarget !== "adjacentAllyOrSelf" && activePos == i)
                        disabled = true;

                    if (disabled) {
                        targetMenus[1] +=
                            '<button disabled="disabled" style="visibility:hidden"></button> ';
                    } else if (!pokemon || pokemon.fainted) {
                        targetMenus[1] +=
                            '<button name="chooseMoveTarget" value="' +
                            -(i + 1) +
                            '"><span class="picon" style="' +
                            Dex.getPokemonIcon("missingno") +
                            '"></span></button> ';
                    } else {
                        targetMenus[1] +=
                            '<button name="chooseMoveTarget" value="' +
                            -(i + 1) +
                            '" class="has-tooltip" data-tooltip="' +
                            BattleLog.escapeHTML(tooltipArgs) +
                            '"><span class="picon" style="' +
                            Dex.getPokemonIcon(pokemon) +
                            '"></span>' +
                            BattleLog.escapeHTML(pokemon.name) +
                            '<span class="' +
                            pokemon.getHPColorClass() +
                            '"><span style="width:' +
                            (Math.round((pokemon.hp * 92) / pokemon.maxhp) || 1) +
                            'px"></span></span>' +
                            (pokemon.status
                                ? '<span class="status ' +
                                pokemon.status +
                                '"></span>'
                                : "") +
                            "</button> ";
                    }
                }

                this.controls.html(
                    '<div class="controls">' +
                    '<div class="whatdo">' +
                    requestTitle +
                    this.getTimerHTML() +
                    "</div>" +
                    '<div class="switchmenu" style="display:block">' +
                    targetMenus[0] +
                    '<div style="clear:both"></div> </div>' +
                    '<div class="switchmenu" style="display:block">' +
                    targetMenus[1] +
                    "</div>" +
                    "</div>"
                );
            } else {
                // Move chooser
                var hpBar =
                    '<small class="' +
                    (hpRatio < 0.2
                        ? "critical"
                        : hpRatio < 0.5
                            ? "weak"
                            : "healthy") +
                    '">HP ' +
                    switchables[pos].hp +
                    "/" +
                    switchables[pos].maxhp +
                    "</small>";
                requestTitle +=
                    " What will <strong>" +
                    BattleLog.escapeHTML(switchables[pos].name) +
                    "</strong> do? " +
                    hpBar;

                var hasMoves = false;
                var moveMenu = "";
                var movebuttons = "";
                var activePos =
                    this.battle.mySide.n > 1
                        ? pos + this.battle.pokemonControlled
                        : pos;
                var typeValueTracker = new ModifiableValue(
                    this.battle,
                    this.battle.nearSide.active[activePos],
                    this.battle.myPokemon[pos]
                );
                var currentlyDynamaxed = !canDynamax && maxMoves;
                for (var i = 0; i < curActive.moves.length; i++) {
                    var moveData = curActive.moves[i];
                    var move = this.battle.dex.moves.get(moveData.move);
                    var name = move.name;
                    var pp = moveData.pp + "/" + moveData.maxpp;
                    if (!moveData.maxpp) pp = "&ndash;";
                    if (move.id === "Struggle" || move.id === "Recharge")
                        pp = "&ndash;";
                    if (move.id === "Recharge") move.type = "&ndash;";
                    if (name.substr(0, 12) === "Hidden Power")
                        name = "Hidden Power";
                    var moveType = this.tooltips.getMoveType(
                        move,
                        typeValueTracker
                    )[0];
                    var tooltipArgs = "move|" + moveData.move + "|" + pos;
                    if (moveData.disabled) {
                        movebuttons +=
                            '<button disabled="disabled" class="has-tooltip" data-tooltip="' +
                            BattleLog.escapeHTML(tooltipArgs) +
                            '">';
                    } else {
                        movebuttons +=
                            '<button class="type-' +
                            moveType +
                            ' has-tooltip" name="chooseMove" value="' +
                            (i + 1) +
                            '" data-move="' +
                            BattleLog.escapeHTML(moveData.move) +
                            '" data-target="' +
                            BattleLog.escapeHTML(moveData.target) +
                            '" data-tooltip="' +
                            BattleLog.escapeHTML(tooltipArgs) +
                            '">';
                        hasMoves = true;
                    }
                    movebuttons +=
                        name +
                        '<br /><small class="type">' +
                        (moveType ? Dex.types.get(moveType).name : "Unknown") +
                        '</small> <small class="pp">' +
                        pp +
                        "</small>&nbsp;</button> ";
                }
                if (!hasMoves) {
                    moveMenu +=
                        '<button class="movebutton" name="chooseMove" value="0" data-move="Struggle" data-target="randomNormal">Struggle<br /><small class="type">Normal</small> <small class="pp">&ndash;</small>&nbsp;</button> ';
                } else {
                    if (canZMove || canDynamax || currentlyDynamaxed) {
                        var classType = canZMove ? "z" : "max";
                        if (currentlyDynamaxed) {
                            movebuttons = "";
                        } else {
                            movebuttons =
                                '<div class="movebuttons-no' +
                                classType +
                                '">' +
                                movebuttons +
                                '</div><div class="movebuttons-' +
                                classType +
                                '" style="display:none">';
                        }
                        var specialMoves = canZMove ? canZMove : maxMoves.maxMoves;
                        for (var i = 0; i < curActive.moves.length; i++) {
                            if (specialMoves[i]) {
                                // when possible, use Z move to decide type, for cases like Z-Hidden Power
                                var baseMove = this.battle.dex.moves.get(
                                    curActive.moves[i].move
                                );
                                // might not exist, such as for Z status moves - fall back on base move to determine type then
                                var specialMove =
                                    gigantamax ||
                                    this.battle.dex.moves.get(specialMoves[i].move);
                                var moveType = this.tooltips.getMoveType(
                                    specialMove.exists && !specialMove.isMax
                                        ? specialMove
                                        : baseMove,
                                    typeValueTracker,
                                    specialMove.isMax
                                        ? gigantamax ||
                                        switchables[pos].gigantamax ||
                                        true
                                        : undefined
                                )[0];
                                if (
                                    specialMove.isMax &&
                                    specialMove.name !== "Max Guard" &&
                                    !specialMove.id.startsWith("gmax")
                                ) {
                                    specialMove =
                                        this.tooltips.getMaxMoveFromType(moveType);
                                }
                                var tooltipArgs =
                                    classType + "move|" + baseMove.id + "|" + pos;
                                if (specialMove.id.startsWith("gmax"))
                                    tooltipArgs += "|" + specialMove.id;
                                var isDisabled = specialMoves[i].disabled
                                    ? 'disabled="disabled"'
                                    : "";
                                movebuttons +=
                                    "<button " +
                                    isDisabled +
                                    ' class="type-' +
                                    moveType +
                                    ' has-tooltip" name="chooseMove" value="' +
                                    (i + 1) +
                                    '" data-move="' +
                                    BattleLog.escapeHTML(specialMoves[i].move) +
                                    '" data-target="' +
                                    BattleLog.escapeHTML(specialMoves[i].target) +
                                    '" data-tooltip="' +
                                    BattleLog.escapeHTML(tooltipArgs) +
                                    '">';
                                var pp =
                                    curActive.moves[i].pp +
                                    "/" +
                                    curActive.moves[i].maxpp;
                                if (canZMove) {
                                    pp = "1/1";
                                } else if (!curActive.moves[i].maxpp) {
                                    pp = "&ndash;";
                                }
                                movebuttons +=
                                    specialMove.name +
                                    '<br /><small class="type">' +
                                    (moveType
                                        ? Dex.types.get(moveType).name
                                        : "Unknown") +
                                    '</small> <small class="pp">' +
                                    pp +
                                    "</small>&nbsp;</button> ";
                            } else {
                                movebuttons +=
                                    '<button disabled="disabled">&nbsp;</button>';
                            }
                        }
                        if (!currentlyDynamaxed) movebuttons += "</div>";
                    }
                    moveMenu += movebuttons;
                }
                if (canMegaEvo) {
                    moveMenu +=
                        '<br /><label class="megaevo"><input type="checkbox" name="megaevo" />&nbsp;Mega&nbsp;Evolution</label>';
                } else if (canZMove) {
                    moveMenu +=
                        '<br /><label class="megaevo"><input type="checkbox" name="zmove" />&nbsp;Z-Power</label>';
                } else if (canUltraBurst) {
                    moveMenu +=
                        '<br /><label class="megaevo"><input type="checkbox" name="ultraburst" />&nbsp;Ultra Burst</label>';
                } else if (canDynamax) {
                    moveMenu +=
                        '<br /><label class="megaevo"><input type="checkbox" name="dynamax" />&nbsp;Dynamax</label>';
                } else if (canTerastallize) {
                    moveMenu +=
                        '<br /><label class="megaevo"><input type="checkbox" name="terastallize" />&nbsp;Terastallize<br />' +
                        Dex.getTypeIcon(canTerastallize) +
                        "</label>";
                }
                if (this.finalDecisionMove) {
                    moveMenu +=
                        '<em style="display:block;clear:both">You <strong>might</strong> have some moves disabled, so you won\'t be able to cancel an attack!</em><br/>';
                }
                moveMenu += '<div style="clear:left"></div>';

                var moveControls =
                    '<div class="movecontrols">' +
                    '<div class="moveselect"><button name="selectMove">Attack</button></div>' +
                    '<div class="movemenu">' +
                    moveMenu +
                    "</div>" +
                    "</div>";

                var shiftControls = "";
                if (this.battle.gameType === "triples" && pos !== 1) {
                    shiftControls +=
                        '<div class="shiftselect"><button name="chooseShift">Shift</button></div>';
                }

                var switchMenu = "";
                if (trapped) {
                    switchMenu +=
                        "<em>You are trapped and cannot switch!</em><br />";
                    switchMenu += this.displayParty(switchables, trapped);
                } else {
                    switchMenu += this.displayParty(switchables, trapped);
                    if (this.finalDecisionSwitch && this.battle.gen > 2) {
                        switchMenu +=
                            '<em style="display:block;clear:both">You <strong>might</strong> be trapped, so you won\'t be able to cancel a switch!</em><br/>';
                    }
                }
            }
            this.controls = controls;
        },
        displayParty: function (switchables, trapped) {
            var party = "";
            for (var i = 0; i < switchables.length; i++) {
                var pokemon = switchables[i];
                pokemon.name = pokemon.ident.substr(4);
                var tooltipArgs = "switchpokemon|" + i;
                if (
                    pokemon.fainted ||
                    i < this.battle.pokemonControlled ||
                    this.choice.switchFlags[i] ||
                    trapped
                ) {
                    party +=
                        '<button class="disabled has-tooltip" name="chooseDisabled" value="' +
                        BattleLog.escapeHTML(pokemon.name) +
                        (pokemon.fainted
                            ? ",fainted"
                            : trapped
                                ? ",trapped"
                                : i < this.battle.nearSide.active.length
                                    ? ",active"
                                    : "") +
                        '" data-tooltip="' +
                        BattleLog.escapeHTML(tooltipArgs) +
                        '"><span class="picon" style="' +
                        Dex.getPokemonIcon(pokemon) +
                        '"></span>' +
                        BattleLog.escapeHTML(pokemon.name) +
                        (pokemon.hp
                            ? '<span class="' +
                            pokemon.getHPColorClass() +
                            '"><span style="width:' +
                            (Math.round((pokemon.hp * 92) / pokemon.maxhp) || 1) +
                            'px"></span></span>' +
                            (pokemon.status
                                ? '<span class="status ' +
                                pokemon.status +
                                '"></span>'
                                : "")
                            : "") +
                        "</button> ";
                } else {
                    party +=
                        '<button name="chooseSwitch" value="' +
                        i +
                        '" class="has-tooltip" data-tooltip="' +
                        BattleLog.escapeHTML(tooltipArgs) +
                        '"><span class="picon" style="' +
                        Dex.getPokemonIcon(pokemon) +
                        '"></span>' +
                        BattleLog.escapeHTML(pokemon.name) +
                        '<span class="' +
                        pokemon.getHPColorClass() +
                        '"><span style="width:' +
                        (Math.round((pokemon.hp * 92) / pokemon.maxhp) || 1) +
                        'px"></span></span>' +
                        (pokemon.status
                            ? '<span class="status ' + pokemon.status + '"></span>'
                            : "") +
                        "</button> ";
                }
            }
            if (this.battle.mySide.ally) party += this.displayAllyParty();
            return party;
        },
        displayAllyParty: function () {
            var party = "";
            if (!this.battle.myAllyPokemon) return "";
            var allyParty = this.battle.myAllyPokemon;
            for (var i = 0; i < allyParty.length; i++) {
                var pokemon = allyParty[i];
                pokemon.name = pokemon.ident.substr(4);
                var tooltipArgs = "allypokemon|" + i;
                party +=
                    '<button class="disabled has-tooltip" name="chooseDisabled" value="' +
                    BattleLog.escapeHTML(pokemon.name) +
                    ",notMine" +
                    '" data-tooltip="' +
                    BattleLog.escapeHTML(tooltipArgs) +
                    '"><span class="picon" style="' +
                    Dex.getPokemonIcon(pokemon) +
                    '"></span>' +
                    BattleLog.escapeHTML(pokemon.name) +
                    (pokemon.hp
                        ? '<span class="' +
                        pokemon.getHPColorClass() +
                        '"><span style="width:' +
                        (Math.round((pokemon.hp * 92) / pokemon.maxhp) || 1) +
                        'px"></span></span>' +
                        (pokemon.status
                            ? '<span class="status ' +
                            pokemon.status +
                            '"></span>'
                            : "")
                        : "") +
                    "</button> ";
            }
            return party;
        },
        updateSwitchControls: function (type) {
            var pos = this.choice.choices.length;

            // Needed so it client does not freak out when only 1 mon left wants to switch out
            var atLeast1Reviving = false;
            for (var i = 0; i < this.battle.pokemonControlled; i++) {
                var pokemon = this.battle.myPokemon[i];
                if (pokemon.reviving) {
                    atLeast1Reviving = true;
                    break;
                }
            }

            if (
                type !== "switchposition" &&
                this.request.forceSwitch !== true &&
                (!this.choice.freedomDegrees || atLeast1Reviving)
            ) {
                while (!this.request.forceSwitch[pos] && pos < 6) {
                    pos = this.choice.choices.push("pass");
                }
            }

            var switchables =
                this.request && this.request.side ? this.battle.myPokemon : [];
            var nearActive = this.battle.nearSide.active;
            var isReviving = !!switchables[pos].reviving;

            var requestTitle = "";
            if (type === "switch2" || type === "switchposition") {
                requestTitle += '<button name="clearChoice">Back</button> ';
            }

            // Place selector
            if (type === "switchposition") {
                // TODO? hpbar
                requestTitle += "Which Pokémon will it switch in for?";
                var controls = '<div class="switchmenu" style="display:block">';
                for (var i = 0; i < this.battle.pokemonControlled; i++) {
                    var pokemon = this.battle.myPokemon[i];
                    var tooltipArgs = "switchpokemon|" + i;
                    if (
                        (pokemon && !pokemon.fainted) ||
                        this.choice.switchOutFlags[i]
                    ) {
                        controls +=
                            '<button disabled class="has-tooltip" data-tooltip="' +
                            BattleLog.escapeHTML(tooltipArgs) +
                            '"><span class="picon" style="' +
                            Dex.getPokemonIcon(pokemon) +
                            '"></span>' +
                            BattleLog.escapeHTML(pokemon.name) +
                            (!pokemon.fainted
                                ? '<span class="' +
                                pokemon.getHPColorClass() +
                                '"><span style="width:' +
                                (Math.round((pokemon.hp * 92) / pokemon.maxhp) ||
                                    1) +
                                'px"></span></span>' +
                                (pokemon.status
                                    ? '<span class="status ' +
                                    pokemon.status +
                                    '"></span>'
                                    : "")
                                : "") +
                            "</button> ";
                    } else if (!pokemon) {
                        controls += "<button disabled></button> ";
                    } else {
                        controls +=
                            '<button name="chooseSwitchTarget" value="' +
                            i +
                            '" class="has-tooltip" data-tooltip="' +
                            BattleLog.escapeHTML(tooltipArgs) +
                            '"><span class="picon" style="' +
                            Dex.getPokemonIcon(pokemon) +
                            '"></span>' +
                            BattleLog.escapeHTML(pokemon.name) +
                            '<span class="' +
                            pokemon.getHPColorClass() +
                            '"><span style="width:' +
                            (Math.round((pokemon.hp * 92) / pokemon.maxhp) || 1) +
                            'px"></span></span>' +
                            (pokemon.status
                                ? '<span class="status ' +
                                pokemon.status +
                                '"></span>'
                                : "") +
                            "</button> ";
                    }
                }
            } else {
                if (isReviving) {
                    requestTitle += "Choose a fainted Pokémon to revive!";
                } else if (this.choice.freedomDegrees >= 1) {
                    requestTitle += "Choose a Pokémon to send to battle!";
                } else {
                    requestTitle +=
                        "Switch <strong>" +
                        BattleLog.escapeHTML(switchables[pos].name) +
                        "</strong> to:";
                }

                var switchMenu = "";
                for (var i = 0; i < switchables.length; i++) {
                    var pokemon = switchables[i];
                    var tooltipArgs = "switchpokemon|" + i;
                    if (isReviving) {
                        if (!pokemon.fainted || this.choice.switchFlags[i]) {
                            switchMenu +=
                                '<button class="disabled has-tooltip" name="chooseDisabled" value="' +
                                BattleLog.escapeHTML(pokemon.name) +
                                (pokemon.reviving
                                    ? ",active"
                                    : !pokemon.fainted
                                        ? ",notfainted"
                                        : "") +
                                '" data-tooltip="' +
                                BattleLog.escapeHTML(tooltipArgs) +
                                '">';
                        } else {
                            switchMenu +=
                                '<button name="chooseSwitch" value="' +
                                i +
                                '" class="has-tooltip" data-tooltip="' +
                                BattleLog.escapeHTML(tooltipArgs) +
                                '">';
                        }
                    } else {
                        if (
                            pokemon.fainted ||
                            i < this.battle.pokemonControlled ||
                            this.choice.switchFlags[i]
                        ) {
                            switchMenu +=
                                '<button class="disabled has-tooltip" name="chooseDisabled" value="' +
                                BattleLog.escapeHTML(pokemon.name) +
                                (pokemon.fainted
                                    ? ",fainted"
                                    : i < this.battle.pokemonControlled
                                        ? ",active"
                                        : "") +
                                '" data-tooltip="' +
                                BattleLog.escapeHTML(tooltipArgs) +
                                '">';
                        } else {
                            switchMenu +=
                                '<button name="chooseSwitch" value="' +
                                i +
                                '" class="has-tooltip" data-tooltip="' +
                                BattleLog.escapeHTML(tooltipArgs) +
                                '">';
                        }
                    }
                    switchMenu +=
                        '<span class="picon" style="' +
                        Dex.getPokemonIcon(pokemon) +
                        '"></span>' +
                        BattleLog.escapeHTML(pokemon.name) +
                        (!pokemon.fainted
                            ? '<span class="' +
                            pokemon.getHPColorClass() +
                            '"><span style="width:' +
                            (Math.round((pokemon.hp * 92) / pokemon.maxhp) || 1) +
                            'px"></span></span>' +
                            (pokemon.status
                                ? '<span class="status ' +
                                pokemon.status +
                                '"></span>'
                                : "")
                            : "") +
                        "</button> ";
                }
            }
            this.controls = controls;
        },
        updateTeamControls: function (type) {
            var switchables =
                this.request && this.request.side ? this.battle.myPokemon : [];
            var maxIndex = Math.min(switchables.length, 24);

            var requestTitle = "";
            if (this.choice.done) {
                requestTitle =
                    '<button name="clearChoice">Back</button> ' +
                    "What about the rest of your team?";
            } else {
                requestTitle = "How will you start the battle?";
            }

            var switchMenu = "";
            for (var i = 0; i < maxIndex; i++) {
                var oIndex = this.choice.teamPreview[i] - 1;
                var pokemon = switchables[oIndex];
                var tooltipArgs = "switchpokemon|" + oIndex;
                if (i < this.choice.done) {
                    switchMenu +=
                        '<button disabled="disabled" class="has-tooltip" data-tooltip="' +
                        BattleLog.escapeHTML(tooltipArgs) +
                        '"><span class="picon" style="' +
                        Dex.getPokemonIcon(pokemon) +
                        '"></span>' +
                        BattleLog.escapeHTML(pokemon.name) +
                        "</button> ";
                } else {
                    switchMenu +=
                        '<button name="chooseTeamPreview" value="' +
                        i +
                        '" class="has-tooltip" data-tooltip="' +
                        BattleLog.escapeHTML(tooltipArgs) +
                        '"><span class="picon" style="' +
                        Dex.getPokemonIcon(pokemon) +
                        '"></span>' +
                        BattleLog.escapeHTML(pokemon.name) +
                        "</button> ";
                }
            }
            this.controls = controls;
        },
    }
})();
