window.BattleItems = Items;
window.BattleMovedex = Moves;
window.BattleTypeChart = TypeChart;
window.BattleFormats = FormatsData;
window.BattleAbilities = Abilities;
window.BattlePokedex = Pokedex;
window.BattleBaseSpeciesChart = BattleBaseSpeciesChart;
window.BattleTeambuilderTable = BattleTeambuilderTable;
window.BattleAliases = Aliases;

function serialize(obj) {
    return JSON.decycle(obj);
}

var engine = {
    start: function () {
        this.client = BattleRoom;
        this.reset();
        return 0;
    },

    receive: function (data) {
        this.client.receive(data);
        return 0;
    },

    setGen: function (gen) {
        this.client.battle.dex = Dex.forGen(gen);
        return 0;
    },

    getPid: function () {
        sideid = this.client.side;
        if (sideid === "p1") {
            pid = 0;
        } else if (sideid === "p2") {
            pid = 1;
        }
        return pid;
    },

    getReward: function () {
        sideid = this.client.side;
        if (sideid === "p1") {
            pid = 0;
        } else if (sideid === "p2") {
            pid = 1;
        }
        return {
            pid: pid,
            reward: this.client.reward,
        };
    },

    serialize: function () {
        serialized_state = serialize(this.client);
        // delete serialized_state.battle.stepQueue;
        return serialized_state;
    },

    serializeBattle: function () {
        serialized_battle = serialize(this.client.battle);
        delete serialized_battle.stepQueue;
        return serialized_battle;
    },

    reset: function () {
        this.client.initialize();
        this.client.request = null;
        this.client.side = "";
        this.client.battleEnded = false;
        this.client.title = "";
        return 0;
    },

    // Choices

    chooseMoveTarget: function (posString) {
        return this.client.chooseMoveTarget(posString);
    },

    chooseMove: function (
        pos,
        target,
        isMega,
        isZMove,
        isUltraBurst,
        isDynamax,
        isTerastal
    ) {
        return this.client.chooseMove(
            pos,
            target,
            isMega,
            isZMove,
            isUltraBurst,
            isDynamax,
            isTerastal
        );
    },

    chooseShift: function () {
        return this.client.chooseShift();
    },

    chooseSwitch: function (pos) {
        return this.client.chooseSwitch(pos);
    },

    chooseSwitchTarget: function (pos) {
        return this.client.chooseSwitchTarget(pos);
    },

    chooseTeamPreview: function (pos) {
        return this.client.chooseTeamPreview(pos);
    },

    popOutgoing: function () {
        delete this.client.outgoing_message;
    },

    // Dex

    getSpecies: function (species) {
        return this.client.battle.dex.species.get(species);
    },

    getMove: function (move) {
        return this.client.battle.dex.moves.get(move);
    },

    getItem: function (item) {
        return this.client.battle.dex.items.get(item);
    },

    getAbility: function (ability) {
        return this.client.battle.dex.abilities.get(ability);
    },

    getType: function (type) {
        return this.client.battle.dex.types.get(type);
    },
};
