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
        this.client.initialize();
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

    add: function (command) {
        this.client.battle.add(command);
        return 0;
    },

    instantAdd: function (command) {
        this.client.battle.run(command, true);
        this.client.battle.preemptStepQueue.push(command);
        this.client.battle.add(command);
        return 0;
    },

    addToStepQueue: function (command) {
        this.client.battle.stepQueue.push(command);
        return 0;
    },

    seekTurn: function (turn, forceReset) {
        this.client.battle.seekTurn(turn, forceReset);
        return 0;
    },

    setPerspective: function (sideid) {
        this.client.battle.setPerspective(sideid);
        return 0;
    },

    serialize: function () {
        serialized_state = serialize(this.client);
        // delete serialized_state.battle.stepQueue;
        return serialized_state;
    },

    reset: function () {
        this.client.initialize();
        this.client.request = null;
        this.client.side = "";
        this.client.battleEnded = false;
        this.client.title = "";
        return 0;
    },

    // Battle Funcs

    parsePokemonId: function (pokemonId) {
        return this.client.battle.parsePokemonId(pokemonId);
    },

    getPokemon: function (pokemonId) {
        return this.client.battle.getPokemon(pokemonId);
    },

    getNearSide: function () {
        return this.client.battle.nearSide;
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
