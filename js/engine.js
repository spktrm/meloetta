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

var engine = (function () {
    battle = null;
    choices = null;

    return {
        start: function () {
            battle = new Battle();
            return 0;
        },

        setGen: function (gen) {
            battle.dex = Dex.forGen(gen);
            return 0;
        },

        add: function (command) {
            battle.add(command);
            return 0;
        },

        instantAdd: function (command) {
            battle.run(command, true);
            battle.preemptStepQueue.push(command);
            battle.add(command);
            return 0;
        },

        addToStepQueue: function (command) {
            battle.stepQueue.push(command);
            return 0;
        },

        seekTurn: function (turn, forceReset) {
            battle.seekTurn(turn, forceReset);
            return 0;
        },

        setPerspective: function (sideid) {
            battle.setPerspective(sideid);
            return 0;
        },

        serialize: function () {
            return serialize(battle);
        },

        reset: function () {
            battle = new Battle();
            return 0;
        },

        // Battle Funcs

        parsePokemonId: function (pokemonId) {
            return battle.parsePokemonId(pokemonId);
        },

        getPokemon: function (pokemonId) {
            return battle.getPokemon(pokemonId);
        },

        getNearSide: function () {
            return battle.nearSide;
        },

        // Choices

        getChoices: function (request) {
            choices = new BattleChoiceBuilder(request);
            return JSON.parse(JSON.stringify(choices));
        },

        fixRequest: function (request) {
            BattleChoiceBuilder.fixRequest(request, battle);
            return request;
        },

        // Dex

        getSpecies: function (species) {
            return battle.dex.species.get(species);
        },

        getMove: function (move) {
            return battle.dex.moves.get(move);
        },

        getItem: function (item) {
            return battle.dex.items.get(item);
        },

        getAbility: function (ability) {
            return battle.dex.abilities.get(ability);
        },

        getType: function (type) {
            return battle.dex.types.get(type);
        },
    };
})();

this.engine = engine;
