function serialize(obj) {
    return JSON.decycle(obj);
}



var engine = (function () {
    battle = null;
    choices = null;

    return {
        start: function () {
            battle = new Battle();
        },

        add: function (command) {
            battle.add(command);
        },

        instantAdd: function (command) {
            battle.run(command, true);
            battle.preemptStepQueue.push(command);
            battle.add(command);
        },

        addToStepQueue: function (command) {
            battle.stepQueue.push(command);
        },

        seekTurn: function (turn, forceReset) {
            battle.seekTurn(turn, forceReset);
        },

        setPerspective: function (sideid) {
            battle.setPerspective(sideid);
        },

        serialize: function () {
            return serialize(battle);
        },

        reset: function () {
            battle = new Battle();
        },

        // Battle Funcs

        parsePokemonId: function (pokemonId) {
            return battle.parsePokemonId(pokemonId)
        },

        getPokemon: function (pokemonId) {
            return battle.getPokemon(pokemonId)
        },

        getNearSide: function () {
            return battle.nearSide
        },

        // Choices

        getChoices: function(request) {
            choices = new BattleChoiceBuilder(request)
            return JSON.parse(JSON.stringify(choices))
        },

        fixRequest: function(request) {
            BattleChoiceBuilder.fixRequest(request, battle);
            return request
        }
    };
})();

this.engine = engine;