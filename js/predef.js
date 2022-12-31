module = location = window = exports = this;

Config = window.Config = {
    version: "0",
    bannedHosts: [],
    whitelist: [],
    routes: {
        root: "pokemonshowdown.com",
        client: "play.pokemonshowdown.com",
        dex: "dex.pokemonshowdown.com",
        replays: "replay.pokemonshowdown.com",
        users: "pokemonshowdown.com/users",
    },
    defaultserver: {
        id: "showdown",
        host: "sim3.psim.us",
        port: 443,
        httpport: 8000,
        altport: 80,
        registered: true,
    },
    customcolors: {},
};

Object.prototype.extend = function (obj) {
    for (var i in obj) {
        if (obj.hasOwnProperty(i)) {
            this[i] = obj[i];
        }
    }
    return obj;
};

function refReplacer() {
    let m = new Map(),
        v = new Map(),
        init = null;

    return function (field, value) {
        let p =
            m.get(this) + (Array.isArray(this) ? `[${field}]` : "." + field);
        let isComplex = value === Object(value);

        if (isComplex) m.set(value, p);

        let pp = v.get(value) || "";
        let path = p.replace(/undefined\.\.?/, "");
        let val = pp ? `#REF:${pp[0] == "[" ? "$" : "$."}${pp}` : value;

        !init ? (init = value) : val === init ? (val = "#REF:$") : 0;
        if (!pp && isComplex) v.set(value, path);

        return val;
    };
}

const getCircularReplacer = () => {
    const seen = new WeakSet();
    return (key, value) => {
        if (typeof value === "object" && value !== null) {
            if (seen.has(value)) {
                return;
            }
            seen.add(value);
        }
        return value;
    };
};
