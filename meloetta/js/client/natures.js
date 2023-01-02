"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
    for (var name in all)
        __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
    if ((from && typeof from === "object") || typeof from === "function") {
        for (let key of __getOwnPropNames(from))
            if (!__hasOwnProp.call(to, key) && key !== except)
                __defProp(to, key, {
                    get: () => from[key],
                    enumerable:
                        !(desc = __getOwnPropDesc(from, key)) ||
                        desc.enumerable,
                });
    }
    return to;
};
var __toCommonJS = (mod) =>
    __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var natures_exports = {};
__export(natures_exports, {
    Natures: () => Natures,
});
module.exports = __toCommonJS(natures_exports);
const Natures = {
    adamant: {
        name: "Adamant",
        plus: "atk",
        minus: "spa",
    },
    bashful: {
        name: "Bashful",
    },
    bold: {
        name: "Bold",
        plus: "def",
        minus: "atk",
    },
    brave: {
        name: "Brave",
        plus: "atk",
        minus: "spe",
    },
    calm: {
        name: "Calm",
        plus: "spd",
        minus: "atk",
    },
    careful: {
        name: "Careful",
        plus: "spd",
        minus: "spa",
    },
    docile: {
        name: "Docile",
    },
    gentle: {
        name: "Gentle",
        plus: "spd",
        minus: "def",
    },
    hardy: {
        name: "Hardy",
    },
    hasty: {
        name: "Hasty",
        plus: "spe",
        minus: "def",
    },
    impish: {
        name: "Impish",
        plus: "def",
        minus: "spa",
    },
    jolly: {
        name: "Jolly",
        plus: "spe",
        minus: "spa",
    },
    lax: {
        name: "Lax",
        plus: "def",
        minus: "spd",
    },
    lonely: {
        name: "Lonely",
        plus: "atk",
        minus: "def",
    },
    mild: {
        name: "Mild",
        plus: "spa",
        minus: "def",
    },
    modest: {
        name: "Modest",
        plus: "spa",
        minus: "atk",
    },
    naive: {
        name: "Naive",
        plus: "spe",
        minus: "spd",
    },
    naughty: {
        name: "Naughty",
        plus: "atk",
        minus: "spd",
    },
    quiet: {
        name: "Quiet",
        plus: "spa",
        minus: "spe",
    },
    quirky: {
        name: "Quirky",
    },
    rash: {
        name: "Rash",
        plus: "spa",
        minus: "spd",
    },
    relaxed: {
        name: "Relaxed",
        plus: "def",
        minus: "spe",
    },
    sassy: {
        name: "Sassy",
        plus: "spd",
        minus: "spe",
    },
    serious: {
        name: "Serious",
    },
    timid: {
        name: "Timid",
        plus: "spe",
        minus: "atk",
    },
};
//# sourceMappingURL=natures.js.map
