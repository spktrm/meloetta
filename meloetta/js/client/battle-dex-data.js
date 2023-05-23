/**
 * Pokemon Showdown Dex Data
 *
 * A collection of data and definitions for src/battle-dex.ts.
 *
 * Larger data has their own files in data/, so this is just for small
 * miscellaneous data that doesn't need its own file.
 *
 * Licensing note: PS's client has complicated licensing:
 * - The client as a whole is AGPLv3
 * - The battle replay/animation engine (battle-*.ts) by itself is MIT
 *
 * @author Guangcong Luo <guangcongluo@gmail.com>
 * @license MIT
 */

var BattleNatures = {
    Adamant: {
        plus: "atk",
        minus: "spa",
    },
    Bashful: {},
    Bold: {
        plus: "def",
        minus: "atk",
    },
    Brave: {
        plus: "atk",
        minus: "spe",
    },
    Calm: {
        plus: "spd",
        minus: "atk",
    },
    Careful: {
        plus: "spd",
        minus: "spa",
    },
    Docile: {},
    Gentle: {
        plus: "spd",
        minus: "def",
    },
    Hardy: {},
    Hasty: {
        plus: "spe",
        minus: "def",
    },
    Impish: {
        plus: "def",
        minus: "spa",
    },
    Jolly: {
        plus: "spe",
        minus: "spa",
    },
    Lax: {
        plus: "def",
        minus: "spd",
    },
    Lonely: {
        plus: "atk",
        minus: "def",
    },
    Mild: {
        plus: "spa",
        minus: "def",
    },
    Modest: {
        plus: "spa",
        minus: "atk",
    },
    Naive: {
        plus: "spe",
        minus: "spd",
    },
    Naughty: {
        plus: "atk",
        minus: "spd",
    },
    Quiet: {
        plus: "spa",
        minus: "spe",
    },
    Quirky: {},
    Rash: {
        plus: "spa",
        minus: "spd",
    },
    Relaxed: {
        plus: "def",
        minus: "spe",
    },
    Sassy: {
        plus: "spd",
        minus: "spe",
    },
    Serious: {},
    Timid: {
        plus: "spe",
        minus: "atk",
    },
};
var BattleStatIDs = {
    HP: "hp",
    hp: "hp",
    Atk: "atk",
    atk: "atk",
    Def: "def",
    def: "def",
    SpA: "spa",
    SAtk: "spa",
    SpAtk: "spa",
    spa: "spa",
    spc: "spa",
    Spc: "spa",
    SpD: "spd",
    SDef: "spd",
    SpDef: "spd",
    spd: "spd",
    Spe: "spe",
    Spd: "spe",
    spe: "spe",
};

var BattleStatNames = {
    hp: "HP",
    atk: "Atk",
    def: "Def",
    spa: "SpA",
    spd: "SpD",
    spe: "Spe",
};

var BattleBaseSpeciesChart = [
    "unown",
    "burmy",
    "shellos",
    "gastrodon",
    "deerling",
    "sawsbuck",
    "vivillon",
    "flabebe",
    "floette",
    "florges",
    "furfrou",
    "minior",
    "alcremie",
    "tatsugiri",
    "pokestarufo",
    "pokestarbrycenman",
    "pokestarmt",
    "pokestarmt2",
    "pokestartransport",
    "pokestargiant",
    "pokestarhumanoid",
    "pokestarmonster",
    "pokestarf00",
    "pokestarf002",
    "pokestarspirit",
    "pokestarblackdoor",
    "pokestarwhitedoor",
    "pokestarblackbelt",
];

var BattlePokemonIconIndexes = {
    egg: 1020 + 1,
    pikachubelle: 1020 + 2,
    pikachulibre: 1020 + 3,
    pikachuphd: 1020 + 4,
    pikachupopstar: 1020 + 5,
    pikachurockstar: 1020 + 6,
    pikachucosplay: 1020 + 7,
    unownexclamation: 1020 + 8,
    unownquestion: 1020 + 9,
    unownb: 1020 + 10,
    unownc: 1020 + 11,
    unownd: 1020 + 12,
    unowne: 1020 + 13,
    unownf: 1020 + 14,
    unowng: 1020 + 15,
    unownh: 1020 + 16,
    unowni: 1020 + 17,
    unownj: 1020 + 18,
    unownk: 1020 + 19,
    unownl: 1020 + 20,
    unownm: 1020 + 21,
    unownn: 1020 + 22,
    unowno: 1020 + 23,
    unownp: 1020 + 24,
    unownq: 1020 + 25,
    unownr: 1020 + 26,
    unowns: 1020 + 27,
    unownt: 1020 + 28,
    unownu: 1020 + 29,
    unownv: 1020 + 30,
    unownw: 1020 + 31,
    unownx: 1020 + 32,
    unowny: 1020 + 33,
    unownz: 1020 + 34,
    castformrainy: 1020 + 35,
    castformsnowy: 1020 + 36,
    castformsunny: 1020 + 37,
    deoxysattack: 1020 + 38,
    deoxysdefense: 1020 + 39,
    deoxysspeed: 1020 + 40,
    burmysandy: 1020 + 41,
    burmytrash: 1020 + 42,
    wormadamsandy: 1020 + 43,
    wormadamtrash: 1020 + 44,
    cherrimsunshine: 1020 + 45,
    shelloseast: 1020 + 46,
    gastrodoneast: 1020 + 47,
    rotomfan: 1020 + 48,
    rotomfrost: 1020 + 49,
    rotomheat: 1020 + 50,
    rotommow: 1020 + 51,
    rotomwash: 1020 + 52,
    giratinaorigin: 1020 + 53,
    shayminsky: 1020 + 54,
    unfezantf: 1020 + 55,
    basculinbluestriped: 1020 + 56,
    darmanitanzen: 1020 + 57,
    deerlingautumn: 1020 + 58,
    deerlingsummer: 1020 + 59,
    deerlingwinter: 1020 + 60,
    sawsbuckautumn: 1020 + 61,
    sawsbucksummer: 1020 + 62,
    sawsbuckwinter: 1020 + 63,
    frillishf: 1020 + 64,
    jellicentf: 1020 + 65,
    tornadustherian: 1020 + 66,
    thundurustherian: 1020 + 67,
    landorustherian: 1020 + 68,
    kyuremblack: 1020 + 69,
    kyuremwhite: 1020 + 70,
    keldeoresolute: 1020 + 71,
    meloettapirouette: 1020 + 72,
    vivillonarchipelago: 1020 + 73,
    vivilloncontinental: 1020 + 74,
    vivillonelegant: 1020 + 75,
    vivillonfancy: 1020 + 76,
    vivillongarden: 1020 + 77,
    vivillonhighplains: 1020 + 78,
    vivillonicysnow: 1020 + 79,
    vivillonjungle: 1020 + 80,
    vivillonmarine: 1020 + 81,
    vivillonmodern: 1020 + 82,
    vivillonmonsoon: 1020 + 83,
    vivillonocean: 1020 + 84,
    vivillonpokeball: 1020 + 85,
    vivillonpolar: 1020 + 86,
    vivillonriver: 1020 + 87,
    vivillonsandstorm: 1020 + 88,
    vivillonsavanna: 1020 + 89,
    vivillonsun: 1020 + 90,
    vivillontundra: 1020 + 91,
    pyroarf: 1020 + 92,
    flabebeblue: 1020 + 93,
    flabebeorange: 1020 + 94,
    flabebewhite: 1020 + 95,
    flabebeyellow: 1020 + 96,
    floetteblue: 1020 + 97,
    floetteeternal: 1020 + 98,
    floetteorange: 1020 + 99,
    floettewhite: 1020 + 100,
    floetteyellow: 1020 + 101,
    florgesblue: 1020 + 102,
    florgesorange: 1020 + 103,
    florgeswhite: 1020 + 104,
    florgesyellow: 1020 + 105,
    furfroudandy: 1020 + 106,
    furfroudebutante: 1020 + 107,
    furfroudiamond: 1020 + 108,
    furfrouheart: 1020 + 109,
    furfroukabuki: 1020 + 110,
    furfroulareine: 1020 + 111,
    furfroumatron: 1020 + 112,
    furfroupharaoh: 1020 + 113,
    furfroustar: 1020 + 114,
    meowsticf: 1020 + 115,
    aegislashblade: 1020 + 116,
    xerneasneutral: 1020 + 117,
    hoopaunbound: 1020 + 118,
    rattataalola: 1020 + 119,
    raticatealola: 1020 + 120,
    raichualola: 1020 + 121,
    sandshrewalola: 1020 + 122,
    sandslashalola: 1020 + 123,
    vulpixalola: 1020 + 124,
    ninetalesalola: 1020 + 125,
    diglettalola: 1020 + 126,
    dugtrioalola: 1020 + 127,
    meowthalola: 1020 + 128,
    persianalola: 1020 + 129,
    geodudealola: 1020 + 130,
    graveleralola: 1020 + 131,
    golemalola: 1020 + 132,
    grimeralola: 1020 + 133,
    mukalola: 1020 + 134,
    exeggutoralola: 1020 + 135,
    marowakalola: 1020 + 136,
    greninjaash: 1020 + 137,
    zygarde10: 1020 + 138,
    zygardecomplete: 1020 + 139,
    oricoriopompom: 1020 + 140,
    oricoriopau: 1020 + 141,
    oricoriosensu: 1020 + 142,
    lycanrocmidnight: 1020 + 143,
    wishiwashischool: 1020 + 144,
    miniormeteor: 1020 + 145,
    miniororange: 1020 + 146,
    minioryellow: 1020 + 147,
    miniorgreen: 1020 + 148,
    miniorblue: 1020 + 149,
    miniorindigo: 1020 + 150,
    miniorviolet: 1020 + 151,
    magearnaoriginal: 1020 + 152,
    pikachuoriginal: 1020 + 153,
    pikachuhoenn: 1020 + 154,
    pikachusinnoh: 1020 + 155,
    pikachuunova: 1020 + 156,
    pikachukalos: 1020 + 157,
    pikachualola: 1020 + 158,
    pikachupartner: 1020 + 159,
    lycanrocdusk: 1020 + 160,
    necrozmaduskmane: 1020 + 161,
    necrozmadawnwings: 1020 + 162,
    necrozmaultra: 1020 + 163,
    pikachustarter: 1020 + 164,
    eeveestarter: 1020 + 165,
    meowthgalar: 1020 + 166,
    ponytagalar: 1020 + 167,
    rapidashgalar: 1020 + 168,
    farfetchdgalar: 1020 + 169,
    weezinggalar: 1020 + 170,
    mrmimegalar: 1020 + 171,
    corsolagalar: 1020 + 172,
    zigzagoongalar: 1020 + 173,
    linoonegalar: 1020 + 174,
    darumakagalar: 1020 + 175,
    darmanitangalar: 1020 + 176,
    darmanitangalarzen: 1020 + 177,
    yamaskgalar: 1020 + 178,
    stunfiskgalar: 1020 + 179,
    cramorantgulping: 1020 + 180,
    cramorantgorging: 1020 + 181,
    toxtricitylowkey: 1020 + 182,
    sinisteaantique: 854,
    polteageistantique: 855,
    alcremierubycream: 1020 + 183,
    alcremiematchacream: 1020 + 184,
    alcremiemintcream: 1020 + 185,
    alcremielemoncream: 1020 + 186,
    alcremiesaltedcream: 1020 + 187,
    alcremierubyswirl: 1020 + 188,
    alcremiecaramelswirl: 1020 + 189,
    alcremierainbowswirl: 1020 + 190,
    eiscuenoice: 1020 + 191,
    indeedeef: 1020 + 192,
    morpekohangry: 1020 + 193,
    zaciancrowned: 1020 + 194,
    zamazentacrowned: 1020 + 195,
    slowpokegalar: 1020 + 196,
    slowbrogalar: 1020 + 197,
    zarudedada: 1020 + 198,
    pikachuworld: 1020 + 199,
    articunogalar: 1020 + 200,
    zapdosgalar: 1020 + 201,
    moltresgalar: 1020 + 202,
    slowkinggalar: 1020 + 203,
    calyrexice: 1020 + 204,
    calyrexshadow: 1020 + 205,
    growlithehisui: 1020 + 206,
    arcaninehisui: 1020 + 207,
    voltorbhisui: 1020 + 208,
    electrodehisui: 1020 + 209,
    typhlosionhisui: 1020 + 210,
    qwilfishhisui: 1020 + 211,
    sneaselhisui: 1020 + 212,
    samurotthisui: 1020 + 213,
    lilliganthisui: 1020 + 214,
    zoruahisui: 1020 + 215,
    zoroarkhisui: 1020 + 216,
    braviaryhisui: 1020 + 217,
    sliggoohisui: 1020 + 218,
    goodrahisui: 1020 + 219,
    avalugghisui: 1020 + 220,
    decidueyehisui: 1020 + 221,
    basculegionf: 1020 + 222,
    enamorustherian: 1020 + 223,
    taurospaldea: 1020 + 224,
    taurospaldeafire: 1020 + 225,
    taurospaldeawater: 1020 + 226,
    taurospaldeacombat: 1020 + 224,
    taurospaldeablaze: 1020 + 225,
    taurospaldeaaqua: 1020 + 226,
    wooperpaldea: 1020 + 227,
    oinkolognef: 1020 + 228,
    palafinhero: 1020 + 229,
    mausholdfour: 1020 + 230,
    tatsugiridroopy: 1020 + 231,
    tatsugiristretchy: 1020 + 232,
    squawkabillyblue: 1020 + 233,
    squawkabillyyellow: 1020 + 234,
    squawkabillywhite: 1020 + 235,
    gimmighoulroaming: 1020 + 236,
    dialgaorigin: 1020 + 237,
    palkiaorigin: 1020 + 238,
    basculinwhitestriped: 1020 + 239,

    gumshoostotem: 735,
    raticatealolatotem: 1020 + 120,
    marowakalolatotem: 1020 + 136,
    araquanidtotem: 752,
    lurantistotem: 754,
    salazzletotem: 758,
    vikavolttotem: 738,
    togedemarutotem: 777,
    mimikyutotem: 778,
    mimikyubustedtotem: 778,
    ribombeetotem: 743,
    kommoototem: 784,

    venusaurmega: 1260 + 0,
    charizardmegax: 1260 + 1,
    charizardmegay: 1260 + 2,
    blastoisemega: 1260 + 3,
    beedrillmega: 1260 + 4,
    pidgeotmega: 1260 + 5,
    alakazammega: 1260 + 6,
    slowbromega: 1260 + 7,
    gengarmega: 1260 + 8,
    kangaskhanmega: 1260 + 9,
    pinsirmega: 1260 + 10,
    gyaradosmega: 1260 + 11,
    aerodactylmega: 1260 + 12,
    mewtwomegax: 1260 + 13,
    mewtwomegay: 1260 + 14,
    ampharosmega: 1260 + 15,
    steelixmega: 1260 + 16,
    scizormega: 1260 + 17,
    heracrossmega: 1260 + 18,
    houndoommega: 1260 + 19,
    tyranitarmega: 1260 + 20,
    sceptilemega: 1260 + 21,
    blazikenmega: 1260 + 22,
    swampertmega: 1260 + 23,
    gardevoirmega: 1260 + 24,
    sableyemega: 1260 + 25,
    mawilemega: 1260 + 26,
    aggronmega: 1260 + 27,
    medichammega: 1260 + 28,
    manectricmega: 1260 + 29,
    sharpedomega: 1260 + 30,
    cameruptmega: 1260 + 31,
    altariamega: 1260 + 32,
    banettemega: 1260 + 33,
    absolmega: 1260 + 34,
    glaliemega: 1260 + 35,
    salamencemega: 1260 + 36,
    metagrossmega: 1260 + 37,
    latiasmega: 1260 + 38,
    latiosmega: 1260 + 39,
    kyogreprimal: 1260 + 40,
    groudonprimal: 1260 + 41,
    rayquazamega: 1260 + 42,
    lopunnymega: 1260 + 43,
    garchompmega: 1260 + 44,
    lucariomega: 1260 + 45,
    abomasnowmega: 1260 + 46,
    gallademega: 1260 + 47,
    audinomega: 1260 + 48,
    dianciemega: 1260 + 49,
    charizardgmax: 1260 + 50,
    butterfreegmax: 1260 + 51,
    pikachugmax: 1260 + 52,
    meowthgmax: 1260 + 53,
    machampgmax: 1260 + 54,
    gengargmax: 1260 + 55,
    kinglergmax: 1260 + 56,
    laprasgmax: 1260 + 57,
    eeveegmax: 1260 + 58,
    snorlaxgmax: 1260 + 59,
    garbodorgmax: 1260 + 60,
    melmetalgmax: 1260 + 61,
    corviknightgmax: 1260 + 62,
    orbeetlegmax: 1260 + 63,
    drednawgmax: 1260 + 64,
    coalossalgmax: 1260 + 65,
    flapplegmax: 1260 + 66,
    appletungmax: 1260 + 67,
    sandacondagmax: 1260 + 68,
    toxtricitygmax: 1260 + 69,
    toxtricitylowkeygmax: 1260 + 69,
    centiskorchgmax: 1260 + 70,
    hatterenegmax: 1260 + 71,
    grimmsnarlgmax: 1260 + 72,
    alcremiegmax: 1260 + 73,
    copperajahgmax: 1260 + 74,
    duraludongmax: 1260 + 75,
    eternatuseternamax: 1260 + 76,
    venusaurgmax: 1260 + 77,
    blastoisegmax: 1260 + 78,
    rillaboomgmax: 1260 + 79,
    cinderacegmax: 1260 + 80,
    inteleongmax: 1260 + 81,
    urshifugmax: 1260 + 82,
    urshifurapidstrikegmax: 1260 + 83,

    syclant: 1452 + 0,
    revenankh: 1452 + 1,
    pyroak: 1452 + 2,
    fidgit: 1452 + 3,
    stratagem: 1452 + 4,
    arghonaut: 1452 + 5,
    kitsunoh: 1452 + 6,
    cyclohm: 1452 + 7,
    colossoil: 1452 + 8,
    krilowatt: 1452 + 9,
    voodoom: 1452 + 10,
    tomohawk: 1452 + 11,
    necturna: 1452 + 12,
    mollux: 1452 + 13,
    aurumoth: 1452 + 14,
    malaconda: 1452 + 15,
    cawmodore: 1452 + 16,
    volkraken: 1452 + 17,
    plasmanta: 1452 + 18,
    naviathan: 1452 + 19,
    crucibelle: 1452 + 20,
    crucibellemega: 1452 + 21,
    kerfluffle: 1452 + 22,
    pajantom: 1452 + 23,
    jumbao: 1452 + 24,
    caribolt: 1452 + 25,
    smokomodo: 1452 + 26,
    snaelstrom: 1452 + 27,
    equilibra: 1452 + 28,
    astrolotl: 1452 + 29,
    miasmaw: 1452 + 30,
    chromera: 1452 + 31,
    venomicon: 1452 + 32,
    venomiconepilogue: 1452 + 33,
    saharaja: 1452 + 34,

    syclar: 1488 + 0,
    embirch: 1488 + 1,
    flarelm: 1488 + 2,
    breezi: 1488 + 3,
    scratchet: 1488 + 4,
    necturine: 1488 + 5,
    cupra: 1488 + 6,
    argalis: 1488 + 7,
    brattler: 1488 + 8,
    cawdet: 1488 + 9,
    volkritter: 1488 + 10,
    snugglow: 1488 + 11,
    floatoy: 1488 + 12,
    caimanoe: 1488 + 13,
    pluffle: 1488 + 14,
    rebble: 1488 + 15,
    tactite: 1488 + 16,
    privatyke: 1488 + 17,
    nohface: 1488 + 18,
    monohm: 1488 + 19,
    duohm: 1488 + 20,
    protowatt: 1488 + 21,
    voodoll: 1488 + 22,
    mumbao: 1488 + 23,
    fawnifer: 1488 + 24,
    electrelk: 1488 + 25,
    smogecko: 1488 + 26,
    smoguana: 1488 + 27,
    swirlpool: 1488 + 28,
    coribalis: 1488 + 29,
    justyke: 1488 + 30,
    solotl: 1488 + 31,
    miasmite: 1488 + 32,
    dorsoil: 1488 + 33,
    saharascal: 1488 + 34,
};

var BattlePokemonIconIndexesLeft = {
    pikachubelle: 1344 + 0,
    pikachupopstar: 1344 + 1,
    clefairy: 1344 + 2,
    clefable: 1344 + 3,
    jigglypuff: 1344 + 4,
    wigglytuff: 1344 + 5,
    dugtrioalola: 1344 + 6,
    poliwhirl: 1344 + 7,
    poliwrath: 1344 + 8,
    mukalola: 1344 + 9,
    kingler: 1344 + 10,
    croconaw: 1344 + 11,
    cleffa: 1344 + 12,
    igglybuff: 1344 + 13,
    politoed: 1344 + 14,
    unownb: 1344 + 15,
    unownc: 1344 + 16,
    unownd: 1344 + 17,
    unowne: 1344 + 18,
    unownf: 1344 + 19,
    unowng: 1344 + 20,
    unownh: 1344 + 21,
    unownj: 1344 + 22,
    unownk: 1344 + 23,
    unownl: 1344 + 24,
    unownm: 1344 + 25,
    unownn: 1344 + 26,
    unownp: 1344 + 27,
    unownq: 1344 + 28,
    unownquestion: 1344 + 29,
    unownr: 1344 + 30,
    unowns: 1344 + 31,
    unownt: 1344 + 32,
    unownv: 1344 + 33,
    unownz: 1344 + 34,
    sneasel: 1344 + 35,
    teddiursa: 1344 + 36,
    roselia: 1344 + 37,
    zangoose: 1344 + 38,
    seviper: 1344 + 39,
    castformsnowy: 1344 + 40,
    absolmega: 1344 + 41,
    absol: 1344 + 42,
    regirock: 1344 + 43,
    torterra: 1344 + 44,
    budew: 1344 + 45,
    roserade: 1344 + 46,
    magmortar: 1344 + 47,
    togekiss: 1344 + 48,
    rotomwash: 1344 + 49,
    shayminsky: 1344 + 50,
    emboar: 1344 + 51,
    pansear: 1344 + 52,
    simisear: 1344 + 53,
    drilbur: 1344 + 54,
    excadrill: 1344 + 55,
    sawk: 1344 + 56,
    lilligant: 1344 + 57,
    garbodor: 1344 + 58,
    solosis: 1344 + 59,
    vanilluxe: 1344 + 60,
    amoonguss: 1344 + 61,
    klink: 1344 + 62,
    klang: 1344 + 63,
    klinklang: 1344 + 64,
    litwick: 1344 + 65,
    golett: 1344 + 66,
    golurk: 1344 + 67,
    kyuremblack: 1344 + 68,
    kyuremwhite: 1344 + 69,
    kyurem: 1344 + 70,
    keldeoresolute: 1344 + 71,
    meloetta: 1344 + 72,
    greninja: 1344 + 73,
    greninjaash: 1344 + 74,
    furfroudebutante: 1344 + 75,
    barbaracle: 1344 + 76,
    clauncher: 1344 + 77,
    clawitzer: 1344 + 78,
    sylveon: 1344 + 79,
    klefki: 1344 + 80,
    zygarde: 1344 + 81,
    zygarde10: 1344 + 82,
    zygardecomplete: 1344 + 83,
    dartrix: 1344 + 84,
    steenee: 1344 + 85,
    tsareena: 1344 + 86,
    comfey: 1344 + 87,
    miniormeteor: 1344 + 88,
    minior: 1344 + 89,
    miniororange: 1344 + 90,
    minioryellow: 1344 + 91,
    miniorgreen: 1344 + 92,
    miniorblue: 1344 + 93,
    miniorviolet: 1344 + 94,
    miniorindigo: 1344 + 95,
    dhelmise: 1344 + 96,
    necrozma: 1344 + 97,
    marshadow: 1344 + 98,
    pikachuoriginal: 1344 + 99,
    pikachupartner: 1344 + 100,
    necrozmaduskmane: 1344 + 101,
    necrozmadawnwings: 1344 + 102,
    necrozmaultra: 1344 + 103,
    stakataka: 1344 + 104,
    blacephalon: 1344 + 105,
};

var BattleAvatarNumbers = {
    1: "lucas",
    2: "dawn",
    3: "youngster-gen4dp",
    4: "lass-gen4dp",
    5: "camper",
    6: "picnicker",
    7: "bugcatcher-gen4dp",
    8: "aromalady",
    9: "twins-gen4dp",
    10: "hiker-gen4",
    11: "battlegirl-gen4",
    12: "fisherman-gen4",
    13: "cyclist-gen4",
    14: "cyclistf-gen4",
    15: "blackbelt-gen4dp",
    16: "artist-gen4",
    17: "pokemonbreeder-gen4",
    18: "pokemonbreederf-gen4",
    19: "cowgirl",
    20: "jogger",
    21: "pokefan-gen4",
    22: "pokefanf-gen4",
    23: "pokekid",
    24: "youngcouple-gen4dp",
    25: "acetrainer-gen4dp",
    26: "acetrainerf-gen4dp",
    27: "waitress-gen4",
    28: "veteran-gen4",
    29: "ninjaboy",
    30: "dragontamer",
    31: "birdkeeper-gen4dp",
    32: "doubleteam",
    33: "richboy-gen4",
    34: "lady-gen4",
    35: "gentleman-gen4dp",
    36: "madame-gen4dp",
    37: "beauty-gen4dp",
    38: "collector",
    39: "policeman-gen4",
    40: "pokemonranger-gen4",
    41: "pokemonrangerf-gen4",
    42: "scientist-gen4dp",
    43: "swimmer-gen4dp",
    44: "swimmerf-gen4dp",
    45: "tuber",
    46: "tuberf",
    47: "sailor",
    48: "sisandbro",
    49: "ruinmaniac",
    50: "psychic-gen4",
    51: "psychicf-gen4",
    52: "gambler",
    53: "guitarist-gen4",
    54: "acetrainersnow",
    55: "acetrainersnowf",
    56: "skier",
    57: "skierf-gen4dp",
    58: "roughneck-gen4",
    59: "clown",
    60: "worker-gen4",
    61: "schoolkid-gen4dp",
    62: "schoolkidf-gen4",
    63: "roark",
    64: "barry",
    65: "byron",
    66: "aaron",
    67: "bertha",
    68: "flint",
    69: "lucian",
    70: "cynthia-gen4",
    71: "bellepa",
    72: "rancher",
    73: "mars",
    74: "galacticgrunt",
    75: "gardenia",
    76: "crasherwake",
    77: "maylene",
    78: "fantina",
    79: "candice",
    80: "volkner",
    81: "parasollady-gen4",
    82: "waiter-gen4dp",
    83: "interviewers",
    84: "cameraman",
    85: "reporter",
    86: "idol",
    87: "cyrus",
    88: "jupiter",
    89: "saturn",
    90: "galacticgruntf",
    91: "argenta",
    92: "palmer",
    93: "thorton",
    94: "buck",
    95: "darach-caitlin",
    96: "marley",
    97: "mira",
    98: "cheryl",
    99: "riley",
    100: "dahlia",
    101: "ethan",
    102: "lyra",
    103: "twins-gen4",
    104: "lass-gen4",
    105: "acetrainer-gen4",
    106: "acetrainerf-gen4",
    107: "juggler",
    108: "sage",
    109: "li",
    110: "gentleman-gen4",
    111: "teacher",
    112: "beauty",
    113: "birdkeeper",
    114: "swimmer-gen4",
    115: "swimmerf-gen4",
    116: "kimonogirl",
    117: "scientist-gen4",
    118: "acetrainercouple",
    119: "youngcouple",
    120: "supernerd",
    121: "medium",
    122: "schoolkid-gen4",
    123: "blackbelt-gen4",
    124: "pokemaniac",
    125: "firebreather",
    126: "burglar",
    127: "biker-gen4",
    128: "skierf",
    129: "boarder",
    130: "rocketgrunt",
    131: "rocketgruntf",
    132: "archer",
    133: "ariana",
    134: "proton",
    135: "petrel",
    136: "eusine",
    137: "lucas-gen4pt",
    138: "dawn-gen4pt",
    139: "madame-gen4",
    140: "waiter-gen4",
    141: "falkner",
    142: "bugsy",
    143: "whitney",
    144: "morty",
    145: "chuck",
    146: "jasmine",
    147: "pryce",
    148: "clair",
    149: "will",
    150: "koga",
    151: "bruno",
    152: "karen",
    153: "lance",
    154: "brock",
    155: "misty",
    156: "ltsurge",
    157: "erika",
    158: "janine",
    159: "sabrina",
    160: "blaine",
    161: "blue",
    162: "red",
    163: "red",
    164: "silver",
    165: "giovanni",
    166: "unknownf",
    167: "unknown",
    168: "unknown",
    169: "hilbert",
    170: "hilda",
    171: "youngster",
    172: "lass",
    173: "schoolkid",
    174: "schoolkidf",
    175: "smasher",
    176: "linebacker",
    177: "waiter",
    178: "waitress",
    179: "chili",
    180: "cilan",
    181: "cress",
    182: "nurseryaide",
    183: "preschoolerf",
    184: "preschooler",
    185: "twins",
    186: "pokemonbreeder",
    187: "pokemonbreederf",
    188: "lenora",
    189: "burgh",
    190: "elesa",
    191: "clay",
    192: "skyla",
    193: "pokemonranger",
    194: "pokemonrangerf",
    195: "worker",
    196: "backpacker",
    197: "backpackerf",
    198: "fisherman",
    199: "musician",
    200: "dancer",
    201: "harlequin",
    202: "artist",
    203: "baker",
    204: "psychic",
    205: "psychicf",
    206: "cheren",
    207: "bianca",
    208: "plasmagrunt-gen5bw",
    209: "n",
    210: "richboy",
    211: "lady",
    212: "pilot",
    213: "workerice",
    214: "hoopster",
    215: "scientistf",
    216: "clerkf",
    217: "acetrainerf",
    218: "acetrainer",
    219: "blackbelt",
    220: "scientist",
    221: "striker",
    222: "brycen",
    223: "iris",
    224: "drayden",
    225: "roughneck",
    226: "janitor",
    227: "pokefan",
    228: "pokefanf",
    229: "doctor",
    230: "nurse",
    231: "hooligans",
    232: "battlegirl",
    233: "parasollady",
    234: "clerk",
    235: "clerk-boss",
    236: "backers",
    237: "backersf",
    238: "veteran",
    239: "veteranf",
    240: "biker",
    241: "infielder",
    242: "hiker",
    243: "madame",
    244: "gentleman",
    245: "plasmagruntf-gen5bw",
    246: "shauntal",
    247: "marshal",
    248: "grimsley",
    249: "caitlin",
    250: "ghetsis-gen5bw",
    251: "depotagent",
    252: "swimmer",
    253: "swimmerf",
    254: "policeman",
    255: "maid",
    256: "ingo",
    257: "alder",
    258: "cyclist",
    259: "cyclistf",
    260: "cynthia",
    261: "emmet",
    262: "hilbert-wonderlauncher",
    263: "hilda-wonderlauncher",
    264: "hugh",
    265: "rosa",
    266: "nate",
    267: "colress",
    268: "beauty-gen5bw2",
    269: "ghetsis",
    270: "plasmagrunt",
    271: "plasmagruntf",
    272: "iris-gen5bw2",
    273: "brycenman",
    274: "shadowtriad",
    275: "rood",
    276: "zinzolin",
    277: "cheren-gen5bw2",
    278: "marlon",
    279: "roxie",
    280: "roxanne",
    281: "brawly",
    282: "wattson",
    283: "flannery",
    284: "norman",
    285: "winona",
    286: "tate",
    287: "liza",
    288: "juan",
    289: "guitarist",
    290: "steven",
    291: "wallace",
    292: "bellelba",
    293: "benga",
    294: "ash",
    "#bw2elesa": "elesa-gen5bw2",
    "#teamrocket": "teamrocket",
    "#yellow": "yellow",
    "#zinnia": "zinnia",
    "#clemont": "clemont",
    "#wally": "wally",
    breeder: "pokemonbreeder",
    breederf: "pokemonbreederf",
    "hilbert-dueldisk": "hilbert-wonderlauncher",
    "hilda-dueldisk": "hilda-wonderlauncher",
    "nate-dueldisk": "nate-wonderlauncher",
    "rosa-dueldisk": "rosa-wonderlauncher",

    1001: "#1001",
    1002: "#1002",
    1003: "#1003",
    1005: "#1005",
    1010: "#1010",
};
var PureEffect = function PureEffect(id, name) {
    this.effectType = "PureEffect";
    this.id = void 0;
    this.name = void 0;
    this.gen = void 0;
    this.exists = void 0;
    this.id = id;
    this.name = name;
    this.gen = 0;
    this.exists = false;
};
var Item = function Item(id, name, data) {
    this.effectType = "Item";
    this.id = void 0;
    this.name = void 0;
    this.gen = void 0;
    this.exists = void 0;
    this.num = void 0;
    this.spritenum = void 0;
    this.desc = void 0;
    this.shortDesc = void 0;
    this.megaStone = void 0;
    this.megaEvolves = void 0;
    this.zMove = void 0;
    this.zMoveType = void 0;
    this.zMoveFrom = void 0;
    this.zMoveUser = void 0;
    this.onPlate = void 0;
    this.onMemory = void 0;
    this.onDrive = void 0;
    this.fling = void 0;
    this.naturalGift = void 0;
    this.isPokeball = void 0;
    this.itemUser = void 0;
    if (!data || typeof data !== "object") data = {};
    if (data.name) name = data.name;
    this.name = Dex.sanitizeName(name);
    this.id = id;
    this.gen = data.gen || 0;
    this.exists = "exists" in data ? !!data.exists : true;

    this.num = data.num || 0;
    this.spritenum = data.spritenum || 0;
    this.desc = data.desc || data.shortDesc || "";
    this.shortDesc = data.shortDesc || this.desc;

    this.megaStone = data.megaStone || "";
    this.megaEvolves = data.megaEvolves || "";
    this.zMove = data.zMove || null;
    this.zMoveType = data.zMoveType || "";
    this.zMoveFrom = data.zMoveFrom || "";
    this.zMoveUser = data.zMoveUser || null;
    this.onPlate = data.onPlate || "";
    this.onMemory = data.onMemory || "";
    this.onDrive = data.onDrive || "";
    this.fling = data.fling || null;
    this.naturalGift = data.naturalGift || null;
    this.isPokeball = !!data.isPokeball;
    this.itemUser = data.itemUser;

    if (!this.gen) {
        if (this.num >= 577) {
            this.gen = 6;
        } else if (this.num >= 537) {
            this.gen = 5;
        } else if (this.num >= 377) {
            this.gen = 4;
        } else {
            this.gen = 3;
        }
    }
};
var Move = function Move(id, name, data) {
    var _this$maxMove;
    this.effectType = "Move";
    this.id = void 0;
    this.name = void 0;
    this.gen = void 0;
    this.exists = void 0;
    this.basePower = void 0;
    this.accuracy = void 0;
    this.pp = void 0;
    this.type = void 0;
    this.category = void 0;
    this.priority = void 0;
    this.target = void 0;
    this.pressureTarget = void 0;
    this.flags = void 0;
    this.critRatio = void 0;
    this.desc = void 0;
    this.shortDesc = void 0;
    this.isNonstandard = void 0;
    this.isZ = void 0;
    this.zMove = void 0;
    this.isMax = void 0;
    this.maxMove = void 0;
    this.ohko = void 0;
    this.recoil = void 0;
    this.heal = void 0;
    this.multihit = void 0;
    this.hasCrashDamage = void 0;
    this.noPPBoosts = void 0;
    this.secondaries = void 0;
    this.noSketch = void 0;
    this.num = void 0;
    if (!data || typeof data !== "object") data = {};
    if (data.name) name = data.name;
    this.name = Dex.sanitizeName(name);
    this.id = id;
    this.gen = data.gen || 0;
    this.exists = "exists" in data ? !!data.exists : true;

    this.basePower = data.basePower || 0;
    this.accuracy = data.accuracy || 0;
    this.pp = data.pp || 1;
    this.type = data.type || "???";
    this.category = data.category || "Physical";
    this.priority = data.priority || 0;
    this.target = data.target || "normal";
    this.pressureTarget = data.pressureTarget || this.target;
    this.flags = data.flags || {};
    this.critRatio = data.critRatio === 0 ? 0 : data.critRatio || 1;

    this.desc = data.desc;
    this.shortDesc = data.shortDesc;
    this.isNonstandard = data.isNonstandard || null;
    this.isZ = data.isZ || "";
    this.zMove = data.zMove || {};
    this.ohko = data.ohko || null;
    this.recoil = data.recoil || null;
    this.heal = data.heal || null;
    this.multihit = data.multihit || null;
    this.hasCrashDamage = data.hasCrashDamage || false;
    this.noPPBoosts = data.noPPBoosts || false;
    this.secondaries =
        data.secondaries || (data.secondary ? [data.secondary] : null);
    this.noSketch = !!data.noSketch;

    this.isMax = data.isMax || false;
    this.maxMove = data.maxMove || { basePower: 0 };
    if (
        this.category !== "Status" &&
        !((_this$maxMove = this.maxMove) != null && _this$maxMove.basePower)
    ) {
        if (this.isZ || this.isMax) {
            this.maxMove = { basePower: 1 };
        } else if (!this.basePower) {
            this.maxMove = { basePower: 100 };
        } else if (["Fighting", "Poison"].includes(this.type)) {
            if (this.basePower >= 150) {
                this.maxMove = { basePower: 100 };
            } else if (this.basePower >= 110) {
                this.maxMove = { basePower: 95 };
            } else if (this.basePower >= 75) {
                this.maxMove = { basePower: 90 };
            } else if (this.basePower >= 65) {
                this.maxMove = { basePower: 85 };
            } else if (this.basePower >= 55) {
                this.maxMove = { basePower: 80 };
            } else if (this.basePower >= 45) {
                this.maxMove = { basePower: 75 };
            } else {
                this.maxMove = { basePower: 70 };
            }
        } else {
            if (this.basePower >= 150) {
                this.maxMove = { basePower: 150 };
            } else if (this.basePower >= 110) {
                this.maxMove = { basePower: 140 };
            } else if (this.basePower >= 75) {
                this.maxMove = { basePower: 130 };
            } else if (this.basePower >= 65) {
                this.maxMove = { basePower: 120 };
            } else if (this.basePower >= 55) {
                this.maxMove = { basePower: 110 };
            } else if (this.basePower >= 45) {
                this.maxMove = { basePower: 100 };
            } else {
                this.maxMove = { basePower: 90 };
            }
        }
    }

    if (this.category !== "Status" && !this.isZ && !this.isMax) {
        var basePower = this.basePower;
        this.zMove = {};
        if (Array.isArray(this.multihit)) basePower *= 3;
        if (!basePower) {
            this.zMove.basePower = 100;
        } else if (basePower >= 140) {
            this.zMove.basePower = 200;
        } else if (basePower >= 130) {
            this.zMove.basePower = 195;
        } else if (basePower >= 120) {
            this.zMove.basePower = 190;
        } else if (basePower >= 110) {
            this.zMove.basePower = 185;
        } else if (basePower >= 100) {
            this.zMove.basePower = 180;
        } else if (basePower >= 90) {
            this.zMove.basePower = 175;
        } else if (basePower >= 80) {
            this.zMove.basePower = 160;
        } else if (basePower >= 70) {
            this.zMove.basePower = 140;
        } else if (basePower >= 60) {
            this.zMove.basePower = 120;
        } else {
            this.zMove.basePower = 100;
        }
        if (data.zMove) this.zMove.basePower = data.zMove.basePower;
    }

    this.num = data.num || 0;
    if (!this.gen) {
        if (this.num >= 743) {
            this.gen = 8;
        } else if (this.num >= 622) {
            this.gen = 7;
        } else if (this.num >= 560) {
            this.gen = 6;
        } else if (this.num >= 468) {
            this.gen = 5;
        } else if (this.num >= 355) {
            this.gen = 4;
        } else if (this.num >= 252) {
            this.gen = 3;
        } else if (this.num >= 166) {
            this.gen = 2;
        } else if (this.num >= 1) {
            this.gen = 1;
        }
    }
};
var Ability = function Ability(id, name, data) {
    this.effectType = "Ability";
    this.id = void 0;
    this.name = void 0;
    this.gen = void 0;
    this.exists = void 0;
    this.num = void 0;
    this.shortDesc = void 0;
    this.desc = void 0;
    this.rating = void 0;
    this.isPermanent = void 0;
    this.isNonstandard = void 0;
    if (!data || typeof data !== "object") data = {};
    if (data.name) name = data.name;
    this.name = Dex.sanitizeName(name);
    this.id = id;
    this.gen = data.gen || 0;
    this.exists = "exists" in data ? !!data.exists : true;
    this.num = data.num || 0;
    this.shortDesc = data.shortDesc || data.desc || "";
    this.desc = data.desc || data.shortDesc || "";
    this.rating = data.rating || 1;
    this.isPermanent = !!data.isPermanent;
    this.isNonstandard = !!data.isNonstandard;
    if (!this.gen) {
        if (this.num >= 234) {
            this.gen = 8;
        } else if (this.num >= 192) {
            this.gen = 7;
        } else if (this.num >= 165) {
            this.gen = 6;
        } else if (this.num >= 124) {
            this.gen = 5;
        } else if (this.num >= 77) {
            this.gen = 4;
        } else if (this.num >= 1) {
            this.gen = 3;
        }
    }
};
var Species = function Species(id, name, data) {
    this.effectType = "Species";
    this.id = void 0;
    this.name = void 0;
    this.gen = void 0;
    this.exists = void 0;
    this.baseSpecies = void 0;
    this.forme = void 0;
    this.formeid = void 0;
    this.spriteid = void 0;
    this.baseForme = void 0;
    this.num = void 0;
    this.types = void 0;
    this.abilities = void 0;
    this.baseStats = void 0;
    this.bst = void 0;
    this.weightkg = void 0;
    this.heightm = void 0;
    this.gender = void 0;
    this.color = void 0;
    this.genderRatio = void 0;
    this.eggGroups = void 0;
    this.tags = void 0;
    this.otherFormes = void 0;
    this.cosmeticFormes = void 0;
    this.evos = void 0;
    this.prevo = void 0;
    this.evoType = void 0;
    this.evoLevel = void 0;
    this.evoMove = void 0;
    this.evoItem = void 0;
    this.evoCondition = void 0;
    this.requiredItems = void 0;
    this.tier = void 0;
    this.isTotem = void 0;
    this.isMega = void 0;
    this.cannotDynamax = void 0;
    this.canGigantamax = void 0;
    this.isPrimal = void 0;
    this.battleOnly = void 0;
    this.isNonstandard = void 0;
    this.unreleasedHidden = void 0;
    this.changesFrom = void 0;
    if (!data || typeof data !== "object") data = {};
    if (data.name) name = data.name;
    this.name = Dex.sanitizeName(name);
    this.id = id;
    this.gen = data.gen || 0;
    this.exists = "exists" in data ? !!data.exists : true;
    this.baseSpecies = data.baseSpecies || name;
    this.forme = data.forme || "";
    var baseId = toID(this.baseSpecies);
    this.formeid = baseId === this.id ? "" : "-" + toID(this.forme);
    this.spriteid = baseId + this.formeid;
    if (this.spriteid.slice(-5) === "totem")
        this.spriteid = this.spriteid.slice(0, -5);
    if (this.spriteid.slice(-1) === "-")
        this.spriteid = this.spriteid.slice(0, -1);
    this.baseForme = data.baseForme || "";

    this.num = data.num || 0;
    this.types = data.types || ["???"];
    this.abilities = data.abilities || { 0: "No Ability" };
    this.baseStats = data.baseStats || {
        hp: 0,
        atk: 0,
        def: 0,
        spa: 0,
        spd: 0,
        spe: 0,
    };
    this.bst =
        this.baseStats.hp +
        this.baseStats.atk +
        this.baseStats.def +
        this.baseStats.spa +
        this.baseStats.spd +
        this.baseStats.spe;
    this.weightkg = data.weightkg || 0;

    this.heightm = data.heightm || 0;
    this.gender = data.gender || "";
    this.color = data.color || "";
    this.genderRatio = data.genderRatio || null;
    this.eggGroups = data.eggGroups || [];
    this.tags = data.tags || [];

    this.otherFormes = data.otherFormes || null;
    this.cosmeticFormes = data.cosmeticFormes || null;
    this.evos = data.evos || null;
    this.prevo = data.prevo || "";
    this.evoType = data.evoType || "";
    this.evoLevel = data.evoLevel || 0;
    this.evoMove = data.evoMove || "";
    this.evoItem = data.evoItem || "";
    this.evoCondition = data.evoCondition || "";
    this.requiredItems =
        data.requiredItems || (data.requiredItem ? [data.requiredItem] : []);
    this.tier = data.tier || "";

    this.isTotem = false;
    this.isMega = !!(
        this.forme && ["-mega", "-megax", "-megay"].includes(this.formeid)
    );
    this.cannotDynamax = !!data.cannotDynamax;
    this.canGigantamax = !!data.canGigantamax;
    this.isPrimal = !!(this.forme && this.formeid === "-primal");
    this.battleOnly = data.battleOnly || undefined;
    this.isNonstandard = data.isNonstandard || null;
    this.unreleasedHidden = data.unreleasedHidden || false;
    this.changesFrom = data.changesFrom || undefined;
    if (!this.gen) {
        if (this.num >= 906 || this.formeid.startsWith("-paldea")) {
            this.gen = 9;
        } else if (
            this.num >= 810 ||
            this.formeid.startsWith("-galar") ||
            this.formeid.startsWith("-hisui")
        ) {
            this.gen = 8;
        } else if (
            this.num >= 722 ||
            this.formeid === "-alola" ||
            this.formeid === "-starter"
        ) {
            this.gen = 7;
        } else if (this.isMega || this.isPrimal) {
            this.gen = 6;
            this.battleOnly = this.baseSpecies;
        } else if (
            this.formeid === "-totem" ||
            this.formeid === "-alolatotem"
        ) {
            this.gen = 7;
            this.isTotem = true;
        } else if (this.num >= 650) {
            this.gen = 6;
        } else if (this.num >= 494) {
            this.gen = 5;
        } else if (this.num >= 387) {
            this.gen = 4;
        } else if (this.num >= 252) {
            this.gen = 3;
        } else if (this.num >= 152) {
            this.gen = 2;
        } else if (this.num >= 1) {
            this.gen = 1;
        }
    }
};

if (typeof require === "function") {
    global.BattleBaseSpeciesChart = BattleBaseSpeciesChart;
    global.BattleNatures = BattleNatures;
    global.PureEffect = PureEffect;
    global.Species = Species;
    global.Ability = Ability;
    global.Item = Item;
    global.Move = Move;
}
//# sourceMappingURL=battle-dex-data.js.map
