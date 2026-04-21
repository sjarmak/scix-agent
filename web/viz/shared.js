/* SciX viz shared helpers. Later visualizations attach utilities here. */
(function () {
    "use strict";
    if (typeof window !== "undefined") {
        window.scixViz = window.scixViz || {};
    }
    if (typeof console !== "undefined" && console.log) {
        console.log("scix viz shared");
    }
})();
