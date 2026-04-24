/* SciX viz shared helpers. Later visualizations attach utilities here. */
;(function () {
  'use strict'
  if (typeof window === 'undefined') return

  var scixViz = window.scixViz || {}
  window.scixViz = scixViz

  // --- Resolution state ---------------------------------------------------
  // `community_semantic_{coarse,medium,fine}` is the user-selected Leiden
  // resolution. We persist the choice in localStorage so it sticks across
  // pages and reloads, expose get/set helpers on `window.scixViz`, and emit
  // a `scix:resolution-change` CustomEvent on `window` whenever the choice
  // changes (viz pages subscribe to refetch data + re-render).
  var STORAGE_KEY = 'scix.viz.resolution'
  var VALID_RES = ['coarse', 'medium', 'fine']
  var DEFAULT_RES = 'coarse'
  var RESOLUTION_LABELS = {
    coarse: 'coarse (~20)',
    medium: 'medium (~200)',
    fine: 'fine (~2000)',
  }

  function getResolution() {
    try {
      var v = window.localStorage.getItem(STORAGE_KEY)
      return VALID_RES.indexOf(v) >= 0 ? v : DEFAULT_RES
    } catch (e) {
      return DEFAULT_RES
    }
  }

  function setResolution(res) {
    if (VALID_RES.indexOf(res) < 0) return false
    var prev = getResolution()
    try {
      window.localStorage.setItem(STORAGE_KEY, res)
    } catch (e) {
      /* storage disabled — still fire the event so a session can react */
    }
    if (prev !== res) {
      var evt
      try {
        evt = new CustomEvent('scix:resolution-change', {
          detail: { resolution: res, previous: prev },
        })
      } catch (e) {
        evt = document.createEvent('CustomEvent')
        evt.initCustomEvent('scix:resolution-change', false, false, {
          resolution: res,
          previous: prev,
        })
      }
      window.dispatchEvent(evt)
    }
    return true
  }

  // Returns the candidate URLs to try for the given (or current) resolution.
  // Pages should attempt them in order — the first successful fetch wins.
  // Legacy filenames are included so old generated files keep working.
  function resolutionFiles(res) {
    var r = res || getResolution()
    if (r === 'coarse') {
      return {
        resolution: 'coarse',
        umap: [
          './umap.coarse.json',
          './umap.json',
          '/viz/umap.coarse.json',
          '/viz/umap.json',
          '/data/viz/umap.json',
        ],
        labels: [
          './community_labels.coarse.json',
          './community_labels.json',
          '/viz/community_labels.coarse.json',
          '/viz/community_labels.json',
        ],
        stream: [
          './stream.coarse.json',
          '/viz/stream.coarse.json',
          '/data/viz/stream.coarse.json',
        ],
      }
    }
    if (r === 'medium') {
      return {
        resolution: 'medium',
        umap: [
          './umap.medium.json',
          '/viz/umap.medium.json',
          '/data/viz/umap.medium.json',
        ],
        labels: [
          './community_labels.medium.json',
          './community_labels_medium.json',
          '/viz/community_labels.medium.json',
          '/viz/community_labels_medium.json',
        ],
        stream: [
          './stream.medium.json',
          '/viz/stream.medium.json',
          '/data/viz/stream.medium.json',
        ],
      }
    }
    return {
      resolution: 'fine',
      umap: ['./umap.fine.json', '/viz/umap.fine.json', '/data/viz/umap.fine.json'],
      labels: [
        './community_labels.fine.json',
        '/viz/community_labels.fine.json',
        '/data/viz/community_labels.fine.json',
      ],
      stream: ['./stream.fine.json', '/viz/stream.fine.json', '/data/viz/stream.fine.json'],
    }
  }

  // Fetch the first URL in a candidate list that responds 2xx. Returns the
  // parsed JSON body or rejects with the last error. Always uses no-store so
  // we don't serve a stale medium file after a resolution switch.
  function fetchFirstAvailable(urls) {
    var candidates = (urls || []).slice()
    function tryNext(i, lastErr) {
      if (i >= candidates.length) {
        return Promise.reject(lastErr || new Error('no candidate URL responded'))
      }
      var url = candidates[i]
      return fetch(url, { cache: 'no-store' })
        .then(function (resp) {
          if (!resp.ok) {
            throw new Error(url + ' -> HTTP ' + resp.status)
          }
          return resp.json()
        })
        .catch(function (err) {
          return tryNext(i + 1, err)
        })
    }
    return tryNext(0, null)
  }

  scixViz.getResolution = getResolution
  scixViz.setResolution = setResolution
  scixViz.resolutionFiles = resolutionFiles
  scixViz.fetchFirstAvailable = fetchFirstAvailable
  scixViz.RESOLUTION_LABELS = RESOLUTION_LABELS
  scixViz.VALID_RESOLUTIONS = VALID_RES.slice()

  // --- Community color palette -------------------------------------------
  // Hand-tuned 20-color categorical palette (d3.schemeCategory10 +
  // d3.schemeTableau10) for coarse resolution where readers expect the
  // "branded" demo colors. For medium (~200) / fine (~2000) resolutions we
  // generate hues on a golden-ratio walk around the HSL wheel so adjacent
  // community ids land far apart and there are no manual tuning gaps.
  var COARSE_PALETTE = [
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207], [78, 121, 167], [242, 142, 43],
    [225, 87, 89], [118, 183, 178], [89, 161, 79], [237, 201, 72],
    [176, 122, 161], [255, 157, 167], [156, 117, 95], [186, 176, 172],
  ]
  var FALLBACK_COLOR_RGB = [160, 160, 160]
  var SENTINEL_COLOR_RGB = [180, 180, 180] // for negative ids (e.g. -1)
  var GOLDEN_RATIO = 0.6180339887498949

  function _hslToRgb(h, s, l) {
    function hue2rgb(p, q, t) {
      if (t < 0) t += 1
      if (t > 1) t -= 1
      if (t < 1 / 6) return p + (q - p) * 6 * t
      if (t < 1 / 2) return q
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
      return p
    }
    if (s === 0) {
      var v = Math.round(l * 255)
      return [v, v, v]
    }
    var q = l < 0.5 ? l * (1 + s) : l + s - l * s
    var p = 2 * l - q
    return [
      Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
      Math.round(hue2rgb(p, q, h) * 255),
      Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
    ]
  }

  function colorForCommunity(cid, opts) {
    if (cid == null || Number.isNaN(Number(cid))) return FALLBACK_COLOR_RGB.slice()
    var n = Number(cid)
    if (n < 0) return SENTINEL_COLOR_RGB.slice()
    var res = (opts && opts.resolution) || getResolution()
    if (res === 'coarse') {
      var idx = ((n % COARSE_PALETTE.length) + COARSE_PALETTE.length) % COARSE_PALETTE.length
      return COARSE_PALETTE[idx].slice()
    }
    // Hue varies on golden ratio; saturation/lightness held constant for
    // calm, equal-perceptual-weight chips. h ∈ [0,1).
    var hue = (n * GOLDEN_RATIO) % 1
    if (hue < 0) hue += 1
    return _hslToRgb(hue, 0.62, 0.55)
  }

  scixViz.colorForCommunity = colorForCommunity
  scixViz.COARSE_PALETTE = COARSE_PALETTE.map(function (c) { return c.slice() })

  // --- Navigation ---------------------------------------------------------
  // Inject a small top navigation bar so the viz pages are linked, plus a
  // resolution-picker that drives the helpers above. Runs on
  // DOMContentLoaded so each page gets it without having to include
  // boilerplate HTML. Skipped on index.html (which IS the nav) and when the
  // bar has already been rendered.
  function injectNav() {
    if (document.getElementById('scix-nav')) return
    var path = (location.pathname || '').toLowerCase()
    if (path.endsWith('/') || path.endsWith('/index.html')) return

    var nav = document.createElement('nav')
    nav.id = 'scix-nav'
    nav.style.cssText =
      'position:sticky;top:0;z-index:5;background:rgba(255,255,255,0.96);' +
      'border-bottom:1px solid #e6e6e6;padding:6px 16px;font-size:13px;' +
      'display:flex;gap:14px;align-items:center;backdrop-filter:blur(6px)'

    var home = document.createElement('a')
    home.href = './index.html'
    home.textContent = 'SciX viz'
    home.style.cssText = 'color:#333;text-decoration:none;font-weight:600'
    nav.appendChild(home)

    var pages = [
      { href: './sankey.html', label: 'V2 Sankey' },
      { href: './umap_browser.html', label: 'V3 UMAP' },
      { href: './agent_trace.html', label: 'V4 Trace' },
      { href: './heatmap.html', label: 'V5 Topology' },
      { href: './ego.html', label: 'V6 Ego' },
      { href: './streamgraph.html', label: 'V7 Streams' },
    ]
    pages.forEach(function (p) {
      var a = document.createElement('a')
      a.href = p.href
      a.textContent = p.label
      var active = path.endsWith(p.href.replace('./', '/'))
      a.style.cssText =
        'color:' +
        (active ? '#0b60b0' : '#555') +
        ';text-decoration:none;padding:3px 6px;border-radius:3px;' +
        (active ? 'background:#eef4fa;font-weight:600;' : '')
      nav.appendChild(a)
    })

    // --- Resolution toggle -------------------------------------------------
    var spacer = document.createElement('span')
    spacer.style.cssText = 'flex:1'
    nav.appendChild(spacer)

    var resLabel = document.createElement('span')
    resLabel.textContent = 'Resolution:'
    resLabel.style.cssText = 'color:#666;font-size:11px;margin-right:6px'
    nav.appendChild(resLabel)

    var resWrap = document.createElement('span')
    resWrap.id = 'scix-res-toggle'
    resWrap.setAttribute('role', 'tablist')
    resWrap.setAttribute('aria-label', 'Community resolution')
    resWrap.style.cssText = 'display:inline-flex;gap:4px'
    nav.appendChild(resWrap)

    var current = getResolution()
    var buttons = []
    VALID_RES.forEach(function (res) {
      var btn = document.createElement('button')
      btn.type = 'button'
      btn.setAttribute('role', 'tab')
      btn.dataset.res = res
      btn.textContent = res.charAt(0).toUpperCase() + res.slice(1)
      btn.title = RESOLUTION_LABELS[res] || res
      btn.setAttribute('aria-selected', String(res === current))
      btn.style.cssText =
        'font-size:11px;padding:2px 8px;border:1px solid #cfcfcf;' +
        'background:' +
        (res === current ? '#0b60b0' : '#fafafa') +
        ';color:' +
        (res === current ? '#fff' : '#333') +
        ';border-radius:3px;cursor:pointer'
      btn.addEventListener('click', function () {
        if (setResolution(res)) {
          syncButtons(res)
        }
      })
      buttons.push(btn)
      resWrap.appendChild(btn)
    })

    function syncButtons(active) {
      buttons.forEach(function (btn) {
        var on = btn.dataset.res === active
        btn.setAttribute('aria-selected', String(on))
        btn.style.background = on ? '#0b60b0' : '#fafafa'
        btn.style.color = on ? '#fff' : '#333'
      })
    }

    // Keep the toggle in sync if another tab/window changes the choice.
    window.addEventListener('storage', function (e) {
      if (e.key === STORAGE_KEY) syncButtons(getResolution())
    })
    // Pages may also call setResolution programmatically — stay in sync.
    window.addEventListener('scix:resolution-change', function (e) {
      var next = (e && e.detail && e.detail.resolution) || getResolution()
      syncButtons(next)
    })

    if (document.body.firstChild) {
      document.body.insertBefore(nav, document.body.firstChild)
    } else {
      document.body.appendChild(nav)
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectNav)
  } else {
    injectNav()
  }
})()
