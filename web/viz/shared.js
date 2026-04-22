/* SciX viz shared helpers. Later visualizations attach utilities here. */
(function () {
  'use strict'
  if (typeof window !== 'undefined') {
    window.scixViz = window.scixViz || {}
  }

  // Inject a small top navigation bar so the three views are linked.
  // Runs on DOMContentLoaded so each page gets it without having to include
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
