/* U1.1 NATIVE CHROME: the page side of the bridge (WORLD-BUILDER-WORLDS.md §4c, protocol 1).

   Served only inside the native-chrome variant of the appearance page
   (`native_chrome.py`, insertion B), as a second classic script after the
   page's own. It reads the page; it draws nothing and moves nothing. The page
   stays the only authority for when each state is true: this reports what the
   page's own chrome shows, as data, to the phone, which draws it natively.

   Transport. `window.webkit.messageHandlers.wbChromeV1.postMessage`, a script
   message handler WITH REPLIES. The app runs no script in this page: every
   command it has for the page is the reply to an `await` message the page keeps
   outstanding. No handler -- a desktop browser, `transport=tower`, an older
   app -- and this returns at once and posts nothing, and the page keeps its web
   chrome.

   Shape. A pure unit, `WBCHROME` (labels, snapshot, sameState, viewChanged,
   actionTarget, clamp, captionOf), in the style of the page's FOLLOW and NAV so
   node can test it, and a thin DOM adapter. Everything is inside try/catch: the
   bridge never throws into the page, and as a separate script it cannot stop
   `main()` either. */
"use strict";
const WBCHROME = (() => {
  const PROTOCOL = 1;
  const HANDLER = "wbChromeV1";
  // Table C `action` names -> the page's own buttons. The ring, the edge
  // chevron and the dark line are all `face` on the phone, as they all call
  // `faceTheRoom()` here.
  const ACTIONS = Object.freeze({best: "bOverview", face: "bBack", previous: "bPrev",
                                 next: "bNext", reset: "bReset"});
  // Table H / S bounds, in UTF-16 code units (never more code points or
  // characters than that), so the phone never refuses a message for length.
  const LIMITS = Object.freeze({text: 40, name: 120, ringName: 120, ringLabel: 60, center: 8,
    darkTitle: 80, darkTap: 80, darkName: 160, edge: 80, about: 20, marker: 200,
    status: 200, message: 800, hint: 200, walk: 24, head: 300, line: 600, title: 80,
    body: 1600, tail: 200, sections: 8});
  // Words the page draws on a canvas or writes from code, not into the DOM.
  // Tower test N11 pins each to the template's own line.
  const RING_CENTER = "YOU";              // g.fillText("YOU", c, c);
  const ABOUT_OPEN = "About", ABOUT_CLOSE = "Less";
  const VIEW_EPS = 1e-3;
  // `view` at most every 34 ms (under 30 a second) at any frame rate, so the
  // bridge stays inside spec §6's 70 messages a second on a 120 Hz screen
  const VIEW_MIN_MS = 34;

  function clamp(text, n){
    if (text === null || text === undefined) return null;
    const s = String(text);
    if (s.length <= n) return s;
    let cut = s.slice(0, n);
    const last = cut.charCodeAt(cut.length - 1);
    if (last >= 0xD800 && last <= 0xDBFF) cut = cut.slice(0, -1);   // never half a pair
    return cut;
  }
  const L = LIMITS;
  const label = (r, id) => ({text: clamp(r.text(id), L.text) || "",
                             name: clamp(r.attr(id, "aria-label"), L.name) || ""});

  /* Table H. `reader` is the page as data (`domReader`, or a test's stub). */
  function labels(r){
    const out = {
      best: label(r, ACTIONS.best), face: label(r, ACTIONS.face),
      previous: label(r, ACTIONS.previous), next: label(r, ACTIONS.next),
      reset: label(r, ACTIONS.reset),
      ring: {name: clamp(r.attr("compass", "aria-label"), L.ringName) || "",
             label: clamp(r.lines("clabel").join("\n"), L.ringLabel) || "",
             center: RING_CENTER},
      about: {open: ABOUT_OPEN, close: ABOUT_CLOSE},
    };
    // Only from the page's own hooks. Without them (WebGL failed before `main()`
    // reached them) `hello` lacks both, the phone declines, and the page keeps
    // its web chrome over its own failure message.
    const w = r.words();
    if (w){
      out.dark = {title: clamp(w.dark, L.darkTitle) || "", tap: clamp(w.darkTap, L.darkTap) || "",
                  name: clamp(r.attr("dark", "aria-label"), L.darkName) || ""};
      out.edge = clamp(w.edge, L.edge) || "";
    }
    return out;
  }
  function research(r){
    return {raw: !!r.raw(), marker: r.hidden("rawmark") ? null : clamp(r.text("rawmark"), L.marker)};
  }
  function hello(r){
    return {protocol: [PROTOCOL], kind: r.kind() === "area" ? "area" : "room",
            labels: labels(r), research: research(r)};
  }

  /* The caption, as `updateCaption` builds it: `head` is the <b>; inside
     `.more`, the <i> is the count line, each <em> a section title with the
     <span> after it, and `.tail` the tail. Null while the caption is empty. */
  function captionOf(cap){
    if (!cap) return null;
    const kids = Array.from(cap.children || []);
    const head = kids.find(k => k.tagName === "B");
    if (!head) return null;
    const more = kids.find(k => k.classList && k.classList.contains("more"));
    const parts = more ? Array.from(more.children || []) : [];
    const line = parts.find(k => k.tagName === "I");
    const tail = parts.find(k => k.classList && k.classList.contains("tail"));
    const sections = [];
    parts.forEach((k, i) => {
      if (k.tagName !== "EM") return;
      const next = parts[i + 1];
      const body = next && next.tagName === "SPAN" && !(next.classList && next.classList.contains("tail"))
        ? next.textContent : "";
      sections.push({title: clamp(k.textContent, L.title) || "", body: clamp(body, L.body) || ""});
    });
    if (!sections.length) return null;
    return {head: clamp(head.textContent, L.head) || "",
            line: line ? clamp(line.textContent, L.line) : null,
            sections: sections.slice(0, L.sections),
            tail: tail ? clamp(tail.textContent, L.tail) : null};
  }

  /* Table S: a COMPLETE snapshot of what the chrome shows. */
  function snapshot(r){
    const v = r.view();
    const hintOn = r.has("hint", "on");
    let opacity = hintOn ? parseFloat(r.style("hint", "opacity")) : 0;
    if (!(opacity > 0)) opacity = 0.9;
    const edgeOn = r.has("back", "on");
    const phase = r.phase();
    return {
      active: !!r.active(),
      phase: typeof phase === "string" ? phase : "starting",
      drawn: !!(v && v.drawn), holding: !!r.holding(), restoring: !!(v && v.restoring),
      status: clamp(r.text("status"), L.status) || null,
      message: r.has("msg", "on") ? (clamp(r.text("msg"), L.message) || null) : null,
      hint: hintOn ? {text: clamp(r.text("hint"), L.hint) || "", opacity: Math.min(1, opacity)} : null,
      dark: r.has("dark", "on"),
      edge: edgeOn && r.has("back", "left") ? "left" : edgeOn && r.has("back", "right") ? "right" : null,
      buttons: {best: !r.disabled(ACTIONS.best), face: !r.disabled(ACTIONS.face),
                previous: !r.disabled(ACTIONS.previous), next: !r.disabled(ACTIONS.next),
                reset: !r.disabled(ACTIONS.reset)},
      walk: clamp(r.text("pos"), L.walk) || null,
      ring: v ? {shown: r.has("compass", "on"), lit: v.lit ? Array.from(v.lit) : null,
                 sense: v.sense < 0 ? -1 : 1}
              : {shown: false, lit: null, sense: 1},
      caption: r.caption(),
      research: research(r),
    };
  }
  const sameState = (a, b) => !!a && !!b && JSON.stringify(a) === JSON.stringify(b);
  function viewChanged(prev, next){
    if (!next) return false;
    if (!prev) return true;
    return Math.abs(next.headingRad - prev.headingRad) > VIEW_EPS
        || Math.abs(next.halfFovRad - prev.halfFovRad) > VIEW_EPS;
  }
  const actionTarget = name =>
    Object.prototype.hasOwnProperty.call(ACTIONS, name) ? ACTIONS[name] : null;
  const viewOf = v => (v && Number.isFinite(v.headingRad) && v.halfFovRad > 0
                       && v.halfFovRad < Math.PI / 2)
    ? {headingRad: v.headingRad, halfFovRad: v.halfFovRad} : null;

  /* The page as data, read from the DOM and the page's report object `S`. */
  function domReader(doc, S, config, raw, captionCache){
    const $ = id => doc.getElementById(id);
    return {
      text: id => { const e = $(id); return e ? e.textContent : null; },
      attr: (id, name) => { const e = $(id); return e ? e.getAttribute(name) : null; },
      has: (id, cls) => { const e = $(id); return !!(e && e.classList.contains(cls)); },
      disabled: id => { const e = $(id); return !e || !!e.disabled; },
      hidden: id => { const e = $(id); return !e || !!e.hidden; },
      style: (id, prop) => { const e = $(id); return e ? e.style[prop] : ""; },
      lines: id => { const e = $(id); if (!e) return [];
        return Array.from(e.childNodes || []).filter(n => n.nodeType === 3)
          .map(n => n.textContent.trim()).filter(Boolean); },
      words: () => (S && typeof S.chromeWords === "function") ? S.chromeWords() : null,
      view: () => (S && typeof S.chromeView === "function") ? S.chromeView() : null,
      phase: () => S ? S.phase : null,
      holding: () => !!(S && S.holding),
      raw: () => !!raw,
      kind: () => (config && config.area) ? "area" : "room",
      active: () => doc.body.classList.contains("wbnative"),
      caption: () => captionCache(() => captionOf($("caption"))),
    };
  }

  function randomId(){
    const abc = "abcdefghijklmnopqrstuvwxyz0123456789";
    let out = "";
    try {
      const a = new Uint8Array(16);
      crypto.getRandomValues(a);
      for (const b of a) out += abc[b % 36];
      return out;
    } catch (_){
      for (let i = 0; i < 16; i++) out += abc[Math.floor(Math.random() * 36)];
      return out;
    }
  }

  /* The adapter: the lifecycle of WORLDS §4c on one page load. */
  function start(win, doc){
    const S = win.__wbAppearance || null;
    const handler = win.webkit && win.webkit.messageHandlers && win.webkit.messageHandlers[HANDLER];
    if (!handler || typeof handler.postMessage !== "function"){
      if (S) S.chromeBridge = () => ({handler: false});
      return null;
    }
    const config = (typeof CONFIG !== "undefined") ? CONFIG : null;          // eslint-disable-line no-undef
    const raw = (typeof RAW_IMAGERY !== "undefined") ? RAW_IMAGERY : false;  // eslint-disable-line no-undef
    let caption = null, captionStale = true;
    const reader = domReader(doc, S, config, raw, read => {
      if (captionStale){ caption = read(); captionStale = false; }
      return caption;
    });
    const pageId = randomId();
    let seq = 0, nonce = null, dead = false, awaiting = false;
    let last = null, stateInFlight = false, dirty = false;
    let lastView = null, latestView = null, viewInFlight = false, viewTimer = null, lastViewAt = -Infinity;
    const now = () => (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
    let frameKey = null, observer = null, scheduled = false, sweep = null;
    const posted = {hello: 0, state: 0, view: 0, await: 0};

    function post(type, body){
      const msg = Object.assign({v: 1, type, pageId, seq: seq++},
                                nonce !== null ? {nonce} : {}, body || {});
      posted[type] = (posted[type] || 0) + 1;
      try {
        const r = handler.postMessage(msg);
        return (r && typeof r.then === "function") ? r : Promise.reject(new Error("no reply"));
      } catch (e){
        return Promise.reject(e);
      }
    }
    /* Stops the bridge for good. `restore` puts the page's own chrome back: every
       failure ends on the web chrome, never on a page with no chrome at all. */
    function stop(restore){
      dead = true;
      if (observer){ try { observer.disconnect(); } catch (_){ /* gone */ } observer = null; }
      if (sweep !== null){ clearInterval(sweep); sweep = null; }
      if (viewTimer !== null){ clearTimeout(viewTimer); viewTimer = null; }
      if (S && S.chromeFrame === onFrame) S.chromeFrame = undefined;
      if (restore) doc.body.classList.remove("wbnative");
    }
    function flush(){
      if (nonce === null || dead) return;
      if (stateInFlight){ dirty = true; return; }
      let snap;
      try { snap = snapshot(reader); } catch (_){ return; }
      if (sameState(snap, last)) return;
      last = snap; stateInFlight = true;
      post("state", snap).then(landed, () => { stateInFlight = false; stop(true); });
    }
    /* `deactivate`'s report: `active:false`, posted at once and exactly once. It
       waits on nothing -- not on a `state` whose reply is still pending (that
       reply may hang, or be rejected), not on differing from the last one --
       because the phone waits for it on every deactivate and reloads the page
       without it. Nothing is posted after it. */
    function confirmInactive(){
      let snap;
      try { snap = snapshot(reader); } catch (_){ return; }
      last = snap;
      post("state", snap).then(() => {}, () => {});
    }
    function landed(){
      stateInFlight = false;
      if (dirty){ dirty = false; flush(); }
    }
    /* At most one `view` in flight, and at most one per VIEW_MIN_MS whatever the
       display's frame rate (a 120 Hz screen draws 120 frames a second). A move
       inside the gap is held, not dropped: the timer posts the latest heading. */
    function postView(){
      if (dead || viewInFlight || viewTimer !== null || !viewChanged(lastView, latestView)) return;
      const wait = lastViewAt + VIEW_MIN_MS - now();
      if (wait > 0){
        viewTimer = setTimeout(() => { viewTimer = null; postView(); }, wait);
        return;
      }
      lastView = latestView; viewInFlight = true; lastViewAt = now();
      post("view", lastView).then(() => { viewInFlight = false; postView(); },
                                  () => { viewInFlight = false; stop(true); });
    }
    /* Called by insertion F at the end of every drawn frame. */
    function onFrame(){
      if (dead) return;
      try {
        const v = reader.view();
        const view = viewOf(v);
        if (view){ latestView = view; postView(); }
        const key = JSON.stringify([!!(v && v.drawn), !!(v && v.restoring),
                                    v && v.lit ? Array.from(v.lit) : null, v ? v.sense : 1,
                                    reader.text("pos"), reader.phase(), reader.holding()]);
        if (key !== frameKey){ frameKey = key; flush(); }
      } catch (_){ /* never into the page's frame */ }
    }
    function schedule(){
      if (scheduled) return;
      scheduled = true;
      Promise.resolve().then(() => { scheduled = false; if (!dead) flush(); });
    }
    function observe(){
      if (typeof MutationObserver !== "function") return;
      const cap = doc.getElementById("caption");
      observer = new MutationObserver(records => {
        for (const m of records){
          const t = m.target;
          if (cap && (t === cap || (typeof cap.contains === "function" && cap.contains(t)))){
            captionStale = true;
            break;
          }
        }
        schedule();
      });
      observer.observe(doc.body, {attributes: true, attributeFilter: ["class", "disabled", "hidden", "style"],
                                  childList: true, subtree: true, characterData: true});
    }
    function command(reply){
      const type = reply && reply.v === 1 && typeof reply.type === "string" ? reply.type : null;
      if (type === "activate"){ doc.body.classList.add("wbnative"); return true; }
      if (type === "action"){
        const id = actionTarget(reply.name);
        const el = id && doc.getElementById(id);
        if (el) el.click();                  // a disabled button does not fire
        return true;
      }
      if (type === "deactivate"){
        // the page's chrome back, `active:false` reported, then nothing more.
        // ALWAYS reported: a phone that deactivates before it ever activated
        // (an invalid first `state`, its timeout) waits for this confirmation,
        // and would otherwise reload the page for want of it.
        stop(false);
        doc.body.classList.remove("wbnative");
        confirmInactive();
        return true;
      }
      if (type === "close"){ stop(false); return true; }
      return false;
    }
    function awaitNext(){
      if (dead || awaiting) return;
      awaiting = true;
      post("await").then(reply => {
        awaiting = false;
        // a reply that lands after the bridge has ended (a failed `state` or
        // `view` stopped it) is not obeyed: a late `activate` would hide the
        // page's chrome with no bridge left to draw it
        if (dead) return;
        let known = false;
        try { known = command(reply); } catch (_){ /* a command never throws into the page */ }
        if (dead) return;
        // an unknown reply is ignored; a short pause keeps a confused peer from
        // turning the loop into a busy one
        if (known) awaitNext(); else setTimeout(awaitNext, 250);
      }, () => { awaiting = false; stop(true); });
    }

    if (S) S.chromeBridge = () => ({handler: true, active: doc.body.classList.contains("wbnative"),
                                    welcomed: nonce !== null, dead, pageId,
                                    posted: Object.assign({}, posted), last});
    let first;
    try { first = hello(reader); } catch (_){ stop(false); return null; }
    post("hello", first).then(reply => {
      if (dead) return;
      if (!(reply && reply.v === 1 && reply.type === "welcome" && reply.protocol === PROTOCOL
            && typeof reply.nonce === "string" && reply.nonce.length > 0 && reply.nonce.length <= 128)){
        stop(false);                         // `decline`, or anything else: web chrome for this load
        return;
      }
      nonce = reply.nonce;
      if (S) S.chromeFrame = onFrame;
      observe();
      // A safety net, not the mechanism: the page changes a few reported fields
      // (`S.holding` cleared by a quiet poll) with no DOM change and, when still,
      // no frame. A snapshot is cheap and is posted only when it differs.
      try { sweep = setInterval(() => { if (!dead) flush(); }, 1000); } catch (_){ sweep = null; }
      flush();
      awaitNext();
    }, () => stop(false));
    try { win.addEventListener("pagehide", () => stop(false)); } catch (_){ /* no events */ }
    return {pageId};
  }

  return {PROTOCOL, HANDLER, ACTIONS, LIMITS, RING_CENTER, ABOUT_OPEN, ABOUT_CLOSE, VIEW_MIN_MS,
          clamp, labels, hello, snapshot, sameState, viewChanged, actionTarget, captionOf,
          domReader, start};
})();
try { WBCHROME.start(window, document); } catch (_){ /* the page keeps its web chrome */ }
