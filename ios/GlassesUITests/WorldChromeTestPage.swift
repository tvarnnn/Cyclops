//
//  WorldChromeTestPage.swift
//  GlassesUITests
//
//  U1.1: a text-only stand-in for the appearance page's native-chrome
//  variant, served by `MockTowerHTTPServer`. It carries the appearance
//  page's exact CSP `<meta>`, the `wb-representation`, `wb-revision` and
//  (unless told not to) `wb-chrome` echo metas, and an inline script that
//  speaks the page side of protocol 1 (WORLD-BUILDER-WORLDS.md §4c) as the
//  Tower's `appearance_chrome_bridge.js` does: every field at the top level,
//  `seq` from 0 at `hello`, one outstanding `await`.
//
//  It steps through a scripted list of states: one step per `action` the
//  phone sends, with the action echoed into the next status (`got next`),
//  or by itself after a step's `after` milliseconds. Its own chrome is one
//  line, "PAGE CHROME", hidden while `body.wbnative` is set, so a test can
//  tell whose chrome is on screen.
//
//  The default labels are deliberately NOT the page's words ("BestT",
//  "Best view T: fly"…): the native chrome is proved to draw only what it
//  was sent. `realLabels` are the Tower's own words, for the layout tests,
//  whose widths must be the real ones.
//

import Foundation

enum WorldChromeTestPage {
    /// One scripted state: Table S's fields, and an optional self-advance.
    struct Step {
        var state: [String: Any]
        var after: Int? = nil
    }

    enum Hello {
        /// A valid `hello`.
        case valid
        /// `labels.dark` and `labels.edge` missing (the page's hooks did not
        /// run): the phone must decline.
        case missingHooks
        /// No `hello` at all: the phone must give up after 5 s.
        case silent
    }

    static let areaID = "a1b2c3d4e5f60718"

    static func testLabels(kind: String) -> [String: Any] {
        [
            "best": ["text": "BestT", "name": "Best view T: fly"],
            "face": kind == "area" ? ["text": "FaceAreaT", "name": "Face the area T"]
                                   : ["text": "FaceT", "name": "Face the room T"],
            "previous": ["text": "PrevT", "name": "Previous T"],
            "next": ["text": "NextT", "name": "Next T"],
            "reset": ["text": "ResetT", "name": "Reset T"],
            "ring": ["name": "Ring name T", "label": "ringT\nlabelT", "center": "UT"],
            "about": ["open": "AboutT", "close": "LessT"],
            "dark": ["title": "Dark title T", "tap": "Dark tap T", "name": "Dark name T"],
            "edge": "Edge T",
        ]
    }

    /// The Tower's own words (`appearance_viewer.html` @ cdc84db).
    static func realLabels(kind: String) -> [String: Any] {
        let face = kind == "area" ? "Face the area" : "Face the room"
        return [
            "best": ["text": "Best view", "name": "Best view: fly to the clearest vantage"],
            "face": ["text": face, "name": "\(face): turn to the nearest reconstructed direction"],
            "previous": ["text": "\u{2190}", "name": "Previous recorded view"],
            "next": ["text": "\u{2192}", "name": "Next recorded view"],
            "reset": ["text": "Reset", "name": "Reset: return to the opening view"],
            "ring": ["name": "Which directions are reconstructed from here",
                     "label": "reconstructed\nfrom here", "center": "YOU"],
            "about": ["open": "About", "close": "Less"],
            "dark": ["title": "Not reconstructed from here", "tap": "Tap to turn back",
                     "name": "Not reconstructed from here. Tap to turn back"],
            "edge": "Movement stops here",
        ]
    }

    /// A Table S snapshot with sensible defaults; `edit` changes any field.
    static func state(drawn: Bool = true, status: Any = NSNull(), message: Any = NSNull(),
                      hint: Any = NSNull(), dark: Bool = false, edge: Any = NSNull(),
                      buttons: [String: Bool] = ["best": true, "face": true, "previous": true,
                                                 "next": true, "reset": true],
                      walk: Any = "1 / 3", holding: Bool = false, phase: String? = nil,
                      raw: Bool = false, marker: Any = NSNull(),
                      sections: Int = 2, head: String? = nil) -> [String: Any] {
        [
            "active": false, "phase": phase ?? (drawn ? "ready" : "loading"), "drawn": drawn,
            "holding": holding, "restoring": false, "status": status, "message": message, "hint": hint,
            "dark": dark, "edge": edge, "buttons": buttons, "walk": drawn ? walk : NSNull(),
            "ring": ["shown": drawn, "lit": (0..<36).map { i in Double(i % 9) / 8 }, "sense": 1],
            "caption": drawn ? [
                "head": head ?? (raw ? "RESEARCH T head line" : "Head T: captured images"),
                "line": "Line T: 24 of 24 images",
                "sections": (0..<sections).map { ["title": "Section \($0 + 1) T", "body": "Body \($0 + 1) T."] },
                "tail": "Tail T",
            ] as [String: Any] : NSNull(),
            "research": ["raw": raw, "marker": marker],
        ]
    }

    static func html(steps: [Step], kind: String = "room", echo: Bool = true, hello: Hello = .valid,
                     labels: [String: Any]? = nil, raw: Bool = false, marker: Any = NSNull()) -> String {
        let stepsJSON = json(steps.map { step -> [String: Any] in
            var entry: [String: Any] = ["state": step.state]
            if let after = step.after { entry["after"] = after }
            return entry
        })
        var helloLabels = labels ?? testLabels(kind: kind)
        if hello == .missingHooks {
            helloLabels["dark"] = nil
            helloLabels["edge"] = nil
        }
        let helloJSON = json(["protocol": [1], "kind": kind, "labels": helloLabels,
                              "research": ["raw": raw, "marker": marker]])
        let revision = kind == "area" ? "s1/area:\(areaID)/appearance:1@e1" : "s1/appearance:1@e1"
        let areaMeta = kind == "area" ? #"<meta name="wb-area" content="\#(areaID)">"# : ""
        let echoMeta = echo ? #"<meta name="wb-chrome" content="native">"# : ""
        return """
        <!doctype html><html><head><meta charset="utf-8">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src glasses-world:">
        <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
        <meta name="wb-representation" content="appearance"><meta name="wb-revision" content="\(revision)">\(areaMeta)\(echoMeta)
        <title>chrome test page</title>
        <style>html,body{margin:0;height:100%;background:#0b0d10;color:#e8e9ec;font:14px -apple-system}
        #wrap{position:fixed;inset:0;background:linear-gradient(#335,#0b0d10)}
        #pagechrome{position:fixed;left:8px;bottom:8px;padding:6px;background:#222}
        body.wbnative #pagechrome{display:none!important}</style></head>
        <body><div id="wrap"></div><div id="pagechrome">PAGE CHROME</div>
        <script>
        "use strict";
        (function(){
          const STEPS = \(stepsJSON);
          const HELLO = \(helloJSON);
          const MODE = "\(hello == .silent ? "silent" : "talk")";
          const h = window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.wbChromeV1;
          if (!h || MODE === "silent") return;
          const pageId = "testpage" + Math.random().toString(36).slice(2, 10).replace(/[^a-z0-9]/g, "0");
          let seq = 0, nonce = null, i = 0, got = null, dead = false;
          function post(type, body){
            const m = Object.assign({v: 1, type: type, pageId: pageId, seq: seq++}, nonce ? {nonce: nonce} : {}, body || {});
            return h.postMessage(m);
          }
          function snapshot(){
            const s = JSON.parse(JSON.stringify(STEPS[i].state));
            s.active = document.body.classList.contains("wbnative");
            if (got) s.status = "got " + got;
            return s;
          }
          function sendState(){ if (!dead) post("state", snapshot()).catch(() => { dead = true; }); }
          function schedule(){
            const after = STEPS[i].after;
            if (after) setTimeout(() => { got = null; step(); }, after);
          }
          function step(){ if (i < STEPS.length - 1){ i++; schedule(); } sendState(); }
          async function loop(){
            while (!dead){
              let r;
              try { r = await post("await"); } catch (e){ dead = true; break; }
              if (!r || r.v !== 1) continue;
              if (r.type === "activate"){ document.body.classList.add("wbnative"); sendState(); }
              else if (r.type === "action"){ got = r.name; step(); }
              else if (r.type === "deactivate"){ document.body.classList.remove("wbnative"); sendState(); dead = true; }
              else if (r.type === "close"){ dead = true; }
            }
          }
          // A touch that reaches the canvas says so in the next status, so a
          // test proves empty chrome space passes touches through.
          document.getElementById("wrap").addEventListener("pointerdown", () => {
            if (nonce && !dead){ got = "touch"; sendState(); }
          });
          post("hello", HELLO).then(r => {
            if (!(r && r.type === "welcome" && r.protocol === 1 && r.nonce)) return;
            nonce = r.nonce;
            sendState();
            schedule();
            post("view", {headingRad: 0.4, halfFovRad: 0.5}).catch(() => {});
            loop();
          }, () => {});
        })();
        </script></body></html>
        """
    }

    /// The one-world listing (`w1`/`s1`, "Appearance fixture (Mac)"), with
    /// one area that opens when `withArea`, and the walk's
    /// `finalization.notice` (COMPONENTS §8) when one is given. `notice` is
    /// plain prose: no quotes or backslashes to escape.
    static func listing(withArea: Bool = false, notice: String? = nil) -> String {
        let noticeField = notice.map { #""notice":"\#($0)","# } ?? ""
        return """
        {"contract":"world_builder.worlds/2026-09-10","world_count":1,"worlds":[{"created_at":1790048283.193669,"display_name":"Appearance fixture (Mac)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":{"bytes":239242,"format":"wb-appearance-keyframes/1","imagery":"first-person keyframe imagery of a private space; best-effort face redaction with measured false negatives; not anonymised; screens, documents and bodies are not redacted","imagery_source":"redacted","keyframe_image_set":null,"keyframes":24,"keyframes_phone":24,"label_trusted":true,"privacy_safe":true,"privacy_tags":["raw-imagery","first-person"],"quality":"final","redaction":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","redaction_effective":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","retains_raw_imagery":true,"retention":"kept with the world","state":"served"},"capture_id":"appearance-fixture","dense":null,"end_reason":"stop","ended_at":1790048583.193669,"finalization":{"detail":null,"final_solve":"solved",\(noticeField)"started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"frame_source":"synthetic-fixture","has_geometry":true,"keyframes_accepted":24,"keyframes_journaled":0,"photographic":{"detail":"d","stage":"appearance","state":"complete"},"session_id":"s1","started_at":1790048283.193669,"state":"complete"\(withArea ? ",\"components\":\(components)" : "")}],"updated_at":1790048883.193669,"world_id":"w1"}]}
        """
    }

    static let components = """
        [{"id":"0f1e2d3c4b5a6978","state":"placed","reason":null,"reasons":[],"shown_as":"room","keyframes":20,"keyframes_phone":20,"capture_spans_s":[[0.0,60.0]],"has_geometry":true,"photographic":{"state":"complete","stage":"appearance","detail":"d"}},{"id":"\(areaID)","state":"unplaced","reason":"solved-separately","reasons":["solved-separately"],"shown_as":"area","keyframes":12,"keyframes_phone":12,"capture_spans_s":[[60.0,90.0]],"has_geometry":true,"photographic":{"state":"complete","stage":"appearance","detail":"d"}}]
        """

    static func revision(withArea: Bool = false) -> String {
        """
        {"session_id":"s1","representation":"appearance","revision":"s1/appearance:1@e1","live":false,"appearance":{"revision":"s1/appearance:b1","current":true,"state":"served","epoch":"e1"}\(withArea ? ",\"components\":\(components)" : "")}
        """
    }

    static let areaRevision = """
        {"session_id":"s1","representation":"appearance","revision":"s1/area:\(areaID)/appearance:1@e1","live":false,"appearance":{"revision":"s1/area:\(areaID)/appearance:b1","current":true,"state":"served","epoch":"e1"}}
        """

    private static func json(_ value: Any) -> String {
        let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        // Safe inside a <script>: no "</" can close it.
        return String(decoding: data, as: UTF8.self).replacingOccurrences(of: "</", with: "<\\/")
    }
}
