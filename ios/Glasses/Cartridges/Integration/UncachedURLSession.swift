//
//  UncachedURLSession.swift
//  Glasses
//

import Foundation

/// The session an HTTP client defaults to when what it fetches must leave no
/// copy behind: Object Memory's first-person frames, CV Lab's live previews,
/// World Builder's room imagery and tiles.
///
/// Those routes send `Cache-Control: no-store`, and honouring that must not
/// depend on which session a caller happened to pass. So the configuration is
/// **ephemeral** (nothing on disk), has **`urlCache = nil`** (no in-memory
/// cache either), and asks for `.reloadIgnoringLocalAndRemoteCacheData`.
///
/// Everything else is `.ephemeral`'s, cookies included: an in-memory jar that
/// dies with the session. `WorldAssetClient.uncachedConfiguration()` removes
/// the jar as well.
nonisolated enum UncachedURLSession {
    /// A new configuration on every call, so a caller can add to its copy
    /// without changing anyone else's.
    static func configuration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return configuration
    }

    /// A new session with that configuration.
    static func make() -> URLSession {
        URLSession(configuration: configuration())
    }
}
