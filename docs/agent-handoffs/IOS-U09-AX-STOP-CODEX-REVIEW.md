# U0.9 AX capture and onboarding review note

**Status:** CODEX-ACTING (unreviewed by Claude). Isolated `codex/u09-stop-onboarding` branch from clean `ios/ux-v1` `0f1249e`; no merge, deployment, or phone install.

## Why these changes

The September 29 U0.9 tour shows the World Builder Stop button at the foot of the first viewport on a 17 Pro at default size, but only after five scrolls at AX5. All four AX5 onboarding cards spend the opening viewport on a large decorative symbol, title, and lead. Cards 1, 3, and 4 show only part of the lead above the fixed Continue or Get started footer.

## Changed files

- `ios/Glasses/ContentView.swift`: one DEBUG Stop button in a bottom safe-area inset while World Builder is selected and a camera claim is held. Its small leaf observes `GlassesConnection`; the root does not observe frames. The inset reserves space for content to scroll above it. Stop is disabled during teardown.
- `ios/Glasses/Workspaces/WorldBuilder/WorldBuilderWorkspaceView.swift`: removes the duplicate Stop from the scrolling workspace and keeps Start there. The paused explanation remains; teardown says it is stopping instead of offering Start.
- `ios/Glasses/Workspaces/WorldBuilder/WorldCanvasView.swift` and `WorldBuilderSessionController.swift`: keep the guidance and source comment accurate now that the status line is no longer under Stop.
- `ios/Glasses/Onboarding.swift` and `ios/Glasses/Views/OnboardingView.swift`: each card has a short AX lead. At AX sizes the decorative hero symbol is hidden and the title uses `.title`; point rows and the fixed footer remain. Default-size copy and layout stay as before.
- `ios/GlassesTests/OnboardingTests.swift`, `ios/GlassesUITests/OnboardingUITests.swift`, `ios/GlassesUITests/DeadEndsUITests.swift`: check AX lead content and first-viewport placement, and require one reachable Stop at AX5 after scrolling to the top and bottom.

## Verification and gaps

- `git diff --check`: passed.
- `swiftc -frontend -parse` on all modified Swift files: passed. This checks syntax only.
- Generic iOS Debug device build: `** BUILD SUCCEEDED **` with signing disabled, the usual redirected caches, and copied prior package cache (`/private/tmp/glasses-codex-u09-seeded/build2.log`). An initial fresh-cache attempt stalled resolving the Meta Wearables package; the seeded attempt first found a missing DAT import in the new Stop leaf, fixed by using `CaptureClaim.ending`, then passed. Xcode reported pre-existing Swift concurrency warnings elsewhere.
- Generic iOS `build-for-testing`: `** TEST BUILD SUCCEEDED **` (`/private/tmp/glasses-codex-u09-seeded/build-for-testing.log`). This compiles the changed unit and UI test targets but does not execute them.
- Independent CODEX-ACTING static review found that the initial-viewport lead assertion could pass with the text scrolled above the window. A follow-up now requires the lead to be hittable and its full frame to begin inside the window; the AX5 Stop test also checks singularity and reachability after scrolling to the bottom. Generic iOS `build-for-testing` passed again (`/private/tmp/glasses-codex-u09-seeded/build-for-testing-f2.log`). These assertions remain unexecuted.
- Swift unit/UI execution and screenshots: unavailable in this sandbox because `xcrun simctl list devices available` cannot connect to CoreSimulatorService (`Connection refused`). The new XCTest assertions are unexecuted.
- Required visual follow-up on a working Mac: 17 Pro and iPhone SE, default and AX5, light/dark as time permits. Capture with the World Builder scroll at top and at canvas bottom; verify Stop is visible, tappable, singular for VoiceOver, and does not cover the canvas or final footnote. Check running, paused, and stopping transitions. For all four onboarding cards, capture the initial viewport, then drag in the card center to check every point and the second card's Settings and Connections links. Verify VoiceOver reading order from title through points to footer and to the persistent Stop.

The existing U0.9 tour has no SE manifest or `INDEX.md`; its AX5 viewer test failed, so it is not a complete UX sign-off. The tour used right-edge drags for onboarding and produced no follow-on card shots; that alone does not establish that the lower content is unreachable.

## Isolation note

The canonical `.git` directory is sandbox read-only, so direct `git worktree add` failed before creating a branch. This is a linked worktree under `/private/tmp/Glasses-codex-u09-stop-onboarding`, registered to a separate local bare clone at `/private/tmp/Glasses-codex-u09-base.git`. Its branch must be fetched or bundled into the canonical repository by an authorized owner before temporary files are removed.
