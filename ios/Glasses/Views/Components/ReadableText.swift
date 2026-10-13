//
//  ReadableText.swift
//  Glasses
//

import SwiftUI
import UIKit

/// Secondary text that stays readable: at least 4.5:1 against every surface
/// it sits on, in light and in dark.
///
/// ## Why not `.secondary` and `.tertiary`
///
/// The system's secondary label is 60 % of its label colour, which is about
/// 3.5:1 on a white card in light mode, and the tertiary label is 30 %, about
/// 1.8:1 in light and 2.6:1 on a dark card. The accessibility audit reports
/// the first as "nearly passed" and the second as failed, and the UX audit
/// found the grey world and session ids and the short-stretch footers "close
/// to invisible" in dark mode. Captions, ids and footers are text a person has
/// to read, so they are held to the 4.5:1 that WCAG sets for body text.
///
/// ## The two values, and their margins
///
/// - **Light:** `#636368`. 6.0:1 on white (a card), 5.4:1 on the grouped page
///   background `#F2F2F7`, and 5.2:1 on a tertiary-fill capsule.
/// - **Dark:** `#AAAAB0`. 7.3:1 on the dark card `#1C1C1E`, 9:1 on black, and
///   5.5:1 on a tertiary-fill capsule over the card.
///
/// Both are further from the 4.5:1 line than the minimum, because the audit
/// measures rendered, anti-aliased glyphs, which read lighter than their
/// colour. With Increase Contrast on, the system label colour is used instead:
/// the person has asked for the most contrast there is.
///
/// Still visibly grey beside `.primary`, so the hierarchy between a title and
/// its caption survives; what is given up is only the third, fainter level,
/// which carried text too faint to read.
extension UIColor {
    static let readableSecondary = UIColor { traits in
        if traits.accessibilityContrast == .high {
            return .label
        }
        return traits.userInterfaceStyle == .dark
            ? UIColor(red: 0xAA / 255, green: 0xAA / 255, blue: 0xB0 / 255, alpha: 1)
            : UIColor(red: 0x63 / 255, green: 0x63 / 255, blue: 0x68 / 255, alpha: 1)
    }
}

extension Color {
    static let readableSecondary = Color(uiColor: .readableSecondary)
}

extension ShapeStyle where Self == Color {
    /// `.foregroundStyle(.readableSecondary)`, in place of `.secondary` and
    /// `.tertiary` for text. See `Color.readableSecondary`.
    static var readableSecondary: Color { Color.readableSecondary }
}

/// The tint for a `.bordered` button's label: a darker blue in light mode and
/// a lighter one in dark, so the label is at least 4.5:1 on its own tinted
/// capsule.
///
/// The system blue on the bordered style's grey capsule measured about 3.5:1
/// in light mode, and the audit failed it ("Open connections" on Home).
/// `#0050C0` is about 7:1 on white and 5:1 on its 15 % capsule over the
/// grouped background; `#6CB4FF` is about 6:1 on its capsule over a dark
/// card and 5:1 over the lighter card a sheet uses. Only for bordered
/// buttons: a filled (prominent) button keeps the system blue, which white
/// text needs.
extension UIColor {
    static let readableTint = UIColor { traits in
        if traits.accessibilityContrast == .high {
            return traits.userInterfaceStyle == .dark
                ? UIColor(red: 0x9C / 255, green: 0xCC / 255, blue: 1, alpha: 1)
                : UIColor(red: 0, green: 0x3C / 255, blue: 0x99 / 255, alpha: 1)
        }
        return traits.userInterfaceStyle == .dark
            ? UIColor(red: 0x6C / 255, green: 0xB4 / 255, blue: 1, alpha: 1)
            : UIColor(red: 0, green: 0x50 / 255, blue: 0xC0 / 255, alpha: 1)
    }
}

extension Color {
    static let readableTint = Color(uiColor: .readableTint)
}

/// A destructive action's label on a card: `#C4211B` in light (5.9:1 on
/// white) and `#FF6961` in dark (5:1 on a sheet's card). The system red is
/// about 3.6:1 and 4:1 there, and the audit found "Unregister from Meta AI"
/// short of 4.5:1 in both.
extension UIColor {
    static let readableDestructive = UIColor { traits in
        traits.userInterfaceStyle == .dark
            ? UIColor(red: 1, green: 0x69 / 255, blue: 0x61 / 255, alpha: 1)
            : UIColor(red: 0xC4 / 255, green: 0x21 / 255, blue: 0x1B / 255, alpha: 1)
    }
}

extension View {
    /// `.buttonStyle(.bordered)` with a label that stays readable on its
    /// capsule. See `UIColor.readableTint`.
    func readableBorderedButton() -> some View {
        buttonStyle(.bordered).tint(Color.readableTint)
    }
}

