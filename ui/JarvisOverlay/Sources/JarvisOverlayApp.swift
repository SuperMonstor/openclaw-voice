import AppKit
import SwiftUI

@main
struct JarvisOverlayApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSPanel?
    private let viewModel = OverlayViewModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        viewModel.apply(args: CLIArgs.parse())
        window = createPanel()
        window?.makeKeyAndOrderFront(nil)
    }

    private func createPanel() -> NSPanel {
        let panel = NSPanel(
            contentRect: defaultFrame(),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: false
        )
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hasShadow = true
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = false
        panel.isMovableByWindowBackground = true

        let hosting = NSHostingView(rootView: ContentView(viewModel: viewModel))
        hosting.frame = panel.contentView?.bounds ?? .zero
        hosting.autoresizingMask = [.width, .height]
        panel.contentView = hosting
        return panel
    }

    private func defaultFrame() -> NSRect {
        let size = NSSize(width: 380, height: 160)
        guard let screen = NSScreen.main else {
            return NSRect(origin: .zero, size: size)
        }
        let frame = screen.visibleFrame
        let origin = NSPoint(
            x: frame.maxX - size.width - 24,
            y: frame.maxY - size.height - 24
        )
        return NSRect(origin: origin, size: size)
    }
}
