import AppKit
import SwiftUI

extension Notification.Name {
    static let overlayCloseRequested = Notification.Name("overlayCloseRequested")
}

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
    private var statusItem: NSStatusItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        viewModel.apply(args: CLIArgs.parse())
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleCloseRequested),
            name: .overlayCloseRequested,
            object: nil
        )
        window = createPanel()
        window?.makeKeyAndOrderFront(nil)
        configureStatusItem()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
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

    @objc private func handleCloseRequested() {
        viewModel.shutdownEngine()
        window?.orderOut(nil)
        window?.close()
        NSApp.terminate(nil)
    }

    @objc private func quitFromMenu() {
        handleCloseRequested()
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

    private func configureStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = statusIconImage()
        item.button?.imagePosition = .imageOnly
        item.menu = statusMenu()
        statusItem = item
    }

    private func statusMenu() -> NSMenu {
        let menu = NSMenu()
        let quitItem = NSMenuItem(
            title: "Quit Jarvis Overlay",
            action: #selector(quitFromMenu),
            keyEquivalent: "q"
        )
        quitItem.target = self
        menu.addItem(quitItem)
        return menu
    }

    private func statusIconImage() -> NSImage? {
        if let image = NSImage(named: NSImage.Name("StatusIcon")) {
            image.isTemplate = true
            return image
        }
        let fallback = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Jarvis")
        fallback?.isTemplate = true
        return fallback
    }
}
