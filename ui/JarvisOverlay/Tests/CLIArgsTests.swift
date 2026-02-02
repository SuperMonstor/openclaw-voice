import XCTest
@testable import JarvisOverlay

final class CLIArgsTests: XCTestCase {
    func testParseState() {
        let args = CLIArgs.parse(from: ["JarvisOverlay", "--state", "listening"])
        XCTAssertEqual(args.state, .listening)
        XCTAssertFalse(args.testCycle)
    }

    func testParseTestCycle() {
        let args = CLIArgs.parse(from: ["JarvisOverlay", "--test-cycle"])
        XCTAssertNil(args.state)
        XCTAssertTrue(args.testCycle)
    }

    func testParseSocketPath() {
        let args = CLIArgs.parse(from: ["JarvisOverlay", "--socket", "/tmp/test.sock"])
        XCTAssertEqual(args.socketPath, "/tmp/test.sock")
    }
}
