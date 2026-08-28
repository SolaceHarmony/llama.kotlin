#if canImport(Testing)
import Testing
import Llama

@Suite("Llama Export Tests")
struct LlamaExportTests {
    @Test("Swift module loads and imports cleanly")
    func swiftModuleLoads() {
        #expect(Bool(true))
    }
}
#else
import XCTest
import Llama

final class LlamaExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "Llama swift module imported cleanly")
    }
}
#endif
